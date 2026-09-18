from __future__ import annotations

from collections import OrderedDict
from functools import partial
from pathlib import Path
import random
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Set, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .sensor_topology import get_sensor_placement
from .manifest import (
    compute_temporal_blocks,
    validate_manifest_set,
    load_manifest,
)

# Process-level LRU cache for mapped tensor handles per worker
_WORKER_MMAP_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()
_MAX_WORKER_CACHE: int = 4
_CACHE_HEADROOM_BYTES: int = 4 * 1024**3


def compute_mmap_cache_handles(
    trajectory_size_bytes: int,
    num_workers: int,
    available_memory_bytes: Optional[int] = None,
) -> int:
    """Size each worker's mmap LRU while preserving 4 GiB process headroom."""
    if trajectory_size_bytes <= 0:
        return 2
    if available_memory_bytes is None:
        try:
            import psutil

            available_memory_bytes = int(psutil.virtual_memory().available)
        except (ImportError, AttributeError):
            return 4
    usable = max(0, int(available_memory_bytes) - _CACHE_HEADROOM_BYTES)
    handles = usable // (int(trajectory_size_bytes) * max(1, int(num_workers)))
    return max(2, min(6, int(handles)))


def _set_worker_mmap_cache_size(max_handles: int) -> None:
    global _MAX_WORKER_CACHE
    _MAX_WORKER_CACHE = max(2, min(6, int(max_handles)))


def _get_worker_mmap_tensor(file_path: Path) -> torch.Tensor:
    """Retrieves mmap tensor handle from worker LRU cache without duplicating full tensor memory."""
    key = str(file_path.resolve())
    if key in _WORKER_MMAP_CACHE:
        _WORKER_MMAP_CACHE.move_to_end(key)
        return _WORKER_MMAP_CACHE[key]

    if len(_WORKER_MMAP_CACHE) >= _MAX_WORKER_CACHE:
        _WORKER_MMAP_CACHE.popitem(last=False)

    data = torch.load(file_path, map_location="cpu", mmap=True, weights_only=True)
    tensor = data if isinstance(data, torch.Tensor) else data["tensor"]
    _WORKER_MMAP_CACHE[key] = tensor
    return tensor


def worker_init_fn(worker_id: int, max_cache_handles: int = 4) -> None:
    """Initializes worker seed deterministically."""
    _set_worker_mmap_cache_size(max_cache_handles)
    # DataLoader already folds worker_id into torch.initial_seed().
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class TrajectoryLocalityBatchSampler(Sampler[List[int]]):
    """Shuffle training windows in short same-trajectory chunks for mmap locality."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        batch_size: int,
        seed: int,
        locality_batches: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.locality_batches = max(1, int(locality_batches))
        self.epoch = 0
        self.by_trajectory: Dict[str, List[int]] = {}
        for index, (stem, _) in enumerate(samples):
            self.by_trajectory.setdefault(stem, []).append(index)
        self.num_samples = len(samples)

    def __len__(self) -> int:
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.by_trajectory.values()
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        chunk_size = self.batch_size * self.locality_batches
        chunks: List[List[int]] = []
        for stem in sorted(self.by_trajectory):
            # Dataset indices are ordered by time within each trajectory. Keep
            # short runs contiguous so mmap reads are sequential, then shuffle
            # the runs globally to retain stochastic training order.
            indices = np.asarray(self.by_trajectory[stem], dtype=np.int64)
            chunks.extend(
                indices[start : start + chunk_size].tolist()
                for start in range(0, len(indices), chunk_size)
            )
        rng.shuffle(chunks)
        for chunk in chunks:
            for start in range(0, len(chunk), self.batch_size):
                yield chunk[start : start + self.batch_size]


def required_batch_fields(
    model_name: str,
    stage: str,
    supervision_mode: str,
    selection_strategy: str = "dense_val",
) -> Set[str]:
    """
    Declares the exact minimal set of batch fields required by model architecture and execution stage.
    Avoids assembling and stacking unneeded 4D/5D full-field tensors on CPU.
    """
    fields: Set[str] = {"traj_stem", "t_start"}
    lower_name = model_name.lower()
    compact_name = lower_name.replace("_", "")
    is_masked_grid_model = lower_name in {
        "masked_unet",
        "masked_unet3d",
        "masked_fno",
        "masked_fno3d",
    } or lower_name.startswith("masked_") or compact_name in {"maskedunet3d", "maskedfno3d"}
    is_misf = "misf" in lower_name

    # Dense Sim pre-training bypasses the sparse observation path. Masked
    # baselines construct a dense-with-mask input in the trainer from x_full.
    if stage == "sim_pretrain":
        fields.update({"x_full", "y_full"})
        return fields

    # Sparse model inputs used during Real adaptation/evaluation.
    if is_masked_grid_model:
        fields.add("x_sparse")
    else:
        fields.add("sensor_values")
        if is_misf:
            fields.add("sensor_coords")

    if stage == "real_finetune":
        if supervision_mode == "sparse_sensors":
            fields.add("y_sensor_values")
        else:
            fields.add("y_full")
    elif stage in ("val", "test", "dense_eval", "sensor_eval"):
        if stage == "sensor_eval" or selection_strategy == "sensor_val":
            fields.add("y_sensor_values")
        else:
            fields.add("y_full")

    return fields


class SelectiveBatchCollator:
    """Stacks only declared required fields into batch tensors, conserving CPU memory and bus bandwidth."""

    def __init__(self, required_fields: Set[str]):
        self.required_fields = set(required_fields)

    def __call__(self, batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch_list:
            return {}

        collated: Dict[str, Any] = {}
        keys = [k for k in batch_list[0].keys() if k in self.required_fields]

        for k in keys:
            elem = batch_list[0][k]
            if isinstance(elem, torch.Tensor):
                collated[k] = torch.stack([d[k] for d in batch_list], dim=0)
            elif isinstance(elem, (str, int, float)):
                collated[k] = [d[k] for d in batch_list]
            else:
                collated[k] = [d[k] for d in batch_list]

        return collated


class SparseTrajectoryDataset(Dataset):
    """
    Scalable dataset for fluid trajectories with zero-copy mmap slicing,
    strict fail-hard manifest validation, and non-overlapping evaluation accounting.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset_type: Literal["numerical", "real"] = "numerical",
        mode: Literal["train", "val", "test"] = "train",
        test_mode: Literal["all", "in_dist", "out_dist"] = "all",
        topology_type: str = "wall",
        num_sensors: int = 64,
        in_step: int = 20,
        out_step: int = 20,
        interval: Optional[int] = None,
        few_shot_k: Optional[int] = None,
        max_trajectories: Optional[int] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
        pod_basis: Optional[torch.Tensor | dict] = None,
        sparse_fill_method: str = "zero",
        manifest_dir: Optional[Path | str] = None,
        verify_checksums: bool = False,
        required_fields: Optional[Set[str]] = None,
        allow_manifest_fallback: bool = False,
        use_all_frames: bool = False,
        sensor_seed: Optional[int] = None,
        sensor_indices_override: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.dataset_type = dataset_type
        self.mode = mode
        self.test_mode = test_mode
        self.in_step = in_step
        self.out_step = out_step
        self.window_len = in_step + out_step
        self.few_shot_k = few_shot_k
        self.seed = seed
        self.sensor_seed = seed if sensor_seed is None else int(sensor_seed)
        self.sparse_fill_method = sparse_fill_method
        self.manifest_dir = Path(manifest_dir) if manifest_dir else self._find_manifest_dir()
        self.required_fields = set(required_fields) if required_fields is not None else {
            "traj_stem", "t_start", "x_sparse", "y_full", "y_sensor_values",
            "x_full", "sensor_values", "sensor_coords", "sensor_mask",
        }
        self.allow_manifest_fallback = bool(allow_manifest_fallback)
        self.use_all_frames = bool(use_all_frames)

        # Non-overlapping evaluation stride by default (stride = Tout)
        if interval is None:
            self.interval = out_step if mode in ("val", "test") else 5
        else:
            self.interval = interval

        # 1. Resolve tensor directory
        self.tensor_dir = self._find_tensor_dir()

        # 2. Setup sensor topology operator
        self.sensor_op = get_sensor_placement(
            topology_type=topology_type,
            num_sensors=num_sensors,
            grid_shape=(64, 128),
            seed=self.sensor_seed,
            pod_basis=pod_basis,
            sensor_indices=sensor_indices_override,
        )

        # 3. Resolve trajectory file list and temporal frame bounds with fail-hard validation
        self.trajectory_info: Dict[str, Tuple[Path, int, int]] = {}  # stem -> (path, t_start, t_end)
        self._resolve_trajectory_split(train_ratio, max_trajectories, verify_checksums)

        # 4. Precompute window index table and frame accounting
        self.samples: List[Tuple[str, int]] = []  # (stem, t_window_start)
        self.frame_accounting: Dict[str, int] = {
            "total_frames": 0,
            "context_frames": 0,
            "evaluated_target_frames": 0,
            "discarded_tail_frames": 0,
        }
        self._build_index()

    def _find_tensor_dir(self) -> Path:
        candidates = [
            self.data_root / "tensor_cache_64x128" / self.dataset_type,
            self.data_root / "foil" / "tensor_cache_64x128" / self.dataset_type,
            self.data_root / self.dataset_type,
            Path("data/foil/tensor_cache_64x128") / self.dataset_type,
            Path("data/tensor_cache_64x128") / self.dataset_type,
            Path("../POD-Sim2Real/data/tensor_cache_64x128") / self.dataset_type,
        ]
        for cand in candidates:
            if cand.exists() and list(cand.glob("*.pt")):
                return cand
        return self.data_root / "tensor_cache_64x128" / self.dataset_type

    def _find_manifest_dir(self) -> Path:
        candidates = [
            self.data_root / "manifests",
            self.data_root / "foil" / "manifests",
            Path("manifests"),
            Path("data/manifests"),
        ]
        for cand in candidates:
            if cand.exists() and (cand / "sim_source_manifest.json").exists():
                return cand
        return Path("manifests")

    def _resolve_trajectory_split(
        self,
        train_ratio: float,
        max_trajectories: Optional[int],
        verify_checksums: bool,
    ) -> None:
        if not self.tensor_dir.exists():
            raise FileNotFoundError(f"Tensor directory not found: {self.tensor_dir}")

        all_files = sorted(list(self.tensor_dir.glob("*.pt")))
        if not all_files:
            raise FileNotFoundError(f"No trajectory .pt files found in: {self.tensor_dir}")

        file_map = {f.stem: f for f in all_files}
        all_stems = set(file_map.keys())

        # Check for formal manifests
        sim_manifest_file = self.manifest_dir / "sim_source_manifest.json"
        real_manifest_file = self.manifest_dir / "real_split_manifest.json"

        if self.dataset_type == "numerical" and sim_manifest_file.exists():
            sim_m = load_manifest(sim_manifest_file)
            trajs = sim_m.get("trajectories", {})
            expected_ids = set(trajs.keys())
            matched_files = validate_manifest_set(
                expected_ids=expected_ids,
                discovered_files=all_files,
                verify_checksums=verify_checksums,
                recorded_metadata=trajs,
            )
            for f in matched_files:
                meta = trajs[f.stem]
                if self.mode == "train" and self.use_all_frames:
                    t_range = [0, int(meta["num_frames"])]
                else:
                    t_range = meta["train_range"] if self.mode == "train" else meta["val_range"]
                self.trajectory_info[f.stem] = (f, t_range[0], t_range[1])

        elif self.dataset_type == "real" and real_manifest_file.exists():
            real_m = load_manifest(real_manifest_file)
            if self.mode == "train":
                if self.few_shot_k is not None:
                    # Look up nested subsets
                    subsets = real_m.get("nested_few_shot_subsets", {}).get(str(self.seed), {})
                    if str(self.few_shot_k) in subsets:
                        target_stems = subsets[str(self.few_shot_k)]
                    else:
                        if not self.allow_manifest_fallback:
                            raise KeyError(
                                f"Formal Real manifest has no frozen few-shot subset for "
                                f"seed={self.seed}, K={self.few_shot_k}. Rebuild the manifest "
                                "with this seed/K instead of sampling during training."
                            )
                        rng = np.random.RandomState(self.seed)
                        pool = real_m["real_train_pool"]
                        target_stems = sorted(rng.choice(pool, size=min(self.few_shot_k, len(pool)), replace=False).tolist())
                else:
                    target_stems = real_m["real_train_pool"]
            elif self.mode == "val":
                target_stems = real_m["real_val_pool"]
            else:  # test
                if self.test_mode == "in_dist":
                    target_stems = real_m["test_id_pool"]
                elif self.test_mode == "out_dist":
                    target_stems = real_m["test_ood_pool"]
                else:
                    target_stems = real_m["test_id_pool"] + real_m["test_ood_pool"]

            target_set = set(target_stems)
            # Fail-hard verification
            missing = target_set.difference(all_stems)
            if missing:
                raise RuntimeError(
                    f"Fail-Hard Manifest Validation Failed! Missing Real trajectories: {sorted(list(missing))[:5]}"
                )

            for stem in target_stems:
                f = file_map[stem]
                # Inspect dynamic length
                t_tensor = _get_worker_mmap_tensor(f)
                t_len = t_tensor.shape[0]
                self.trajectory_info[stem] = (f, 0, t_len)

        else:
            if not self.allow_manifest_fallback:
                required_manifest = sim_manifest_file if self.dataset_type == "numerical" else real_manifest_file
                raise FileNotFoundError(
                    f"Formal run requires provenance manifest: {required_manifest}. "
                    "Set allow_manifest_fallback=true only for an explicit smoke test."
                )
            # Fallback deterministic partition when manifests are not pre-built
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(all_files))
            rng.shuffle(indices)

            n_train = int(len(all_files) * train_ratio)
            n_val = (len(all_files) - n_train) // 2

            if self.mode == "train":
                selected_idx = indices[:n_train]
            elif self.mode == "val":
                selected_idx = indices[n_train : n_train + n_val]
            else:  # test
                selected_idx = indices[n_train + n_val :]

            selected_files = [all_files[i] for i in selected_idx]

            if self.dataset_type == "real" and self.mode == "train" and self.few_shot_k is not None:
                if self.few_shot_k < len(selected_files):
                    chosen = rng.choice(len(selected_files), size=self.few_shot_k, replace=False)
                    selected_files = [selected_files[i] for i in sorted(chosen)]

            for f in selected_files:
                t_tensor = _get_worker_mmap_tensor(f)
                t_len = t_tensor.shape[0]
                if self.dataset_type == "numerical" and not self.use_all_frames:
                    (t0, t1), (v0, v1) = compute_temporal_blocks(t_len, train_ratio=train_ratio, in_step=self.in_step, out_step=self.out_step)
                    t_range = (t0, t1) if self.mode == "train" else (v0, v1)
                    self.trajectory_info[f.stem] = (f, t_range[0], t_range[1])
                else:
                    self.trajectory_info[f.stem] = (f, 0, t_len)

        if max_trajectories is not None and max_trajectories > 0:
            stems = sorted(list(self.trajectory_info.keys()))[:max_trajectories]
            self.trajectory_info = {s: self.trajectory_info[s] for s in stems}

    def _build_index(self) -> None:
        self.samples = []
        tot_frames = 0
        ctx_frames = 0
        eval_frames = 0
        tail_frames = 0

        for stem, (path, t_start_bound, t_end_bound) in self.trajectory_info.items():
            span = t_end_bound - t_start_bound
            tot_frames += span
            if span < self.window_len:
                tail_frames += span
                continue

            n_windows = (span - self.window_len) // self.interval + 1
            for w in range(n_windows):
                w_start = t_start_bound + w * self.interval
                self.samples.append((stem, w_start))

            # Accounting for evaluation windows
            if self.mode in ("val", "test") and n_windows > 0:
                ctx_frames += self.in_step
                eval_frames += n_windows * self.out_step
                last_end = t_start_bound + (n_windows - 1) * self.interval + self.window_len
                tail_frames += (t_end_bound - last_end)

        self.frame_accounting["total_frames"] = tot_frames
        self.frame_accounting["context_frames"] = ctx_frames
        self.frame_accounting["evaluated_target_frames"] = eval_frames
        self.frame_accounting["discarded_tail_frames"] = tail_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem, t_start = self.samples[idx]
        path, _, _ = self.trajectory_info[stem]

        # Retrieve mmap tensor from worker LRU
        full_traj = _get_worker_mmap_tensor(path)

        # Slice the mmap view. Dense HWC copies are created only for fields that
        # the current model/stage explicitly requested.
        window_raw = full_traj[t_start : t_start + self.window_len]
        raw_in = window_raw[: self.in_step]
        raw_out = window_raw[self.in_step : self.window_len]

        def to_hwc(raw: torch.Tensor) -> torch.Tensor:
            if raw.ndim != 4:
                raise ValueError(f"Expected a 4D trajectory slice, got {tuple(raw.shape)}")
            if raw.shape[1] == 2:
                return raw.permute(0, 2, 3, 1).contiguous().float()
            if raw.shape[-1] == 2:
                return raw.contiguous().float()
            raise ValueError(f"Cannot identify velocity channels in shape {tuple(raw.shape)}")

        def extract_sensors(raw: torch.Tensor) -> torch.Tensor:
            indices = self.sensor_op.indices_1d
            if raw.shape[1] == 2:
                flat = raw.reshape(raw.shape[0], 2, -1)
                return flat.index_select(2, indices).permute(0, 2, 1).contiguous().float()
            if raw.shape[-1] == 2:
                flat = raw.reshape(raw.shape[0], -1, 2)
                return flat.index_select(1, indices).contiguous().float()
            raise ValueError(f"Cannot identify velocity channels in shape {tuple(raw.shape)}")

        sample: Dict[str, Any] = {"traj_stem": stem, "t_start": t_start}
        if "x_full" in self.required_fields:
            sample["x_full"] = to_hwc(raw_in)
        if "y_full" in self.required_fields:
            sample["y_full"] = to_hwc(raw_out)
        if "sensor_values" in self.required_fields:
            sample["sensor_values"] = extract_sensors(raw_in)
        if "y_sensor_values" in self.required_fields:
            sample["y_sensor_values"] = extract_sensors(raw_out)
        if "x_sparse" in self.required_fields:
            input_sensors = extract_sensors(raw_in)
            sample["x_sparse"] = self.sensor_op.sensor_values_to_sparse_tensor(
                input_sensors, append_mask=True, fill_method=self.sparse_fill_method
            )
        if "sensor_coords" in self.required_fields:
            sample["sensor_coords"] = self.sensor_op.coords
        if "sensor_mask" in self.required_fields:
            sample["sensor_mask"] = self.sensor_op.mask
        return sample


def create_dataloaders(
    data_root: str | Path,
    config: Dict,
    pod_basis: Optional[torch.Tensor | dict] = None,
    stage: str = "train",
    include_test: bool = False,
    include_val: bool = True,
    sensor_indices_override: Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates memory-optimized Train and Val DataLoaders (and optionally Test Loader).
    Uses SelectiveBatchCollator to filter out unused fields before stacking.
    """
    stage_dataset_type = config.get("dataset_type", "numerical")
    val_dataset_type = config.get("val_dataset_type", stage_dataset_type)
    model_name = config.get("model_name", "")
    supervision_mode = config.get("supervision_mode", "sparse_sensors")

    selection_strategy = config.get("val_selection_strategy", "dense_val")
    train_fields = required_batch_fields(
        model_name, stage=stage, supervision_mode=supervision_mode,
        selection_strategy=selection_strategy,
    )
    # Development Sim validation must use the same dense input contract as
    # dense Sim pre-training. Treating it as a generic validation batch omits
    # x_full and silently pushes tolerant models through their sparse path,
    # making the selected S* incomparable to the source-training objective.
    if stage_dataset_type == "numerical" and stage == "sim_pretrain":
        val_stage = "sim_pretrain"
    else:
        val_stage = "sensor_eval" if selection_strategy == "sensor_val" else "val"
    val_fields = required_batch_fields(
        model_name, stage=val_stage, supervision_mode=supervision_mode,
        selection_strategy=selection_strategy,
    )
    dataset_seed = int(config.get("few_shot_traj_seed", config.get("seed", 42)))
    sensor_seed = int(config.get("sensor_placement_seed", config.get("seed", 42)))
    allow_fallback = bool(config.get("allow_manifest_fallback", config.get("smoke_test", False)))
    final_source = stage_dataset_type == "numerical" and config.get("sim_mode", "dev") == "final"
    num_workers = int(config.get("num_workers", 0))

    train_dataset = SparseTrajectoryDataset(
        data_root=data_root,
        dataset_type=stage_dataset_type,
        mode="train",
        topology_type=config.get("sensor_topology", "wall"),
        num_sensors=config.get("num_sensors", 64),
        in_step=config.get("in_step", 20),
        out_step=config.get("out_step", 20),
        interval=config.get("interval", 5),
        few_shot_k=config.get("few_shot_k", None),
        seed=dataset_seed,
        sensor_seed=sensor_seed,
        pod_basis=pod_basis,
        sparse_fill_method=config.get("sparse_fill_method", "zero"),
        manifest_dir=config.get("manifest_dir", None),
        verify_checksums=config.get("verify_checksums", False),
        required_fields=train_fields,
        allow_manifest_fallback=allow_fallback,
        use_all_frames=final_source,
        sensor_indices_override=sensor_indices_override,
    )
    trajectory_sizes = [p.stat().st_size for p in train_dataset.tensor_dir.glob("*.pt")]
    cache_handles = compute_mmap_cache_handles(
        max(trajectory_sizes, default=0),
        num_workers=max(1, num_workers),
    )
    train_dataset.mmap_cache_handles = cache_handles
    _set_worker_mmap_cache_size(cache_handles)
    seeded_worker_init = partial(worker_init_fn, max_cache_handles=cache_handles)

    val_dataset = None
    if include_val:
        val_dataset = SparseTrajectoryDataset(
            data_root=data_root,
            dataset_type=val_dataset_type,
            mode="val",
            topology_type=config.get("sensor_topology", "wall"),
            num_sensors=config.get("num_sensors", 64),
            in_step=config.get("in_step", 20),
            out_step=config.get("out_step", 20),
            interval=config.get("out_step", 20),
            seed=dataset_seed,
            sensor_seed=sensor_seed,
            pod_basis=pod_basis,
            sparse_fill_method=config.get("sparse_fill_method", "zero"),
            manifest_dir=config.get("manifest_dir", None),
            verify_checksums=config.get("verify_checksums", False),
            required_fields=val_fields,
            allow_manifest_fallback=allow_fallback,
            sensor_indices_override=sensor_indices_override,
        )

    batch_size = config.get("batch_size", 8)
    use_persistent = (num_workers > 0)

    generator = torch.Generator()
    generator.manual_seed(int(config.get("model_init_seed", config.get("seed", 42))))
    pin_memory = bool(config.get("pin_memory", torch.cuda.is_available()))

    locality_batches = int(config.get("trajectory_locality_batches", 8))
    if locality_batches > 0:
        train_batch_sampler = TrajectoryLocalityBatchSampler(
            train_dataset.samples,
            batch_size=batch_size,
            seed=int(config.get("model_init_seed", config.get("seed", 42))),
            locality_batches=locality_batches,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seeded_worker_init if num_workers > 0 else None,
            collate_fn=SelectiveBatchCollator(train_fields),
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seeded_worker_init if num_workers > 0 else None,
            collate_fn=SelectiveBatchCollator(train_fields),
            generator=generator,
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seeded_worker_init if num_workers > 0 else None,
            collate_fn=SelectiveBatchCollator(val_fields),
        )

    test_loader = None
    if include_test:
        test_fields = required_batch_fields(
            model_name, stage="test", supervision_mode=supervision_mode,
            selection_strategy="dense_val",
        )
        test_dataset = SparseTrajectoryDataset(
            data_root=data_root,
            dataset_type="real",
            mode="test",
            test_mode=config.get("test_mode", "all"),
            topology_type=config.get("sensor_topology", "wall"),
            num_sensors=config.get("num_sensors", 64),
            in_step=config.get("in_step", 20),
            out_step=config.get("out_step", 20),
            interval=config.get("out_step", 20),  # Stride = Tout
            seed=dataset_seed,
            sensor_seed=sensor_seed,
            pod_basis=pod_basis,
            sparse_fill_method=config.get("sparse_fill_method", "zero"),
            manifest_dir=config.get("manifest_dir", None),
            verify_checksums=config.get("verify_checksums", False),
            required_fields=test_fields,
            allow_manifest_fallback=allow_fallback,
            sensor_indices_override=sensor_indices_override,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seeded_worker_init if num_workers > 0 else None,
            collate_fn=SelectiveBatchCollator(test_fields),
        )

    return train_loader, val_loader, test_loader
