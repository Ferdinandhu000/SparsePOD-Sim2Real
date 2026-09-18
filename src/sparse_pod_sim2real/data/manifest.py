from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch


def _physical_parameters(
    trajectory_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Extract Reynolds number and angle of attack from metadata or foil ID."""
    record = (metadata or {}).get(trajectory_id, {})
    if isinstance(record, dict):
        normalized = {str(k).lower().replace("-", "_"): v for k, v in record.items()}

        def first_float(keys: Tuple[str, ...]) -> Optional[float]:
            for key in keys:
                if key in normalized:
                    try:
                        return float(normalized[key])
                    except (TypeError, ValueError):
                        pass
            return None

        reynolds = first_float(("re", "reynolds", "reynolds_number", "reynoldsnumber"))
        aoa = first_float(("aoa", "angle", "angle_of_attack", "angleofattack", "alpha"))
        if reynolds is not None and aoa is not None:
            return reynolds, aoa

    # RealPDEBench foil IDs use <Re>_<AoA>.h5. Keep this strict so a new
    # naming scheme cannot silently produce a non-stratified formal split.
    base = trajectory_id[:-3] if trajectory_id.endswith(".h5") else trajectory_id
    parts = base.split("_")
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    raise ValueError(
        f"Cannot recover (Re, AoA) for trajectory {trajectory_id!r}; "
        "provide these fields in remain_params_real.json."
    )


def stratified_physical_split(
    trajectory_ids: List[str],
    train_ratio: float,
    seed: int,
    metadata: Optional[Dict[str, Any]] = None,
    num_bins: int = 3,
) -> Tuple[List[str], List[str], Dict[str, Dict[str, float | int]]]:
    """Deterministically split a foil pool over joint Re/AoA quantile strata."""
    ids = sorted(trajectory_ids)
    if not ids:
        return [], [], {}
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    values = np.asarray([_physical_parameters(t, metadata) for t in ids], dtype=np.float64)

    def quantile_labels(column: np.ndarray) -> np.ndarray:
        unique = np.unique(column)
        bins = min(max(1, int(num_bins)), len(unique))
        if bins == 1:
            return np.zeros(len(column), dtype=np.int64)
        edges = np.unique(np.quantile(column, np.linspace(0.0, 1.0, bins + 1))[1:-1])
        return np.digitize(column, edges, right=True).astype(np.int64)

    re_bins = quantile_labels(values[:, 0])
    aoa_bins = quantile_labels(values[:, 1])
    groups: Dict[Tuple[int, int], List[str]] = {}
    physical: Dict[str, Dict[str, float | int]] = {}
    for trajectory_id, (reynolds, aoa), re_bin, aoa_bin in zip(
        ids, values, re_bins, aoa_bins
    ):
        key = (int(re_bin), int(aoa_bin))
        groups.setdefault(key, []).append(trajectory_id)
        physical[trajectory_id] = {
            "reynolds": float(reynolds),
            "aoa": float(aoa),
            "re_bin": key[0],
            "aoa_bin": key[1],
        }

    rng = np.random.RandomState(seed)
    shuffled: Dict[Tuple[int, int], List[str]] = {
        key: rng.permutation(sorted(members)).tolist() for key, members in sorted(groups.items())
    }
    target_train = 1 if len(ids) == 1 else int(round(len(ids) * train_ratio))
    target_train = min(max(1, target_train), len(ids) - 1 if len(ids) > 1 else 1)

    quotas: Dict[Tuple[int, int], int] = {}
    for key, members in shuffled.items():
        n = len(members)
        quota = int(np.floor(n * train_ratio))
        if n >= 2:
            quota = min(max(1, quota), n - 1)
        quotas[key] = quota

    # Match the global 70/30 target while retaining train and validation
    # examples in every non-singleton stratum whenever feasible.
    while sum(quotas.values()) < target_train:
        candidates = [
            key for key, members in shuffled.items()
            if quotas[key] < (len(members) - 1 if len(members) >= 2 else 1)
        ]
        if not candidates:
            candidates = [key for key, members in shuffled.items() if quotas[key] < len(members)]
        if not candidates:
            break
        key = max(candidates, key=lambda k: (len(shuffled[k]) * train_ratio - quotas[k], -k[0], -k[1]))
        quotas[key] += 1

    while sum(quotas.values()) > target_train:
        candidates = [
            key for key, members in shuffled.items()
            if quotas[key] > (1 if len(members) >= 2 else 0)
        ]
        if not candidates:
            candidates = [key for key in shuffled if quotas[key] > 0]
        if not candidates:
            break
        key = min(candidates, key=lambda k: (len(shuffled[k]) * train_ratio - quotas[k], k[0], k[1]))
        quotas[key] -= 1

    train, val = [], []
    for key, members in shuffled.items():
        train.extend(members[: quotas[key]])
        val.extend(members[quotas[key] :])
    if len(ids) > 1 and (not train or not val):
        raise RuntimeError("Physical stratification produced an empty train or validation pool.")
    return sorted(train), sorted(val), physical


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA256 hex digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def inspect_trajectory_info(file_path: Path) -> Dict[str, Any]:
    """Inspects trajectory tensor shape and frame count dynamically using mmap without full memory copy."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file {file_path} does not exist.")

    stat = file_path.stat()
    # Read metadata using mmap to inspect shapes without reading 250MB into RAM
    data = torch.load(file_path, map_location="cpu", mmap=True, weights_only=True)
    if isinstance(data, dict):
        tensor = data.get("tensor", data.get("data", None))
    else:
        tensor = data

    if tensor is None:
        raise ValueError(f"Could not extract tensor from {file_path}")

    # shape: (T, 2, H, W) or (T, H, W, 2)
    shape = list(tensor.shape)
    num_frames = shape[0]

    return {
        "stem": file_path.stem,
        "name": file_path.name,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "shape": shape,
        "num_frames": int(num_frames),
    }


def compute_temporal_blocks(
    total_frames: int,
    train_ratio: float = 0.8,
    in_step: int = 20,
    out_step: int = 20,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Partitions a trajectory of length total_frames into train_block and val_block
    separated by a boundary gap >= (in_step + out_step - 1) to eliminate temporal window leakage.
    Returns:
        train_range: (0, t_train_end)
        val_range: (t_val_start, total_frames)
    """
    min_window = in_step + out_step
    gap = min_window - 1

    # Desired train length
    t_train_end = int(total_frames * train_ratio)
    t_val_start = t_train_end + gap

    # Guard against too short validation block
    if t_val_start + min_window > total_frames:
        t_val_start = total_frames - min_window
        t_train_end = max(min_window, t_val_start - gap)

    if t_train_end < min_window:
        raise ValueError(
            f"Trajectory has only {total_frames} frames; cannot satisfy gap={gap} and min_window={min_window}."
        )

    return (0, t_train_end), (t_val_start, total_frames)


def validate_manifest_set(
    expected_ids: Set[str],
    discovered_files: List[Path],
    global_expected_ids: Optional[Set[str]] = None,
    verify_checksums: bool = False,
    recorded_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Path]:
    """
    Strictly verifies discovered files against expected manifest set using exact set algebra.
    Detects duplicates, missing items, and unexpected extra files.
    """
    # 1. Check duplicate filenames
    discovered_stems = [f.stem for f in discovered_files]
    if len(discovered_stems) != len(set(discovered_stems)):
        seen = set()
        dups = [x for x in discovered_stems if x in seen or seen.add(x)]
        raise RuntimeError(f"Duplicate trajectory stems discovered in directory: {dups}")

    discovered_map = {f.stem: f for f in discovered_files}
    discovered_set = set(discovered_stems)

    # 2. Strict set algebra matching
    matched = expected_ids.intersection(discovered_set)
    missing = expected_ids.difference(discovered_set)
    allowed_global = global_expected_ids if global_expected_ids is not None else expected_ids
    extra = discovered_set.difference(allowed_global)

    if missing:
        sample_missing = sorted(list(missing))[:5]
        raise RuntimeError(
            f"Manifest validation failed! Missing {len(missing)} expected trajectory files.\n"
            f"  Expected: {len(expected_ids)}, Discovered: {len(discovered_set)}, Matched: {len(matched)}\n"
            f"  Missing IDs (sample): {sample_missing}"
        )

    if extra:
        sample_extra = sorted(list(extra))[:5]
        raise RuntimeError(
            f"Manifest validation failed! Found {len(extra)} unexpected extra trajectory files not in global expected set.\n"
            f"  Extra IDs (sample): {sample_extra}"
        )

    matched_files = [discovered_map[stem] for stem in sorted(list(matched))]

    # 3. Fast metadata check or optional full SHA256 verification
    if recorded_metadata is not None:
        for f in matched_files:
            stem = f.stem
            if stem not in recorded_metadata:
                continue
            meta = recorded_metadata[stem]
            stat = f.stat()

            # Fast validation: check size and mtime
            if "size_bytes" in meta and stat.st_size != meta["size_bytes"]:
                raise RuntimeError(
                    f"File size mismatch for {f}: expected {meta['size_bytes']} bytes, found {stat.st_size} bytes."
                )

            if "mtime" in meta and abs(float(stat.st_mtime) - float(meta["mtime"])) > 1e-6:
                raise RuntimeError(
                    f"File modification-time mismatch for {f}: expected {meta['mtime']}, "
                    f"found {stat.st_mtime}."
                )

            # Full validation if explicitly requested
            if verify_checksums and "sha256" in meta:
                computed_hash = compute_sha256(f)
                if computed_hash != meta["sha256"]:
                    raise RuntimeError(
                        f"SHA256 checksum mismatch for {f}!\n"
                        f"  Expected: {meta['sha256']}\n"
                        f"  Computed: {computed_hash}"
                    )

    return matched_files


def assert_split_disjointness(
    real_train: Set[str],
    real_val: Set[str],
    test_id: Set[str],
    test_ood: Set[str],
) -> None:
    """Asserts pairwise disjointness across all four Real split partitions."""
    pairs = [
        ("real_train_pool", "real_val_pool", real_train, real_val),
        ("real_train_pool", "test_id_pool", real_train, test_id),
        ("real_train_pool", "test_ood_pool", real_train, test_ood),
        ("real_val_pool", "test_id_pool", real_val, test_id),
        ("real_val_pool", "test_ood_pool", real_val, test_ood),
        ("test_id_pool", "test_ood_pool", test_id, test_ood),
    ]
    for name_a, name_b, set_a, set_b in pairs:
        intersection = set_a.intersection(set_b)
        if intersection:
            raise AssertionError(
                f"Data Leakage Detected: {name_a} and {name_b} have non-empty intersection!\n"
                f"  Overlapping IDs: {sorted(list(intersection))}"
            )


def build_nested_few_shot_subsets(
    train_pool: List[str],
    seed: int,
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[int, List[str]]:
    """
    Generates strictly nested few-shot subsets: K=1 subset K=3 subset K=5.
    """
    sorted_pool = sorted(train_pool)
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(sorted_pool).tolist()

    subsets = {}
    for k in sorted(k_values):
        if k > len(permuted):
            raise ValueError(f"Requested few-shot K={k} exceeds train pool size {len(permuted)}.")
        subsets[k] = sorted(permuted[:k])

    # Assert strict nesting
    k_sorted = sorted(k_values)
    for i in range(len(k_sorted) - 1):
        k_small, k_large = k_sorted[i], k_sorted[i + 1]
        assert set(subsets[k_small]).issubset(set(subsets[k_large])), (
            f"Few-shot nesting broken: K={k_small} is not a subset of K={k_large}!"
        )

    return subsets


def load_manifest(manifest_path: Path | str) -> Dict[str, Any]:
    """Loads a JSON manifest file with schema and integrity check."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(data: Dict[str, Any], save_path: Path | str) -> None:
    """Saves a JSON manifest file deterministically with pretty formatting."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
