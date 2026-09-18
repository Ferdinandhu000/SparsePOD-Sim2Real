#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparse_pod_sim2real.data.manifest import (
    compute_sha256,
    inspect_trajectory_info,
    compute_temporal_blocks,
    assert_split_disjointness,
    build_nested_few_shot_subsets,
    stratified_physical_split,
    save_manifest,
)
from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement


def build_sensor_manifests(
    manifest_dir: Path,
    sim_manifest_hash: str,
    real_manifest_hash: str,
    pod_basis_path: Optional[Path],
    sensor_seed: int = 202,
) -> None:
    basis_payload = None
    basis_hash = None
    if pod_basis_path is not None:
        if not pod_basis_path.exists():
            raise FileNotFoundError(f"POD basis for sensor manifests not found: {pod_basis_path}")
        basis_payload = torch.load(pod_basis_path, map_location="cpu", weights_only=True)
        basis_hash = compute_sha256(pod_basis_path)

    for topo in ["wall", "wake_rake", "uniform", "random", "group_q_deim"]:
        group_selection_order = None
        if topo == "group_q_deim" and basis_payload is not None:
            max_op = get_sensor_placement(
                topology_type=topo,
                num_sensors=64,
                grid_shape=(64, 128),
                seed=sensor_seed,
                pod_basis=basis_payload,
            )
            group_selection_order = max_op.selection_order_1d
        for ps in [16, 32, 64]:
            if topo == "group_q_deim" and basis_payload is None:
                print("[WARN] Skipping group_q_deim sensor manifests until --pod-basis is supplied.")
                break
            op = get_sensor_placement(
                topology_type=topo,
                num_sensors=ps,
                grid_shape=(64, 128),
                seed=sensor_seed,
                pod_basis=basis_payload if topo == "group_q_deim" else None,
                sensor_indices=(
                    group_selection_order[:ps]
                    if group_selection_order is not None else None
                ),
            )
            sensor_data = {
                "topology_type": topo,
                "num_sensors": ps,
                "grid_shape": [64, 128],
                "flatten_order": "C",
                "channel_order": ["u", "v"],
                "sensor_indices": op.indices_1d.tolist(),
                "sensor_coordinates": op.coords.tolist(),
                "sim_manifest_hash": sim_manifest_hash,
                "real_manifest_hash": real_manifest_hash,
                "basis_hash": basis_hash,
                "sensor_placement_seed": sensor_seed,
                "nested_prefix_of_p64": topo == "group_q_deim",
            }
            save_manifest(sensor_data, manifest_dir / f"sensor_manifest_{topo}_{ps}.json")


def build_manifests(
    data_root: Path,
    manifest_dir: Path,
    train_ratio: float = 0.8,
    real_train_ratio: float = 0.7,
    in_step: int = 20,
    out_step: int = 20,
    seeds: List[int] = [42, 101, 202, 303, 404],
    pod_basis_path: Optional[Path] = None,
    sensor_only: bool = False,
    sensor_seed: int = 202,
    metadata_dir: Optional[Path] = None,
    allow_fallback_splits: bool = False,
    expected_sim_count: Optional[int] = None,
    expected_real_count: Optional[int] = None,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if sensor_only:
        sim_path = manifest_dir / "sim_source_manifest.json"
        real_path = manifest_dir / "real_split_manifest.json"
        if not sim_path.exists() or not real_path.exists():
            raise FileNotFoundError("--sensor-only requires existing Sim and Real manifests.")
        build_sensor_manifests(
            manifest_dir,
            compute_sha256(sim_path),
            compute_sha256(real_path),
            pod_basis_path,
            sensor_seed,
        )
        print("[OK] Sensor manifest generation complete!")
        return
    print("=" * 80)
    print("BUILDING FORMAL RESEARCH PROVENANCE MANIFESTS")
    print(f"Data Root: {data_root}")
    print(f"Output Manifest Directory: {manifest_dir}")
    print("=" * 80)

    # 1. Inspect Sim trajectories
    sim_dir = data_root / "tensor_cache_64x128" / "numerical"
    if not sim_dir.exists():
        sim_dir = data_root / "foil" / "tensor_cache_64x128" / "numerical"
    if not sim_dir.exists():
        sim_dir = Path("data/foil/tensor_cache_64x128/numerical")

    sim_files = sorted(list(sim_dir.glob("*.pt"))) if sim_dir.exists() else []
    print(f"Found {len(sim_files)} Sim trajectory files in {sim_dir}")
    if expected_sim_count is not None and len(sim_files) != expected_sim_count:
        raise RuntimeError(
            f"Expected exactly {expected_sim_count} Sim trajectories, found {len(sim_files)}."
        )

    sim_manifest: Dict[str, Any] = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": f"Simulation source pool containing all {len(sim_files)} available trajectories partitioned into train and validation blocks with safety gap.",
        "num_trajectories": len(sim_files),
        "in_step": in_step,
        "out_step": out_step,
        "gap": in_step + out_step - 1,
        "trajectories": {},
    }

    for f in sim_files:
        info = inspect_trajectory_info(f)
        info["sha256"] = compute_sha256(f)
        t_len = info["num_frames"]
        (t_train0, t_train1), (t_val0, t_val1) = compute_temporal_blocks(
            t_len, train_ratio=train_ratio, in_step=in_step, out_step=out_step
        )
        info["train_range"] = [t_train0, t_train1]
        info["val_range"] = [t_val0, t_val1]
        sim_manifest["trajectories"][f.stem] = info

    sim_manifest_path = manifest_dir / "sim_source_manifest.json"
    save_manifest(sim_manifest, sim_manifest_path)
    sim_manifest_hash = compute_sha256(sim_manifest_path)
    print(f"[OK] Saved Sim source manifest to {sim_manifest_path} (Hash: {sim_manifest_hash[:12]}...)")

    # 2. Inspect Real trajectories & partition
    real_dir = data_root / "tensor_cache_64x128" / "real"
    if not real_dir.exists():
        real_dir = data_root / "foil" / "tensor_cache_64x128" / "real"
    if not real_dir.exists():
        real_dir = Path("data/foil/tensor_cache_64x128/real")

    real_files = sorted(list(real_dir.glob("*.pt"))) if real_dir.exists() else []
    print(f"Found {len(real_files)} Real trajectory files in {real_dir}")
    if expected_real_count is not None and len(real_files) != expected_real_count:
        raise RuntimeError(
            f"Expected exactly {expected_real_count} Real trajectories, found {len(real_files)}."
        )

    # Discover official RealPDEBench test set indices if available
    def normalize_trajectory_id(raw: str) -> str:
        name = Path(raw).name
        return name[:-3] if name.endswith(".pt") else name

    metadata_roots = []
    if metadata_dir is not None:
        metadata_roots.append(metadata_dir)
    metadata_roots.extend([data_root, data_root / "foil"])
    id_names = ["in_dist_test_params_real.json", "subsonic_params_real.json"]
    ood_names = ["out_dist_test_params_real.json", "transonic_params_real.json"]
    remain_names = ["remain_params_real.json"]

    test_id_stems = set()
    test_ood_stems = set()
    remain_stems = set()
    remain_parameter_metadata: Dict[str, object] = {}

    for p in [root / name for root in metadata_roots for name in id_names]:
        if p.exists():
            with open(p, "r") as f:
                test_id_stems = {normalize_trajectory_id(k) for k in json.load(f).keys()}
            break

    for p in [root / name for root in metadata_roots for name in ood_names]:
        if p.exists():
            with open(p, "r") as f:
                test_ood_stems = {normalize_trajectory_id(k) for k in json.load(f).keys()}
            break

    for p in [root / name for root in metadata_roots for name in remain_names]:
        if p.exists():
            with open(p, "r") as f:
                raw_remain = json.load(f)
            if not isinstance(raw_remain, dict):
                raise ValueError(f"Expected a JSON object in {p}, got {type(raw_remain).__name__}.")
            remain_parameter_metadata = {
                normalize_trajectory_id(k): value for k, value in raw_remain.items()
            }
            remain_stems = set(remain_parameter_metadata)
            break

    all_real_stems = set([f.stem for f in real_files])

    official_split_metadata = bool(test_id_stems and test_ood_stems and remain_stems)
    if not official_split_metadata:
        if not allow_fallback_splits:
            raise FileNotFoundError(
                "The complete official RealPDEBench remain/ID/OOD parameter JSON set was not found. "
                "Pass --metadata-dir; use --allow-fallback-splits only for a local smoke test."
            )
        # If running in local small subset mode or no metadata files present:
        # Deterministically partition all_real_stems into remain (train/val), test_id, test_ood
        sorted_stems = sorted(list(all_real_stems))
        n = len(sorted_stems)
        n_id = max(1, int(n * 0.15))
        n_ood = max(1, int(n * 0.15))
        n_remain = max(1, n - n_id - n_ood)

        remain_stems = set(sorted_stems[:n_remain])
        test_id_stems = set(sorted_stems[n_remain : n_remain + n_id])
        test_ood_stems = set(sorted_stems[n_remain + n_id :])
    else:
        official_union = remain_stems | test_id_stems | test_ood_stems
        missing_files = official_union - all_real_stems
        unassigned_files = all_real_stems - official_union
        if missing_files or unassigned_files:
            raise RuntimeError(
                "Official Real split metadata does not match available tensors. "
                f"missing={sorted(missing_files)[:5]}, unassigned={sorted(unassigned_files)[:5]}"
            )
        assert_split_disjointness(remain_stems, set(), test_id_stems, test_ood_stems)

    # Intersect with available files
    remain_available = sorted(list(remain_stems.intersection(all_real_stems)))
    test_id_available = sorted(list(test_id_stems.intersection(all_real_stems)))
    test_ood_available = sorted(list(test_ood_stems.intersection(all_real_stems)))

    # Deterministic joint Re/AoA-stratified split of the official remain pool.
    real_train_pool, real_val_pool, physical_conditions = stratified_physical_split(
        remain_available,
        train_ratio=real_train_ratio,
        seed=42,
        metadata=remain_parameter_metadata,
    )

    # Assert pairwise disjointness
    assert_split_disjointness(
        real_train=set(real_train_pool),
        real_val=set(real_val_pool),
        test_id=set(test_id_available),
        test_ood=set(test_ood_available),
    )

    # Build nested few-shot subsets per seed
    few_shot_subsets: Dict[int, Dict[int, List[str]]] = {}
    for s in seeds:
        few_shot_subsets[s] = build_nested_few_shot_subsets(
            train_pool=real_train_pool,
            seed=s,
            k_values=(1, 3, 5) if len(real_train_pool) >= 5 else tuple(range(1, len(real_train_pool) + 1)),
        )

    real_manifest: Dict[str, Any] = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": (
            "Official remain pool deterministically partitioned into train (70%) and val (30%) "
            "with joint Re/AoA quantile stratification; official subsonic ID and transonic OOD tests."
            if official_split_metadata else
            "Local smoke-test fallback split; not valid for formal benchmark reporting."
        ),
        "official_split_metadata": official_split_metadata,
        "split_method": (
            "joint_re_aoa_quantile_stratified"
            if official_split_metadata else "local_smoke_fallback"
        ),
        "split_seed": 42,
        "physical_conditions": physical_conditions,
        "real_train_pool": real_train_pool,
        "real_val_pool": real_val_pool,
        "test_id_pool": test_id_available,
        "test_ood_pool": test_ood_available,
        "nested_few_shot_subsets": few_shot_subsets,
    }

    real_manifest_path = manifest_dir / "real_split_manifest.json"
    save_manifest(real_manifest, real_manifest_path)
    real_manifest_hash = compute_sha256(real_manifest_path)
    print(f"[OK] Saved Real split manifest to {real_manifest_path} (Hash: {real_manifest_hash[:12]}...)")
    print(f"     Real Train: {len(real_train_pool)}, Real Val: {len(real_val_pool)}, ID Test: {len(test_id_available)}, OOD Test: {len(test_ood_available)}")

    # 3. Build canonical sensor manifests for primary topologies
    build_sensor_manifests(
        manifest_dir,
        sim_manifest_hash,
        real_manifest_hash,
        pod_basis_path,
        sensor_seed,
    )

    print("[OK] Manifest generation complete!")


def main():
    parser = argparse.ArgumentParser(description="Generate formal provenance manifests")
    parser.add_argument("--data-root", type=Path, default=Path("data/foil"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--pod-basis", type=Path, default=None)
    parser.add_argument("--sensor-only", action="store_true")
    parser.add_argument("--sensor-seed", type=int, default=202)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--allow-fallback-splits", action="store_true")
    parser.add_argument("--expected-sim-count", type=int, default=None)
    parser.add_argument("--expected-real-count", type=int, default=None)
    args = parser.parse_args()

    build_manifests(
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        pod_basis_path=args.pod_basis,
        sensor_only=args.sensor_only,
        sensor_seed=args.sensor_seed,
        metadata_dir=args.metadata_dir,
        allow_fallback_splits=args.allow_fallback_splits,
        expected_sim_count=args.expected_sim_count,
        expected_real_count=args.expected_real_count,
    )


if __name__ == "__main__":
    main()
