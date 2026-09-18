#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys
import time
from typing import Dict, List

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sparse_pod_sim2real.data.manifest import (
    compute_temporal_blocks,
    compute_sha256,
    validate_manifest_set,
    load_manifest,
    save_manifest,
)


def allocate_snapshot_quotas(candidate_counts: List[int], budget: int) -> List[int]:
    """Allocate a finite SVD budget across every non-empty trajectory fairly."""
    if budget <= 0:
        raise ValueError("Snapshot budget must be positive.")
    active = [index for index, count in enumerate(candidate_counts) if count > 0]
    if not active:
        return [0] * len(candidate_counts)
    if budget < len(active):
        raise ValueError(
            f"max_snapshots={budget} is smaller than the {len(active)} non-empty trajectories; "
            "increase the budget so every Sim trajectory contributes to the POD basis."
        )

    quotas = [0] * len(candidate_counts)
    remaining = min(budget, sum(candidate_counts))
    open_indices = active[:]
    while remaining > 0 and open_indices:
        share = max(1, remaining // len(open_indices))
        next_open = []
        for index in open_indices:
            capacity = candidate_counts[index] - quotas[index]
            take = min(share, capacity, remaining)
            quotas[index] += take
            remaining -= take
            if quotas[index] < candidate_counts[index]:
                next_open.append(index)
            if remaining == 0:
                break
        open_indices = next_open
    return quotas


def main():
    parser = argparse.ArgumentParser(
        description="Extract centered POD modes from balanced snapshots (Dev train blocks or Final full temporal ranges)"
    )
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/foil/tensor_cache_64x128/numerical"))
    parser.add_argument("--manifest", type=Path, default=None, help="Path to sim_source_manifest.json")
    parser.add_argument("--stage", type=str, choices=["dev", "final"], default="final", help="dev samples train blocks; final samples complete temporal ranges")
    parser.add_argument("--output-file", type=Path, default=Path("data/foil/pod_basis_64x128.pt"))
    parser.add_argument("--rank", type=int, default=64, help="Number of POD spatial modes (K) to retain")
    parser.add_argument("--perp-rank", type=int, default=16, help="Number of orthogonal complement modes (r) for Grassmann adaptation")
    parser.add_argument("--stride", type=int, default=5, help="Temporal sampling stride across frames")
    parser.add_argument("--max-snapshots", type=int, default=5000, help="Max snapshots for SVD")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train block ratio when splitting dynamically")
    parser.add_argument("--in-step", type=int, default=20)
    parser.add_argument("--out-step", type=int, default=20)
    parser.add_argument("--verify-checksums", action="store_true", help="Perform full SHA256 verification")
    args = parser.parse_args()

    tensor_dir = args.tensor_dir
    if not tensor_dir.exists():
        candidates = [
            Path("data/tensor_cache_64x128/numerical"),
            Path("data/foil/tensor_cache_64x128/numerical"),
            Path("../POD-Sim2Real/data/tensor_cache_64x128/numerical"),
            Path("artifacts/tensor_cache_64x128/numerical"),
        ]
        for c in candidates:
            if c.exists() and list(c.glob("*.pt")):
                tensor_dir = c
                break

    if not tensor_dir.exists():
        print(f"Error: Tensor directory {args.tensor_dir} not found.")
        sys.exit(1)

    all_files = sorted(list(tensor_dir.glob("*.pt")))
    if not all_files:
        print(f"No .pt files found in {tensor_dir}")
        sys.exit(1)

    # Load or build manifest
    manifest_data = None
    manifest_hash = None
    frame_ranges: Dict[str, List[int]] = {}

    if args.manifest and args.manifest.exists():
        manifest_data = load_manifest(args.manifest)
        manifest_hash = compute_sha256(args.manifest)
        expected_stems = set(manifest_data.get("trajectories", {}).keys())
        discovered_files = validate_manifest_set(
            expected_ids=expected_stems,
            discovered_files=all_files,
            verify_checksums=args.verify_checksums,
            recorded_metadata=manifest_data.get("trajectories", {}),
        )
        train_files = discovered_files
        for stem, info in manifest_data.get("trajectories", {}).items():
            if args.stage == "dev":
                frame_ranges[stem] = info.get("train_range", [0, int(info["num_frames"] * args.train_ratio)])
            else:
                frame_ranges[stem] = [0, info["num_frames"]]
    else:
        train_files = all_files
        for f in train_files:
            # Inspect length using mmap
            t_data = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
            t_len = t_data.shape[0] if isinstance(t_data, torch.Tensor) else t_data["tensor"].shape[0]
            if args.stage == "dev":
                (t0, t1), _ = compute_temporal_blocks(t_len, train_ratio=args.train_ratio, in_step=args.in_step, out_step=args.out_step)
                frame_ranges[f.stem] = [t0, t1]
            else:
                frame_ranges[f.stem] = [0, t_len]

    print("=" * 80)
    print(f"Computing Centered POD Modes [{args.stage.upper()} Stage] from {len(train_files)} Sim Trajectories")
    print(f"Target Rank K: {args.rank}, Orthogonal Complement Rank r: {args.perp_rank}, Stride: {args.stride}")
    print(f"Sampling Mode: {'Train-block temporal ranges (Dev Run)' if args.stage == 'dev' else 'Complete temporal ranges of all Sim trajectories (Final Run)'}")
    print("=" * 80)

    candidate_counts = []
    for f in train_files:
        start, end = frame_ranges[f.stem]
        candidate_counts.append(max(0, (int(end) - int(start) + args.stride - 1) // args.stride))
    snapshot_quotas = allocate_snapshot_quotas(candidate_counts, args.max_snapshots)

    snapshots = []
    sampled_counts: Dict[str, int] = {}
    total_snaps = 0
    h, w = None, None

    for f, quota in tqdm(list(zip(train_files, snapshot_quotas)), desc="Gathering snapshots"):
        if quota <= 0:
            continue
        # Load mmap to slice frames without duplicating full tensor in memory
        tensor = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
        if isinstance(tensor, dict):
            tensor = tensor.get("tensor", tensor.get("data"))

        # Permute if necessary to (T, 2, H, W)
        if tensor.shape[-1] == 2:
            tensor = tensor.permute(0, 3, 1, 2)

        t_total, c, h, w = tensor.shape
        t_start, t_end = frame_ranges[f.stem]
        t_start = max(0, min(t_start, t_total))
        t_end = max(t_start, min(t_end, t_total))

        candidate_indices = torch.arange(t_start, t_end, args.stride, dtype=torch.long)
        if candidate_indices.numel() == 0:
            continue
        if candidate_indices.numel() > quota:
            positions = torch.linspace(
                0, candidate_indices.numel() - 1, steps=quota
            ).round().long().unique(sorted=True)
            candidate_indices = candidate_indices.index_select(0, positions)
        sampled = tensor.index_select(0, candidate_indices)

        # Flatten: (T_sub, 2*H*W)
        flat = sampled.reshape(sampled.shape[0], -1).float()
        snapshots.append(flat)
        total_snaps += flat.shape[0]
        sampled_counts[f.stem] = int(flat.shape[0])

    if not snapshots:
        raise RuntimeError("No snapshots collected from trajectories.")

    X = torch.cat(snapshots, dim=0).T.float()  # (M, N_snapshots) where M = 2*H*W
    print(f"Snapshot matrix shape: {X.shape[0]} x {X.shape[1]}")

    # 1. Compute Base Mean Flow (Reynolds Decomposition)
    mean_flow = X.mean(dim=1, keepdim=True)  # (M, 1)
    X_fluc = X - mean_flow                   # (M, N_snapshots) fluctuating velocity
    print(f"Computed mean base flow: mean norm = {torch.norm(mean_flow).item():.4f}")

    t0 = time.time()
    print("Executing SVD on Reynolds fluctuating flow fields...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_dev = X_fluc.to(device)
    U, S, _ = torch.linalg.svd(X_dev, full_matrices=False)
    elapsed = time.time() - t0

    total_rank = args.rank + args.perp_rank
    if U.shape[1] < total_rank:
        raise RuntimeError(
            f"POD requests rank+perp_rank={total_rank}, but only {U.shape[1]} centered "
            "snapshots/modes are available. Reduce the ranks or increase --max-snapshots."
        )
    U_k = U[:, :args.rank].cpu()
    U_perp = U[:, args.rank:total_rank].cpu()
    S_k = S[:args.rank].cpu()

    var_total = torch.sum(S ** 2).item()
    var_k = torch.sum(S_k ** 2).item()
    energy_ratio = var_k / max(1e-12, var_total)

    print(f"SVD completed in {elapsed:.2f}s!")
    print(f"Top {args.rank} fluctuating modes capture {energy_ratio * 100:.2f}% of fluctuation energy.")

    manifest = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage,
        "manifest_path": str(args.manifest) if args.manifest else None,
        "manifest_hash": manifest_hash,
        "total_trajectories": len(train_files),
        "train_trajectories": [f.stem for f in train_files],
        "frame_ranges": frame_ranges,
        "total_snapshots": total_snaps,
        "sampled_snapshots_per_trajectory": sampled_counts,
        "snapshot_sampling_strategy": "balanced_across_trajectories_then_uniform_in_time",
        "k": args.rank,
        "r": args.perp_rank,
        "energy_ratio": energy_ratio,
        "grid_shape": [h, w],
        "stride": args.stride,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "basis": U_k,                         # [2*M, K]
        "basis_perp": U_perp,                 # [2*M, r]
        "mean": mean_flow.squeeze().cpu(),    # [2*M]
        "singular_values": S[:total_rank].cpu(),
        "energy_ratio": energy_ratio,
        "grid_shape": [h, w],
        "rank": args.rank,
        "perp_rank": args.perp_rank,
        "manifest": manifest,
    }
    temporary_basis = args.output_file.with_name(
        f".{args.output_file.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(payload, temporary_basis)
        os.replace(temporary_basis, args.output_file)
    finally:
        temporary_basis.unlink(missing_ok=True)

    manifest_path = args.output_file.with_name(args.output_file.stem + "_manifest.json")
    save_manifest(manifest, manifest_path)

    print(f"Saved centered POD basis to: {args.output_file}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
