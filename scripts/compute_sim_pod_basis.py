#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract and save centered POD modes and orthogonal complement from simulation train split")
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/foil/tensor_cache_64x128/numerical"))
    parser.add_argument("--output-file", type=Path, default=Path("data/foil/pod_basis_64x128.pt"))
    parser.add_argument("--rank", type=int, default=64, help="Number of POD spatial modes (K) to retain")
    parser.add_argument("--perp-rank", type=int, default=16, help="Number of orthogonal complement modes (r) for Grassmann adaptation")
    parser.add_argument("--stride", type=int, default=5, help="Temporal sampling stride across frames")
    parser.add_argument("--max-snapshots", type=int, default=5000, help="Max snapshots for SVD")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of simulation trajectories considered as train split")
    args = parser.parse_args()

    if not args.tensor_dir.exists():
        print(f"Error: Tensor directory {args.tensor_dir} not found.")
        return

    all_files = sorted(list(args.tensor_dir.glob("*.pt")))
    if not all_files:
        print(f"No .pt files found in {args.tensor_dir}")
        return

    # Strictly filter Sim Train trajectories only (prevent test leakage into basis)
    n_train = max(1, int(len(all_files) * args.train_ratio))
    train_files = all_files[:n_train]

    print("=" * 80)
    print(f"Computing Centered POD Modes from {len(train_files)} Sim-Train Trajectories (out of {len(all_files)})")
    print(f"Target Rank K: {args.rank}, Orthogonal Complement Rank r: {args.perp_rank}, Stride: {args.stride}")
    print("=" * 80)

    snapshots = []
    total_snaps = 0

    for f in tqdm(train_files, desc="Gathering snapshots"):
        tensor = torch.load(f, map_location="cpu")  # (T, 2, H, W)
        t, c, h, w = tensor.shape
        # Subsample frames
        sampled = tensor[::args.stride]  # (T_sub, 2, H, W)
        # Flatten: (T_sub, 2*H*W)
        flat = sampled.reshape(sampled.shape[0], -1)
        snapshots.append(flat)
        total_snaps += flat.shape[0]
        if total_snaps >= args.max_snapshots:
            break

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
        "total_trajectories": len(all_files),
        "train_trajectories": [f.name for f in train_files],
        "total_snapshots": total_snaps,
        "k": args.rank,
        "r": args.perp_rank,
        "energy_ratio": energy_ratio,
        "grid_shape": [h, w],
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": U_k,                         # [2*M, K]
        "basis_perp": U_perp,                 # [2*M, r]
        "mean": mean_flow.squeeze().cpu(),    # [2*M]
        "singular_values": S[:total_rank].cpu(),
        "energy_ratio": energy_ratio,
        "grid_shape": [h, w],
        "rank": args.rank,
        "perp_rank": args.perp_rank,
        "manifest": manifest,
    }, args.output_file)

    manifest_path = args.output_file.with_name(args.output_file.stem + "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Saved centered POD basis to: {args.output_file}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()

