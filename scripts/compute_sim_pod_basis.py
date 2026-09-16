#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract and save POD/SVD spatial modes from numerical simulation tensors")
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/foil/tensor_cache_64x128/numerical"))
    parser.add_argument("--output-file", type=Path, default=Path("data/foil/pod_basis_64x128.pt"))
    parser.add_argument("--rank", type=int, default=64, help="Number of POD spatial modes to retain")
    parser.add_argument("--stride", type=int, default=5, help="Temporal sampling stride across frames")
    parser.add_argument("--max-snapshots", type=int, default=5000, help="Max snapshots for SVD")
    args = parser.parse_args()

    if not args.tensor_dir.exists():
        print(f"Error: Tensor directory {args.tensor_dir} not found.")
        return

    files = sorted(list(args.tensor_dir.glob("*.pt")))
    if not files:
        print(f"No .pt files found in {args.tensor_dir}")
        return

    print("=" * 80)
    print(f"Computing POD Spatial Modes from {len(files)} Simulation Trajectories")
    print(f"Target Rank: {args.rank}, Subsampling Stride: {args.stride}")
    print("=" * 80)

    snapshots = []
    total_snaps = 0

    for f in tqdm(files, desc="Gathering snapshots"):
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

    t0 = time.time()
    # Compute SVD: X = U * S * V^T
    print("Executing SVD...")
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_dev = X.to(device)
    U, S, _ = torch.linalg.svd(X_dev, full_matrices=False)
    elapsed = time.time() - t0

    U_k = U[:, :args.rank].cpu()
    S_k = S[:args.rank].cpu()

    var_total = torch.sum(S ** 2).item()
    var_k = torch.sum(S_k ** 2).item()
    energy_ratio = var_k / var_total

    print(f"SVD completed in {elapsed:.2f}s!")
    print(f"Top {args.rank} modes capture {energy_ratio * 100:.2f}% of system energy.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": U_k,
        "singular_values": S_k,
        "energy_ratio": energy_ratio,
        "grid_shape": [64, 128],
        "rank": args.rank,
    }, args.output_file)

    print(f"Saved POD basis to: {args.output_file}")


if __name__ == "__main__":
    main()
