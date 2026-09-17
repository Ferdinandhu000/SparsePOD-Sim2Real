#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from sparse_pod_sim2real.data.preprocess import preprocess_domain


def main():
    parser = argparse.ArgumentParser(description="Preprocess RealPDEBench raw data into fast downsampled 64x128 .pt files")
    parser.add_argument("--data-root", type=Path, default=Path("data/foil"), help="Root path of foil dataset (e.g. data/foil)")
    parser.add_argument("--real-dir", type=Path, default=None, help="Path to real data directory (arrow or h5)")
    parser.add_argument("--sim-dir", type=Path, default=None, help="Path to numerical/sim data directory (arrow or h5)")
    parser.add_argument("--resolution", nargs=2, type=int, default=[64, 128], help="Target downsampled spatial resolution [H, W]")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output root directory for cached tensors")
    parser.add_argument("--prefix-frames", type=float, default=None, help="Optional frame cutoff (default None: all frames)")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing .pt files")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Worker processes")
    args = parser.parse_args()

    res = tuple(args.resolution)

    # 1. Resolve source directories
    real_dir = args.real_dir
    sim_dir = args.sim_dir
    if real_dir is None or sim_dir is None:
        candidates = [
            (args.data_root / "hf_dataset" / "real", args.data_root / "hf_dataset" / "sim"),
            (args.data_root / "hf_dataset" / "real", args.data_root / "hf_dataset" / "numerical"),
            (args.data_root / "data_real", args.data_root / "data_sim"),
            (args.data_root / "real", args.data_root / "numerical"),
            (Path("data/foil/hf_dataset/real"), Path("data/foil/hf_dataset/sim")),
            (Path("data/data_real"), Path("data/data_sim")),
            (Path("data/foil/real"), Path("data/foil/numerical")),
        ]
        for r_cand, s_cand in candidates:
            if r_cand.exists() and s_cand.exists():
                real_dir = real_dir or r_cand
                sim_dir = sim_dir or s_cand
                break

    if real_dir is None or not real_dir.exists():
        print(f"Notice: Real source directory not found at default locations. Please ensure data is placed in {args.data_root}")
        return
    if sim_dir is None or not sim_dir.exists():
        print(f"Notice: Sim source directory not found at default locations. Please ensure data is placed in {args.data_root}")
        return

    # 2. Resolve output directory
    if args.output_dir is not None:
        out_root = args.output_dir
    else:
        out_root = args.data_root / f"tensor_cache_{res[0]}x{res[1]}"

    real_out = out_root / "real"
    sim_out = out_root / "numerical"

    print("=" * 80)
    print("Precomputing Downsampled Trajectory Tensors for Fast In-Memory Training")
    print(f"Resolution:       {res[0]} x {res[1]}")
    print(f"Real Source:      {real_dir}")
    print(f"Sim Source:       {sim_dir}")
    print(f"Output Directory: {out_root}")
    print("=" * 80)

    t0 = time.perf_counter()
    real_meta = preprocess_domain(real_dir, real_out, res, args.prefix_frames, args.overwrite, args.num_workers, desc="Real Data")
    sim_meta = preprocess_domain(sim_dir, sim_out, res, args.prefix_frames, args.overwrite, args.num_workers, desc="Sim Data")
    elapsed = time.perf_counter() - t0

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": list(res),
        "real_count": len(real_meta),
        "sim_count": len(sim_meta),
        "real_trajectories": real_meta,
        "sim_trajectories": sim_meta,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Successfully processed {len(real_meta)} real and {len(sim_meta)} sim trajectories in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
