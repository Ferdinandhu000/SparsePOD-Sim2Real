#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "src")

from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement
from sparse_pod_sim2real.model.our_model.gappy_solver import DifferentiableGappySolver
from sparse_pod_sim2real.training.losses import compute_vorticity_2d, relative_l2_per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-tensor", type=Path, default=Path(r"D:\hj\Y-3 S-1\POD-Sim2Real\data\tensor_cache_64x128\real\10000_15.0.h5.pt"))
    parser.add_argument("--pod-basis", type=Path, default=Path("artifacts/pod_basis_64x128.pt"))
    parser.add_argument("--num-sensors", type=int, default=64)
    parser.add_argument("--topology", type=str, default="wall", choices=["wall", "uniform", "wake_rake", "random"])
    parser.add_argument("--output-img", type=Path, default=Path("artifacts/gappy_pod_reconstruction_effect.png"))
    parser.add_argument("--artifact-copy", type=Path, default=Path(r"C:\Users\HJ000\.gemini\antigravity\brain\6ff22eeb-82f1-4f43-9929-95d5a86e01a1\gappy_pod_reconstruction_effect.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Gappy-POD Reconstruction Demo on device: {device}")

    # 1. Load Ground Truth Test Trajectory (Shape: T, 2, 64, 128)
    if not args.test_tensor.exists():
        raise FileNotFoundError(f"Test tensor not found: {args.test_tensor}")
    raw_traj = torch.load(args.test_tensor, map_location="cpu")  # (T, 2, H, W)
    if raw_traj.shape[1] == 2:
        # permute to (T, H, W, 2)
        traj = raw_traj.permute(0, 2, 3, 1).float()
    else:
        traj = raw_traj.float()
    t_total, h, w, c = traj.shape
    print(f"Loaded Real Flow Field Trajectory: shape {traj.shape}, total frames = {t_total}")

    # 2. Load Centered POD Basis
    if not args.pod_basis.exists():
        raise FileNotFoundError(f"POD basis not found: {args.pod_basis}")
    basis_payload = torch.load(args.pod_basis, map_location=device)
    pod_basis = basis_payload["basis"].to(device)
    mean_flow = basis_payload["mean"].to(device)
    k = pod_basis.shape[1]
    print(f"Loaded POD Basis: Rank K={k}, Mean Flow Norm={torch.norm(mean_flow).item():.4f}")

    # 3. Setup Sensor Topology Operator
    sensor_op = get_sensor_placement(topology_type=args.topology, num_sensors=args.num_sensors, grid_shape=(h, w))
    sensor_indices = sensor_op.indices_1d.to(device)

    # 4. Instantiate Gappy-POD Solver
    gappy_solver = DifferentiableGappySolver(
        pod_basis=pod_basis,
        sensor_indices=sensor_indices,
        mean_flow=mean_flow,
        h=h,
        w=w,
        reg_lambda=1e-3,
    ).to(device)

    # 5. Extract a representative 20-frame continuous temporal window
    t_start = 50  # sample frame in fully developed vortex shedding
    window_len = 20
    gt_window = traj[t_start : t_start + window_len].to(device)  # [20, H, W, 2]

    # Extract 64 sensor readings: [20, 64, 2]
    sensor_readings = sensor_op.extract_sensor_values(gt_window)  # [20, Ps, 2]
    print(f"Simulated {args.num_sensors} discrete sensor measurements: shape {sensor_readings.shape}")

    # Solve frame-by-frame via Gappy-POD
    with torch.no_grad():
        a_coeffs, recon_window = gappy_solver(sensor_readings.unsqueeze(0))
        recon_window = recon_window.squeeze(0)  # [20, H, W, 2]

    # Calculate Quantitative Reconstruction Metrics across the window
    rel_l2 = relative_l2_per_sample(recon_window, gt_window).mean().item()
    u_rmse = torch.sqrt(F.mse_loss(recon_window[..., 0], gt_window[..., 0])).item()
    v_rmse = torch.sqrt(F.mse_loss(recon_window[..., 1], gt_window[..., 1])).item()

    vort_gt = compute_vorticity_2d(gt_window)
    vort_recon = compute_vorticity_2d(recon_window)
    vort_rel_l2 = relative_l2_per_sample(vort_recon, vort_gt).mean().item()

    print("=" * 80)
    print(f"Gappy-POD Reconstruction Performance ({args.num_sensors} '{args.topology}' sensors):")
    print(f"  - Relative L2 Error:   {rel_l2 * 100:.2f}%")
    print(f"  - u-velocity RMSE:     {u_rmse:.5f}")
    print(f"  - v-velocity RMSE:     {v_rmse:.5f}")
    print(f"  - Vorticity Rel-L2:    {vort_rel_l2 * 100:.2f}%")
    print("=" * 80)

    # 6. Generate Publication-Quality Visualizations
    # We display 3 key representative frames: t = 0, t = 8, t = 16
    frame_indices = [0, 8, 16]
    fig, axes = plt.subplots(4, 3, figsize=(14, 11), dpi=220)

    # Airfoil contour polygon
    xc, yc, chord, t_rel = 24.5, 32.0, 35.0, 0.25
    x_foil = np.linspace(xc - chord / 2.0, xc + chord / 2.0, 200)
    s = np.clip((x_foil - (xc - chord / 2.0)) / chord, 0.0, 1.0)
    yt = 5.0 * t_rel * chord * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * (s ** 2) + 0.2843 * (s ** 3) - 0.1015 * (s ** 4))
    foil_upper = np.column_stack([x_foil, yc + yt])
    foil_lower = np.column_stack([x_foil[::-1], (yc - yt)[::-1]])
    foil_polygon = np.vstack([foil_upper, foil_lower])

    sensor_y = sensor_indices.cpu().numpy() // w
    sensor_x = sensor_indices.cpu().numpy() % w

    gt_cpu = gt_window.cpu().numpy()
    recon_cpu = recon_window.cpu().numpy()
    vort_gt_cpu = vort_gt.cpu().numpy()
    vort_recon_cpu = vort_recon.cpu().numpy()

    # Determine common colorbar limits
    speed_gt = np.sqrt(gt_cpu[..., 0]**2 + gt_cpu[..., 1]**2)
    speed_recon = np.sqrt(recon_cpu[..., 0]**2 + recon_cpu[..., 1]**2)
    speed_max = max(speed_gt.max(), speed_recon.max())
    speed_min = min(speed_gt.min(), speed_recon.min())

    row_titles = [
        "Ground Truth Flow (with 64 Sensors)",
        "Gappy-POD Reconstruction (Full Field)",
        "Pointwise Error |Truth - Recon|",
        "Vorticity Comparison (Recon)",
    ]

    for col_idx, f_idx in enumerate(frame_indices):
        t_label = f"Frame t = {t_start + f_idx}"

        # Row 1: Ground Truth
        ax1 = axes[0, col_idx]
        im1 = ax1.imshow(speed_gt[f_idx], cmap="turbo", vmin=speed_min, vmax=speed_max, origin="lower")
        patch1 = Polygon(foil_polygon, closed=True, facecolor="#212529", edgecolor="none", zorder=2)
        ax1.add_patch(patch1)
        ax1.scatter(sensor_x, sensor_y, c="#ffffff", edgecolors="#d90429", s=14, linewidths=0.8, zorder=4, label="Sensors")
        ax1.set_title(f"Ground Truth ({t_label})", fontsize=11, fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Row 2: Gappy-POD Reconstruction
        ax2 = axes[1, col_idx]
        im2 = ax2.imshow(speed_recon[f_idx], cmap="turbo", vmin=speed_min, vmax=speed_max, origin="lower")
        patch2 = Polygon(foil_polygon, closed=True, facecolor="#212529", edgecolor="none", zorder=2)
        ax2.add_patch(patch2)
        ax2.set_title(f"Gappy-POD Recon ({t_label})", fontsize=11, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Row 3: Absolute Error
        ax3 = axes[2, col_idx]
        err_map = np.abs(speed_gt[f_idx] - speed_recon[f_idx])
        im3 = ax3.imshow(err_map, cmap="inferno", vmin=0.0, vmax=0.15, origin="lower")
        patch3 = Polygon(foil_polygon, closed=True, facecolor="#212529", edgecolor="none", zorder=2)
        ax3.add_patch(patch3)
        frame_rel = np.linalg.norm(err_map) / (np.linalg.norm(speed_gt[f_idx]) + 1e-8)
        ax3.set_title(f"Pointwise Error (Rel-L2: {frame_rel*100:.1f}%)", fontsize=11, fontweight="bold")
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Row 4: Vorticity field of reconstruction
        ax4 = axes[3, col_idx]
        im4 = ax4.imshow(vort_recon_cpu[f_idx], cmap="coolwarm", vmin=-0.08, vmax=0.08, origin="lower")
        patch4 = Polygon(foil_polygon, closed=True, facecolor="#212529", edgecolor="none", zorder=2)
        ax4.add_patch(patch4)
        ax4.set_title(f"Reconstructed Vorticity (Wake)", fontsize=11, fontweight="bold")
        ax4.set_xticks([])
        ax4.set_yticks([])

    # Colorbars
    fig.subplots_adjust(right=0.88, hspace=0.28, wspace=0.15)
    cbar_ax1 = fig.add_axes([0.90, 0.54, 0.015, 0.38])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label("Velocity Magnitude ||u||", fontsize=10)

    cbar_ax2 = fig.add_axes([0.90, 0.30, 0.015, 0.18])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2)
    cbar2.set_label("Speed Error Magnitude", fontsize=10)

    cbar_ax3 = fig.add_axes([0.90, 0.06, 0.015, 0.18])
    cbar3 = fig.colorbar(im4, cax=cbar_ax3)
    cbar3.set_label("Vorticity ω (1/s)", fontsize=10)

    fig.suptitle(
        f"Sparse-to-Full Gappy-POD Reconstruction from {args.num_sensors} Sensors (0.78% Spatial Coverage)\n"
        f"Temporal Window Average: Rel-L2 = {rel_l2*100:.2f}%, Vorticity Rel-L2 = {vort_rel_l2*100:.2f}%, u-RMSE = {u_rmse:.4f}, v-RMSE = {v_rmse:.4f}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    args.output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_img, bbox_inches="tight")
    print(f"Saved visualization to: {args.output_img}")

    if args.artifact_copy:
        args.artifact_copy.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(args.output_img, args.artifact_copy)
        print(f"Copied visualization to conversation artifact directory: {args.artifact_copy}")

    plt.close()


if __name__ == "__main__":
    main()
