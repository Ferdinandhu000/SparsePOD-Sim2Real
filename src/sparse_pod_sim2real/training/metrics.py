from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .losses import compute_vorticity_2d, relative_l2_per_sample


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Comprehensive evaluation metrics aligned with RealPDEBench standard:
    pred, target: [B, Tout, H, W, 2]
    """
    b = pred.size(0)

    # 1. Official Sample-wise Relative L2 Error
    rel_l2 = relative_l2_per_sample(pred, target, eps=eps).item()

    # Per-channel relative L2
    u_rel_l2 = relative_l2_per_sample(pred[..., 0:1], target[..., 0:1], eps=eps).item()
    v_rel_l2 = relative_l2_per_sample(pred[..., 1:2], target[..., 1:2], eps=eps).item()

    # 2. Pixel-level RMSE & MAE & MSE
    diff = pred - target
    mse = torch.mean(diff ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()

    # 3. R2 Score
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2).item()
    ss_res = torch.sum(diff ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + eps))

    # 4. Vorticity Relative L2
    vort_pred = compute_vorticity_2d(pred)
    vort_target = compute_vorticity_2d(target)
    vort_rel_l2 = relative_l2_per_sample(vort_pred, vort_target, eps=eps).item()
    vort_mse = F.mse_loss(vort_pred, vort_target).item()

    # 5. Kinetic Energy Error (KE)
    u_pred, v_pred = pred[..., 0], pred[..., 1]
    u_tgt, v_tgt = target[..., 0], target[..., 1]
    ke_pred = 0.5 * torch.mean((u_pred - torch.mean(u_pred, dim=1, keepdim=True)) ** 2 +
                              (v_pred - torch.mean(v_pred, dim=1, keepdim=True)) ** 2)
    ke_tgt = 0.5 * torch.mean((u_tgt - torch.mean(u_tgt, dim=1, keepdim=True)) ** 2 +
                             (v_tgt - torch.mean(v_tgt, dim=1, keepdim=True)) ** 2)
    ke_error = torch.abs(ke_pred - ke_tgt).item()

    # 6. Mean Velocity Profile Error (MVPE) along vertical slices
    # Sample 4 vertical probe lines across domain width
    w = pred.shape[-2]
    probe_x = [w // 4, w // 2, 3 * w // 4, w - 5]
    u_mean_pred = torch.mean(pred[..., 0], dim=(0, 1))  # (H, W)
    u_mean_tgt = torch.mean(target[..., 0], dim=(0, 1))  # (H, W)
    mvpe = torch.mean(torch.abs(u_mean_pred[:, probe_x] - u_mean_tgt[:, probe_x])).item()

    return {
        "rel_l2": rel_l2,
        "u_rel_l2": u_rel_l2,
        "v_rel_l2": v_rel_l2,
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "vorticity_rel_l2": vort_rel_l2,
        "vorticity_mse": vort_mse,
        "ke_error": ke_error,
        "mvpe": mvpe,
    }
