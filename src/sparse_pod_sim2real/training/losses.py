from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Official RealPDEBench metric:
    Computes per-sample relative L2 error: (1/N) * sum_i ||pred_i - target_i||_2 / ||target_i||_2
    """
    b = pred.size(0)
    pred_flat = pred.reshape(b, -1)
    target_flat = target.reshape(b, -1)
    diff_norm = torch.norm(pred_flat - target_flat, dim=1)
    target_norm = torch.norm(target_flat, dim=1)
    return torch.mean(diff_norm / (target_norm + eps))


def compute_vorticity_2d(u_field: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D vorticity: omega = dv/dx - du/dy
    u_field: [..., H, W, 2] (u, v)
    """
    u = u_field[..., 0]
    v = u_field[..., 1]

    # Central difference with boundary padding
    du_dy = (torch.roll(u, shifts=-1, dims=-2) - torch.roll(u, shifts=1, dims=-2)) * 0.5
    dv_dx = (torch.roll(v, shifts=-1, dims=-1) - torch.roll(v, shifts=1, dims=-1)) * 0.5
    vorticity = dv_dx - du_dy
    return vorticity


def vorticity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 loss on 2D vorticity field."""
    vort_pred = compute_vorticity_2d(pred)
    vort_target = compute_vorticity_2d(target)
    return relative_l2_per_sample(vort_pred, vort_target)


def sensor_point_loss(pred: torch.Tensor, target_sensors: torch.Tensor, sensor_indices: torch.Tensor) -> torch.Tensor:
    """
    Evaluate MSE loss specifically at sensor locations.
    pred: [B, Tout, H, W, 2]
    target_sensors: [B, Tout, Ps, 2]
    sensor_indices: [Ps]
    """
    b, tout, h, w, c = pred.shape
    pred_flat = pred.reshape(b, tout, h * w, c)
    pred_at_sensors = torch.index_select(pred_flat, dim=2, index=sensor_indices.to(pred.device))
    return F.mse_loss(pred_at_sensors, target_sensors)


class CompositePhysicalLoss(nn.Module):
    """
    Composite loss combining full-field MSE, channel-weighted MSE (focusing on v),
    vorticity physics loss, and optional modal/orthogonality regularization.
    """

    def __init__(
        self,
        v_weight: float = 2.0,
        vorticity_weight: float = 0.1,
        sensor_weight: float = 1.0,
        ortho_weight: float = 0.01,
    ):
        super().__init__()
        self.v_weight = v_weight
        self.vorticity_weight = vorticity_weight
        self.sensor_weight = sensor_weight
        self.ortho_weight = ortho_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sensor_indices: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        is_real_finetuning: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # Channel weighted MSE
        u_err = F.mse_loss(pred[..., 0], target[..., 0])
        v_err = F.mse_loss(pred[..., 1], target[..., 1])
        base_loss = u_err + self.v_weight * v_err

        loss = base_loss
        loss_dict = {"mse_u": u_err.item(), "mse_v": v_err.item(), "base_mse": base_loss.item()}

        # Vorticity loss
        if self.vorticity_weight > 0.0 and not is_real_finetuning:
            v_loss = vorticity_loss(pred, target)
            loss = loss + self.vorticity_weight * v_loss
            loss_dict["vorticity_loss"] = v_loss.item()

        # If model has orthogonality constraint
        if self.ortho_weight > 0.0 and model is not None and hasattr(model, "warping") and model.warping is not None:
            o_loss = model.warping.orthogonality_loss()
            loss = loss + self.ortho_weight * o_loss
            loss_dict["ortho_loss"] = o_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict
