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


def compute_vorticity_2d(
    u_field: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute 2D vorticity: omega = dv/dx - du/dy
    u_field: [..., H, W, 2] (u, v)
    """
    u = u_field[..., 0]
    v = u_field[..., 1]

    # Match evaluation: central differences in the interior and one-sided
    # differences at non-periodic domain boundaries.
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(v)
    dv_dx[..., :, 1:-1] = (v[..., :, 2:] - v[..., :, :-2]) / (2.0 * dx)
    dv_dx[..., :, 0] = (v[..., :, 1] - v[..., :, 0]) / dx
    dv_dx[..., :, -1] = (v[..., :, -1] - v[..., :, -2]) / dx
    du_dy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dy)
    du_dy[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dy
    du_dy[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy
    vorticity = dv_dx - du_dy
    if valid_mask is not None:
        mask = valid_mask.to(device=vorticity.device, dtype=torch.bool)
        vorticity = torch.where(mask, vorticity, torch.zeros_like(vorticity))
    return vorticity


def vorticity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Relative L2 loss on 2D vorticity field."""
    vort_pred = compute_vorticity_2d(pred, dx=dx, dy=dy, valid_mask=valid_mask)
    vort_target = compute_vorticity_2d(target, dx=dx, dy=dy, valid_mask=valid_mask)
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
    Composite physical loss supporting two distinct Sim2Real regimes:
    1. 'sparse_sensors': During Real fine-tuning, strictly supervise ONLY at sensor locations (no full field).
    2. 'few_shot_full': During Real fine-tuning, supervise full field from few-shot calibration trajectories.
    """

    def __init__(
        self,
        v_weight: float = 2.0,
        vorticity_weight: float = 0.1,
        sensor_weight: float = 1.0,
        ortho_weight: float = 0.01,
        supervision_mode: str = "sparse_sensors",
        grid_dx: float = 1.0,
        grid_dy: float = 1.0,
        vorticity_valid_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.v_weight = v_weight
        self.vorticity_weight = vorticity_weight
        self.sensor_weight = sensor_weight
        self.ortho_weight = ortho_weight
        self.supervision_mode = supervision_mode
        self.grid_dx = float(grid_dx)
        self.grid_dy = float(grid_dy)
        self.vorticity_valid_mask = vorticity_valid_mask

    def forward(
        self,
        pred: torch.Tensor,
        target_or_batch: torch.Tensor | dict,
        model: Optional[nn.Module] = None,
        is_real_finetuning: bool = False,
        target_sensors: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        loss_dict = {}

        if isinstance(target_or_batch, dict):
            target_full = target_or_batch.get("y_full", None)
            if target_sensors is None:
                target_sensors = target_or_batch.get("y_sensor_values", None)
            if sensor_indices is None and model is not None:
                sensor_indices = getattr(model, "sensor_indices", None)
                if sensor_indices is None and hasattr(model, "gappy"):
                    sensor_indices = getattr(model.gappy, "sensor_indices", None)
        else:
            target_full = target_or_batch

        # Mode A: Real Fine-tuning with strictly sparse sensor supervision
        if is_real_finetuning and self.supervision_mode == "sparse_sensors":
            if target_sensors is None or sensor_indices is None:
                raise ValueError("Sparse sensor supervision requires target_sensors and sensor_indices.")
            s_loss = sensor_point_loss(pred, target_sensors, sensor_indices)
            loss = self.sensor_weight * s_loss
            loss_dict["sensor_mse"] = s_loss.item()

            # Grassmann tangent regularizer if model contains warping
            if (
                self.ortho_weight > 0.0
                and model is not None
                and hasattr(model, "warping")
                and model.warping is not None
            ):
                reg_loss = model.warping.regularization_loss()
                loss = loss + self.ortho_weight * reg_loss
                loss_dict["grassmann_reg"] = reg_loss.item()

            loss_dict["total_loss"] = loss.item()
            return (loss, loss_dict) if return_dict else loss

        # Mode B: Numerical Pre-training OR few_shot_full supervision
        if target_full is None:
            raise ValueError("Full-field supervision requires target_full tensor or 'y_full' in batch dict.")

        u_err = F.mse_loss(pred[..., 0], target_full[..., 0])
        v_err = F.mse_loss(pred[..., 1], target_full[..., 1])
        base_loss = u_err + self.v_weight * v_err

        loss = base_loss
        loss_dict = {"mse_u": u_err.item(), "mse_v": v_err.item(), "base_mse": base_loss.item()}

        # Vorticity loss (for numerical pre-training)
        if self.vorticity_weight > 0.0 and not is_real_finetuning:
            v_loss = vorticity_loss(
                pred,
                target_full,
                dx=self.grid_dx,
                dy=self.grid_dy,
                valid_mask=self.vorticity_valid_mask,
            )
            loss = loss + self.vorticity_weight * v_loss
            loss_dict["vorticity_loss"] = v_loss.item()

        # Orthogonality / Grassmann regularization
        if (
            self.ortho_weight > 0.0
            and model is not None
            and hasattr(model, "warping")
            and model.warping is not None
        ):
            reg_loss = model.warping.regularization_loss()
            loss = loss + self.ortho_weight * reg_loss
            loss_dict["grassmann_reg"] = reg_loss.item()

        loss_dict["total_loss"] = loss.item()
        return (loss, loss_dict) if return_dict else loss
