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
    ):
        super().__init__()
        self.v_weight = v_weight
        self.vorticity_weight = vorticity_weight
        self.sensor_weight = sensor_weight
        self.ortho_weight = ortho_weight
        self.supervision_mode = supervision_mode

    def forward(
        self,
        pred: torch.Tensor,
        target_full: Optional[torch.Tensor] = None,
        target_sensors: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
        is_real_finetuning: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        loss_dict = {}

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
            return loss, loss_dict

        # Mode B: Numerical Pre-training OR few_shot_full supervision
        if target_full is None:
            raise ValueError("Full-field supervision requires target_full tensor.")

        u_err = F.mse_loss(pred[..., 0], target_full[..., 0])
        v_err = F.mse_loss(pred[..., 1], target_full[..., 1])
        base_loss = u_err + self.v_weight * v_err

        loss = base_loss
        loss_dict = {"mse_u": u_err.item(), "mse_v": v_err.item(), "base_mse": base_loss.item()}

        # Vorticity loss (for numerical pre-training)
        if self.vorticity_weight > 0.0 and not is_real_finetuning:
            v_loss = vorticity_loss(pred, target_full)
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
        return loss, loss_dict

