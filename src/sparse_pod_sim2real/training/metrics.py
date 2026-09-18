from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .losses import relative_l2_per_sample


def build_naca0025_fluid_mask(
    grid_shape: Tuple[int, int] = (64, 128),
    center_y: float = 32.0,
    center_x: float = 24.5,
    chord: float = 35.0,
    relative_thickness: float = 0.25,
) -> torch.Tensor:
    """Builds the canonical Foil fluid mask used by the 64x128 benchmark grid."""
    h, w = grid_shape
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    s = (xx - (center_x - chord / 2.0)) / chord
    inside_chord = (s >= 0.0) & (s <= 1.0)
    s_clip = torch.clamp(s, 0.0, 1.0)
    half_thickness = 5.0 * relative_thickness * chord * (
        0.2969 * torch.sqrt(s_clip)
        - 0.1260 * s_clip
        - 0.3516 * s_clip.square()
        + 0.2843 * s_clip.pow(3)
        - 0.1015 * s_clip.pow(4)
    )
    solid = inside_chord & ((yy - center_y).abs() <= half_thickness)
    return (~solid).float()


def compute_vorticity_2d(
    u_field: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    eroded_fluid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes 2D vorticity omega = dv/dx - du/dy using 2nd-order interior central differences.

    ``dx`` and ``dy`` may be physical spacings or grid-cell spacings. RealPDEBench
    Foil stores Sim coordinates in metres and Real PIV coordinates in pixels, so
    the experiment configs use isotropic grid-cell units (dx=dy=1). The relative
    vorticity error is invariant to this common scale.
    Optionally masks out solid body boundary points via an eroded fluid mask.
    u_field: [..., H, W, 2]
    """
    u = u_field[..., 0]
    v = u_field[..., 1]

    # Interior central differences
    dv_dx = torch.zeros_like(v)
    du_dy = torch.zeros_like(u)

    # 2nd-order central differences in interior: (x+1 - x-1) / (2*dx)
    dv_dx[..., :, 1:-1] = (v[..., :, 2:] - v[..., :, :-2]) / (2.0 * dx)
    # One-sided forward/backward differences at boundary
    dv_dx[..., :, 0] = (v[..., :, 1] - v[..., :, 0]) / dx
    dv_dx[..., :, -1] = (v[..., :, -1] - v[..., :, -2]) / dx

    du_dy[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * dy)
    du_dy[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / dy
    du_dy[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / dy

    vorticity = dv_dx - du_dy

    if eroded_fluid_mask is not None:
        mask = eroded_fluid_mask.to(vorticity.device).bool()
        vorticity = torch.where(mask, vorticity, torch.zeros_like(vorticity))

    return vorticity


def erode_fluid_mask(fluid_mask: torch.Tensor, stencil_radius: int = 1) -> torch.Tensor:
    """
    Erodes the fluid mask by stencil_radius cells (equivalent to dilating the solid body mask).
    A cell remains valid fluid if all its stencil neighbors are fluid.
    fluid_mask: [H, W] or [1, 1, H, W]
    """
    if fluid_mask.ndim == 2:
        m = fluid_mask.unsqueeze(0).unsqueeze(0).float()
    else:
        m = fluid_mask.float()

    k_size = 2 * stencil_radius + 1
    # Minimum pooling over stencil window
    # min(x) = -max(-x)
    eroded = -F.max_pool2d(-m, kernel_size=k_size, stride=1, padding=stencil_radius)
    if fluid_mask.ndim == 2:
        return (eroded.squeeze(0).squeeze(0) > 0.5).float()
    return (eroded > 0.5).float()


def compute_unobserved_mask(
    grid_shape: Tuple[int, int],
    sensor_indices_1d: torch.Tensor,
    radius_pixels: float = 3.0,
    fluid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generates a binary mask of fluid cells located outside a pixel neighborhood radius from all sensors.
    """
    h, w = grid_shape
    s_idx = sensor_indices_1d.detach().cpu()
    y_grid, x_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Sensor (y, x) coordinates on CPU
    s_y = (s_idx // w).float()
    s_x = (s_idx % w).float()

    # Distance map: min distance to any sensor
    # shape: [H, W, Ps]
    dist_sq = (y_grid.unsqueeze(-1).float() - s_y) ** 2 + (x_grid.unsqueeze(-1).float() - s_x) ** 2
    min_dist = torch.sqrt(torch.min(dist_sq, dim=-1).values)

    unobserved = (min_dist > radius_pixels).float()
    if fluid_mask is not None:
        unobserved = unobserved * fluid_mask.cpu().float()

    return unobserved


class StreamingMetricAccumulator:
    """
    Online metric accumulator that aggregates running squared errors and reference energies
    per trajectory in O(1) memory without collecting all batch predictions onto CPU.
    Computes unweighted average of per-trajectory relative L2.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (64, 128),
        fluid_mask: Optional[torch.Tensor] = None,
        sensor_indices: Optional[torch.Tensor] = None,
        unobserved_radius: float = 3.0,
        dx: float = 1.0,
        dy: float = 1.0,
        eps: float = 1e-8,
    ):
        self.grid_shape = grid_shape
        self.eps = eps
        self.dx = float(dx)
        self.dy = float(dy)
        self.fluid_mask = fluid_mask
        self.eroded_fluid_mask = erode_fluid_mask(fluid_mask) if fluid_mask is not None else None
        self.sensor_indices = sensor_indices

        if sensor_indices is not None:
            self.unobserved_mask = compute_unobserved_mask(
                grid_shape=grid_shape,
                sensor_indices_1d=sensor_indices,
                radius_pixels=unobserved_radius,
                fluid_mask=fluid_mask,
            )
        else:
            self.unobserved_mask = None

        # Trajectory-level running sums: stem -> dict of accumulators
        self.traj_records: Dict[str, Dict[str, float]] = {}

    def update(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor],
        traj_stems: List[str],
        sensor_values_target: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Updates running squared error metrics from a batch.
        pred: [B, Tout, H, W, 2]. target may be omitted for a strict
        sensor-only selector when sensor_values_target is supplied.
        """
        b, tout, h, w, c = pred.shape

        if target is None:
            if sensor_values_target is None or self.sensor_indices is None:
                raise ValueError("Sensor-only metric update requires sensor targets and sensor indices.")
            idx = self.sensor_indices.to(pred.device)
            pred_sensor = pred.detach().float().reshape(b, tout, h * w, c).index_select(2, idx).cpu()
            target_sensor = sensor_values_target.detach().float().cpu()
            for i in range(b):
                stem = traj_stems[i]
                rec = self.traj_records.setdefault(stem, self._empty_record())
                diff_sensor = pred_sensor[i] - target_sensor[i]
                rec["sensor_diff_sum"] += diff_sensor.square().sum().item()
                rec["sensor_tgt_sum"] += target_sensor[i].square().sum().item()
                rec["frame_count"] += tout
            return

        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()

        diff = pred - target
        diff_sq = diff ** 2
        tgt_sq = target ** 2

        # Vorticity
        vort_pred = compute_vorticity_2d(
            pred, dx=self.dx, dy=self.dy, eroded_fluid_mask=self.eroded_fluid_mask
        )
        vort_tgt = compute_vorticity_2d(
            target, dx=self.dx, dy=self.dy, eroded_fluid_mask=self.eroded_fluid_mask
        )
        vort_diff_sq = (vort_pred - vort_tgt) ** 2
        vort_tgt_sq = vort_tgt ** 2

        # Masks
        fluid_m = self.fluid_mask.cpu() if self.fluid_mask is not None else torch.ones(h, w)
        unobs_m = self.unobserved_mask.cpu() if self.unobserved_mask is not None else torch.ones(h, w)

        for i in range(b):
            stem = traj_stems[i]
            if stem not in self.traj_records:
                self.traj_records[stem] = self._empty_record()

            rec = self.traj_records[stem]
            rec["frame_count"] += tout

            # 1. Full-field & Channels
            rec["full_diff_sum"] += diff_sq[i].sum().item()
            rec["full_tgt_sum"] += tgt_sq[i].sum().item()
            rec["u_diff_sum"] += diff_sq[i, ..., 0].sum().item()
            rec["u_tgt_sum"] += tgt_sq[i, ..., 0].sum().item()
            rec["v_diff_sum"] += diff_sq[i, ..., 1].sum().item()
            rec["v_tgt_sum"] += tgt_sq[i, ..., 1].sum().item()

            # 2. Fluid domain
            rec["fluid_diff_sum"] += (diff_sq[i] * fluid_m.unsqueeze(-1)).sum().item()
            rec["fluid_tgt_sum"] += (tgt_sq[i] * fluid_m.unsqueeze(-1)).sum().item()

            # 3. Unobserved region
            rec["unobs_diff_sum"] += (diff_sq[i] * unobs_m.unsqueeze(-1)).sum().item()
            rec["unobs_tgt_sum"] += (tgt_sq[i] * unobs_m.unsqueeze(-1)).sum().item()

            # 4. Vorticity
            rec["vort_diff_sum"] += vort_diff_sq[i].sum().item()
            rec["vort_tgt_sum"] += vort_tgt_sq[i].sum().item()

            # 5. Sensors
            if self.sensor_indices is not None:
                p_diff = diff_sq[i].reshape(tout, h * w, 2)[:, self.sensor_indices.cpu()]
                p_tgt = tgt_sq[i].reshape(tout, h * w, 2)[:, self.sensor_indices.cpu()]
                rec["sensor_diff_sum"] += p_diff.sum().item()
                rec["sensor_tgt_sum"] += p_tgt.sum().item()
            elif sensor_values_target is not None:
                raise ValueError(
                    "sensor_values_target was supplied without sensor_indices, so predicted "
                    "sensor values cannot be located on the grid."
                )

    @staticmethod
    def _empty_record() -> Dict[str, float]:
        return {
            "full_diff_sum": 0.0,
            "full_tgt_sum": 0.0,
            "fluid_diff_sum": 0.0,
            "fluid_tgt_sum": 0.0,
            "u_diff_sum": 0.0,
            "u_tgt_sum": 0.0,
            "v_diff_sum": 0.0,
            "v_tgt_sum": 0.0,
            "unobs_diff_sum": 0.0,
            "unobs_tgt_sum": 0.0,
            "vort_diff_sum": 0.0,
            "vort_tgt_sum": 0.0,
            "sensor_diff_sum": 0.0,
            "sensor_tgt_sum": 0.0,
            "frame_count": 0,
        }

    def compute(self) -> Dict[str, float]:
        """
        Computes final unweighted trajectory-averaged metrics across all evaluated trajectories.
        """
        if not self.traj_records:
            return {"rel_l2": 1.0, "fluid_fullfield_rel_l2": 1.0}

        full_rel_l2s = []
        fluid_rel_l2s = []
        u_rel_l2s = []
        v_rel_l2s = []
        unobs_rel_l2s = []
        vort_rel_l2s = []
        sensor_rel_l2s = []
        rmses = []

        for stem, rec in self.traj_records.items():
            if rec["full_tgt_sum"] > 0:
                full_rel_l2s.append(np.sqrt(rec["full_diff_sum"]) / max(self.eps, np.sqrt(rec["full_tgt_sum"])))
                fluid_rel_l2s.append(np.sqrt(rec["fluid_diff_sum"]) / max(self.eps, np.sqrt(rec["fluid_tgt_sum"])))
                u_rel_l2s.append(np.sqrt(rec["u_diff_sum"]) / max(self.eps, np.sqrt(rec["u_tgt_sum"])))
                v_rel_l2s.append(np.sqrt(rec["v_diff_sum"]) / max(self.eps, np.sqrt(rec["v_tgt_sum"])))
                unobs_rel_l2s.append(np.sqrt(rec["unobs_diff_sum"]) / max(self.eps, np.sqrt(rec["unobs_tgt_sum"])))
                vort_rel_l2s.append(np.sqrt(rec["vort_diff_sum"]) / max(self.eps, np.sqrt(rec["vort_tgt_sum"])))

            if rec["sensor_tgt_sum"] > 0:
                sensor_rel_l2s.append(np.sqrt(rec["sensor_diff_sum"]) / max(self.eps, np.sqrt(rec["sensor_tgt_sum"])))

            if rec["full_tgt_sum"] > 0:
                total_voxels = rec["frame_count"] * self.grid_shape[0] * self.grid_shape[1] * 2
                rmses.append(np.sqrt(rec["full_diff_sum"] / max(1, total_voxels)))

        result = {
            "sensor_rel_l2": float(np.mean(sensor_rel_l2s)) if sensor_rel_l2s else 0.0,
            "num_trajectories": len(self.traj_records),
        }
        if full_rel_l2s:
            result.update({
                "rel_l2": float(np.mean(full_rel_l2s)),
                "fluid_fullfield_rel_l2": float(np.mean(fluid_rel_l2s)),
                "channel_u_rel_l2": float(np.mean(u_rel_l2s)),
                "channel_v_rel_l2": float(np.mean(v_rel_l2s)),
                "unobserved_rel_l2": float(np.mean(unobs_rel_l2s)),
                "vorticity_rel_l2": float(np.mean(vort_rel_l2s)),
                "rmse": float(np.mean(rmses)),
            })
        return result


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    fluid_mask: Optional[torch.Tensor] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Standard batch evaluation metrics for direct testing and compatibility.
    pred, target: [B, Tout, H, W, 2]
    """
    # 1. Official Sample-wise Relative L2 Error
    rel_l2 = relative_l2_per_sample(pred, target, eps=eps).item()
    if fluid_mask is not None:
        mask = fluid_mask.to(device=pred.device, dtype=pred.dtype).view(1, 1, *fluid_mask.shape, 1)
        fluid_rel_l2 = relative_l2_per_sample(pred * mask, target * mask, eps=eps).item()
    else:
        fluid_rel_l2 = rel_l2
    u_rel_l2 = relative_l2_per_sample(pred[..., 0:1], target[..., 0:1], eps=eps).item()
    v_rel_l2 = relative_l2_per_sample(pred[..., 1:2], target[..., 1:2], eps=eps).item()

    # 2. Pixel-level RMSE & MAE
    diff = pred - target
    mse = torch.mean(diff ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()

    # 3. Vorticity Relative L2 with eroded fluid mask
    eroded_mask = erode_fluid_mask(fluid_mask) if fluid_mask is not None else None
    vort_pred = compute_vorticity_2d(pred, dx=dx, dy=dy, eroded_fluid_mask=eroded_mask)
    vort_target = compute_vorticity_2d(target, dx=dx, dy=dy, eroded_fluid_mask=eroded_mask)
    vort_rel_l2 = relative_l2_per_sample(vort_pred, vort_target, eps=eps).item()

    return {
        "rel_l2": rel_l2,
        "fluid_fullfield_rel_l2": fluid_rel_l2,
        "channel_u_rel_l2": u_rel_l2,
        "channel_v_rel_l2": v_rel_l2,
        "rmse": rmse,
        "mae": mae,
        "vorticity_rel_l2": vort_rel_l2,
    }
