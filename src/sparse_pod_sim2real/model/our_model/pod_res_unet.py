from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .gappy_solver import DifferentiableGappySolver
from .subspace_warping import SubspaceWarping
from ..realpdebench.unet import Unet3d as UNet


class ModalLinearPropagator(nn.Module):
    """Lightweight modal time-stepper predicting future modal coefficients from history."""

    def __init__(self, k: int = 64, in_time: int = 20, out_time: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.k = k
        self.in_time = in_time
        self.out_time = out_time

        self.net = nn.Sequential(
            nn.Linear(in_time * k, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_time * k),
        )

    def forward(self, a_in: torch.Tensor) -> torch.Tensor:
        # a_in: [B, Tin, K]
        b = a_in.shape[0]
        flat_in = a_in.reshape(b, -1)
        flat_out = self.net(flat_in)
        return flat_out.reshape(b, self.out_time, self.k)


class PODResUNet3DSparse(nn.Module):
    """
    Flagship SOTA Architecture:
    1. Differentiable Gappy-POD lifts sparse sensor measurements to a globally coherent coarse flow field.
    2. Subspace Warping matrix W_align aligns modal basis with real flow on Grassmann manifold.
    3. Heavy 23M 3D-UNet operates on continuous 40-frame fluid tensor to predict high-frequency non-linear residual.
    """

    def __init__(
        self,
        pod_basis: torch.Tensor,
        sensor_indices: torch.Tensor,
        h: int = 64,
        w: int = 128,
        in_time: int = 20,
        out_time: int = 20,
        k: int = 64,
        reg_lambda: float = 1e-4,
        unet_dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        use_warping: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.in_time = in_time
        self.out_time = out_time
        self.k = k
        self.residual_weight = residual_weight
        self.use_warping = use_warping

        # 1. Gappy-POD physical lifting module
        self.gappy = DifferentiableGappySolver(
            pod_basis=pod_basis[:, :k],
            sensor_indices=sensor_indices,
            h=h,
            w=w,
            reg_lambda=reg_lambda,
        )

        # 2. Subspace Warping for Sim-to-Real modal rotation
        self.warping = SubspaceWarping(k=k, orthogonal=True) if use_warping else None

        # 3. Modal temporal propagator
        self.modal_prop = ModalLinearPropagator(k=k, in_time=in_time, out_time=out_time)

        # 4. High-capacity 23M 3D-UNet for spatio-temporal residual refinement
        total_time = in_time + out_time
        self.time_project = nn.Linear(total_time, out_time)
        self.unet = UNet(
            dim=unet_dim,
            channels=2,
            out_channels=2,
            dim_mults=dim_mults,
            in_time=total_time,
            out_time=total_time,
        )

    def get_alignment_matrix(self) -> Optional[torch.Tensor]:
        return self.warping.get_matrix() if self.warping is not None else None

    def forward(self, batch: dict | torch.Tensor, return_components: bool = False):
        """
        batch: dict with "sensor_values" [B, Tin, Ps, 2]
        """
        if isinstance(batch, dict):
            sensor_values = batch["sensor_values"]
        else:
            sensor_values = batch

        w_align = self.get_alignment_matrix()

        # Step 1: Solve historical modal coefficients from sparse sensors
        # a_in: [B, Tin, K], u_in_pod: [B, Tin, H, W, 2]
        a_in, u_in_pod = self.gappy(sensor_values, w_align=w_align)

        # Step 2: Propagate modal coefficients to future
        # a_out: [B, Tout, K]
        a_out = self.modal_prop(a_in)

        # Step 3: Decode future coarse physical field
        # u_out_pod: [B, Tout, H, W, 2]
        u_out_pod = self.gappy.decode_field(a_out, w_align=w_align)

        # Step 4: Construct continuous 40-frame fluid tensor
        # [B, 40, H, W, 2]
        u_stream = torch.cat([u_in_pod, u_out_pod], dim=1)

        # Step 5: High-capacity 3D UNet predicts high-frequency residual
        # delta_all: [B, 40, H, W, 2]
        delta_all = self.unet(u_stream)
        # Project time 40 -> 20: [B, H, W, C, 40] -> [B, H, W, C, 20] -> [B, 20, H, W, C]
        delta_u = self.time_project(delta_all.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # Step 6: Full-field synthesis
        u_final = u_out_pod + self.residual_weight * delta_u

        if return_components:
            return {
                "u_final": u_final,
                "u_pod": u_out_pod,
                "delta_u": delta_u,
                "a_in": a_in,
                "a_out": a_out,
            }
        return u_final
