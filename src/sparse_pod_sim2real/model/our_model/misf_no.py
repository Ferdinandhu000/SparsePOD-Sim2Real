from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gappy_solver import DifferentiableGappySolver
from .subspace_warping import GrassmannSubspaceAlignment


class PhysicsCrossAttention(nn.Module):
    """
    Point-cloud Physics Cross-Attention:
    Learns high-order non-linear modal perturbations from irregular sparse sensor measurements.
    """

    def __init__(self, k: int = 64, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.n_heads = n_heads

        # Latent modal queries: [K, d_model]
        self.modal_queries = nn.Parameter(torch.randn(k, d_model) * 0.02)

        # Sensor projection: 2 (u, v) + 2 (coords) -> d_model
        self.sensor_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.proj_out = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, sensor_values: torch.Tensor, sensor_coords: torch.Tensor) -> torch.Tensor:
        b, t, ps, _ = sensor_values.shape
        coords_exp = sensor_coords.unsqueeze(0).unsqueeze(0).expand(b, t, ps, 2)
        sensor_tokens = torch.cat([sensor_values, coords_exp], dim=-1)  # [B, T, Ps, 4]

        tokens_flat = sensor_tokens.reshape(b * t, ps, 4)
        tokens_emb = self.sensor_proj(tokens_flat)  # [B*T, Ps, D]

        queries = self.modal_queries.unsqueeze(0).expand(b * t, self.k, self.d_model)

        # Cross attention: queries attend to sensor keys/values
        attn_out, _ = self.attn(queries, tokens_emb, tokens_emb)  # [B*T, K, D]
        delta_a = self.proj_out(attn_out).squeeze(-1)  # [B*T, K]
        return delta_a.reshape(b, t, self.k)


class LatentDynamicsFNO1D(nn.Module):
    """Temporal 1D-FNO operating on low-dimensional modal coordinates."""

    def __init__(self, k: int = 64, in_time: int = 20, out_time: int = 20, modes: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.k = k
        self.in_time = in_time
        self.out_time = out_time
        self.modes = modes
        self.hidden_dim = hidden_dim

        self.fc0 = nn.Linear(k, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k)
        self.time_project = nn.Linear(in_time, out_time)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        x = self.fc0(z_in)  # [B, Tin, D]
        x_conv = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(x + x_conv)

        x_ft = torch.fft.rfft(x, dim=1)
        x_ft[:, self.modes:, :] = 0.0
        x_filtered = torch.fft.irfft(x_ft, n=self.in_time, dim=1)
        x = x + x_filtered

        x = F.gelu(self.fc1(x))
        z_step = self.fc2(x)  # [B, Tin, K]

        z_out = self.time_project(z_step.permute(0, 2, 1)).permute(0, 2, 1)
        return z_out  # [B, Tout, K]


class MISFNO(nn.Module):
    """
    Physics-Modal Sparse-to-Full Neural Operator (MISF-NO):
    1. Dual-Path Gappy Encoder (Bayesian MAP + Physics Cross-Attention)
    2. Latent Dynamics 1D-FNO
    3. Grassmann Subspace Alignment
    """

    def __init__(
        self,
        pod_basis: torch.Tensor | dict,
        sensor_indices: torch.Tensor,
        h: int = 64,
        w: int = 128,
        in_time: int = 20,
        out_time: int = 20,
        k: int = 64,
        reg_lambda: float = 1e-4,
        latent_dim: int = 128,
        use_warping: bool = True,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.in_time = in_time
        self.out_time = out_time
        self.k = k
        self.use_warping = use_warping

        if isinstance(pod_basis, dict):
            basis_tensor = pod_basis["basis"][:, :k]
            mean_tensor = pod_basis.get("mean", None)
            perp_tensor = pod_basis.get("basis_perp", None)
        else:
            basis_tensor = pod_basis[:, :k]
            mean_tensor = None
            perp_tensor = None

        # Path A: Differentiable Gappy Solver
        self.gappy = DifferentiableGappySolver(
            pod_basis=basis_tensor,
            sensor_indices=sensor_indices,
            mean_flow=mean_tensor,
            h=h,
            w=w,
            reg_lambda=reg_lambda,
        )

        # Path B: Physics Cross-Attention
        self.phca = PhysicsCrossAttention(k=k, d_model=latent_dim)
        self.norm_a = nn.LayerNorm(k)

        # Temporal Latent Propagator
        self.ldno = LatentDynamicsFNO1D(k=k, in_time=in_time, out_time=out_time, hidden_dim=latent_dim)

        # Grassmann Subspace Alignment
        r = perp_tensor.shape[1] if perp_tensor is not None else 16
        self.warping = (
            GrassmannSubspaceAlignment(k=k, r=r, phi_sim=basis_tensor, phi_perp=perp_tensor)
            if use_warping
            else None
        )

    def forward(self, batch: dict | torch.Tensor, mode: str = "sparse") -> torch.Tensor:
        if isinstance(batch, dict):
            if mode == "full" and "x_full" in batch:
                z_in, _ = self.gappy.project_full_field(batch["x_full"])
                z_out = self.ldno(z_in)
                return self.gappy.decode_field(z_out)
            sensor_values = batch["sensor_values"]
            sensor_coords = batch["sensor_coords"]
        else:
            raise ValueError("MISFNO expects batch dict with 'sensor_values' and 'sensor_coords'")

        adapted_basis = self.warping.get_adapted_basis(self.gappy.pod_basis) if self.warping is not None else None

        # Path A: Gappy MAP coefficients
        a_map = self.gappy.solve_coefficients(sensor_values, adapted_basis=adapted_basis)  # [B, Tin, K]

        # Path B: Non-linear compensation
        delta_a = self.phca(sensor_values, sensor_coords)  # [B, Tin, K]

        # Dual-path fusion
        z_in = a_map + self.norm_a(delta_a)  # [B, Tin, K]

        # Latent temporal evolution
        z_out = self.ldno(z_in)  # [B, Tout, K]

        # Decode full physical field
        u_pred = self.gappy.decode_field(z_out, adapted_basis=adapted_basis)
        return u_pred
