from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gappy_solver import DifferentiableGappySolver
from .subspace_warping import SubspaceWarping


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
        self.queries = nn.Parameter(torch.randn(k, d_model) * 0.02)

        # Value projection: [2 -> d_model]
        self.val_proj = nn.Linear(2, d_model)
        # Coordinate positional embedding: [2 -> d_model]
        self.pos_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, sensor_values: torch.Tensor, sensor_coords: torch.Tensor) -> torch.Tensor:
        """
        sensor_values: [B, T, Ps, 2]
        sensor_coords: [Ps, 2]
        Returns delta_a: [B, T, K]
        """
        b, t, ps, _ = sensor_values.shape
        # Flatten batch and time for attention: [B*T, Ps, 2]
        val_flat = sensor_values.reshape(b * t, ps, 2)
        v_feat = self.val_proj(val_flat)
        p_feat = self.pos_proj(sensor_coords.to(sensor_values.device)).unsqueeze(0).expand(b * t, -1, -1)
        kv = v_feat + p_feat  # [B*T, Ps, d_model]

        # Expand queries: [B*T, K, d_model]
        q = self.queries.unsqueeze(0).expand(b * t, -1, -1)

        # Cross attention: queries attend to sensor points
        attn_out, _ = self.mha(q, kv, kv)  # [B*T, K, d_model]

        # Project to modal perturbation scalar
        delta_a_flat = self.out_proj(attn_out).squeeze(-1)  # [B*T, K]
        return delta_a_flat.reshape(b, t, self.k)


class LatentDynamicsFNO1D(nn.Module):
    """1D Fourier Neural Operator evolving modal coefficients in compact latent space."""

    def __init__(self, k: int = 64, in_time: int = 20, out_time: int = 20, hidden_dim: int = 128, modes: int = 8):
        super().__init__()
        self.k = k
        self.in_time = in_time
        self.out_time = out_time
        self.modes = modes

        self.fc0 = nn.Linear(k, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k)

        self.time_project = nn.Linear(in_time, out_time)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        # z_in: [B, Tin, K]
        x = self.fc0(z_in)  # [B, Tin, D]
        x_conv = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.gelu(x + x_conv)

        # 1D FFT along temporal dimension
        x_ft = torch.fft.rfft(x, dim=1)
        # Low frequency filter
        x_ft[:, self.modes:, :] = 0.0
        x_filtered = torch.fft.irfft(x_ft, n=self.in_time, dim=1)
        x = x + x_filtered

        x = F.gelu(self.fc1(x))
        z_step = self.fc2(x)  # [B, Tin, K]

        # Project time from Tin to Tout: [B, K, Tin] -> [B, K, Tout]
        z_out = self.time_project(z_step.permute(0, 2, 1)).permute(0, 2, 1)
        return z_out  # [B, Tout, K]


class MISFNO(nn.Module):
    """
    Physics-Modal Sparse-to-Full Neural Operator (MISF-NO):
    1. Dual-Path Gappy Encoder (Bayesian MAP + Physics Cross-Attention)
    2. Latent Dynamics 1D-FNO
    3. Subspace Warping on Grassmann Manifold
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

        # Path A: Differentiable Gappy Solver
        self.gappy = DifferentiableGappySolver(
            pod_basis=pod_basis[:, :k],
            sensor_indices=sensor_indices,
            h=h,
            w=w,
            reg_lambda=reg_lambda,
        )

        # Path B: Physics Cross-Attention
        self.phca = PhysicsCrossAttention(k=k, d_model=latent_dim)
        self.norm_a = nn.LayerNorm(k)

        # Temporal Latent Propagator
        self.ldno = LatentDynamicsFNO1D(k=k, in_time=in_time, out_time=out_time, hidden_dim=latent_dim)

        # Grassmann Subspace Warping
        self.warping = SubspaceWarping(k=k, orthogonal=True) if use_warping else None

    def forward(self, batch: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, dict):
            sensor_values = batch["sensor_values"]
            sensor_coords = batch["sensor_coords"]
        else:
            raise ValueError("MISFNO expects batch dict with 'sensor_values' and 'sensor_coords'")

        w_align = self.warping.get_matrix() if self.use_warping else None

        # Path A: Gappy MAP coefficients
        a_map = self.gappy.solve_coefficients(sensor_values, w_align=w_align)  # [B, Tin, K]

        # Path B: Non-linear compensation
        delta_a = self.phca(sensor_values, sensor_coords)  # [B, Tin, K]

        # Dual-path fusion
        z_in = a_map + self.norm_a(delta_a)  # [B, Tin, K]

        # Latent temporal evolution
        z_out = self.ldno(z_in)  # [B, Tout, K]

        # Decode full physical field
        u_pred = self.gappy.decode_field(z_out, w_align=w_align)
        return u_pred
