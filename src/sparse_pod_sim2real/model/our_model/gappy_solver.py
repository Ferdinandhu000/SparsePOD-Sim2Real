from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGappySolver(nn.Module):
    """
    Differentiable Gappy-POD solver that lifts sparse point measurements to a full-field coarse flow.
    Supports Bayesian MAP prior covariance damping.
    """

    def __init__(
        self,
        pod_basis: torch.Tensor,
        sensor_indices: torch.Tensor,
        h: int = 64,
        w: int = 128,
        reg_lambda: float = 1e-4,
        prior_energy: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.reg_lambda = reg_lambda
        self.k = pod_basis.shape[1]

        # pod_basis: (2*M, K)
        self.register_buffer("pod_basis", pod_basis.float())
        self.register_buffer("sensor_indices", sensor_indices.long())

        m = h * w
        all_sensor_idx = torch.cat([sensor_indices, sensor_indices + m], dim=0)  # (2*Ps,)
        self.register_buffer("all_sensor_idx", all_sensor_idx)

        # Sub-basis: (2*Ps, K)
        phi_p = pod_basis[all_sensor_idx, :]
        self.register_buffer("phi_p", phi_p)

        # Prior energy matrix (Sigma_prior^-1)
        if prior_energy is not None:
            # Normalized diagonal prior
            prior_weights = 1.0 / (prior_energy.float() + 1e-6)
            prior_weights = prior_weights / prior_weights.max()
            self.register_buffer("prior_mat", torch.diag(prior_weights))
        else:
            self.register_buffer("prior_mat", torch.eye(self.k))

    def solve_coefficients(
        self,
        sensor_values: torch.Tensor,
        w_align: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sensor_values: [B, T, Ps, 2]
        w_align: optional [K, K] alignment matrix rotating the basis
        Returns modal coefficients a: [B, T, K]
        """
        b, t, ps, c = sensor_values.shape
        u_val = sensor_values[..., 0]
        v_val = sensor_values[..., 1]
        y_vec = torch.cat([u_val, v_val], dim=-1)  # [B, T, 2*Ps]

        phi_p = self.phi_p
        if w_align is not None:
            phi_p = phi_p @ w_align  # (2*Ps, K)

        # Gram matrix: [K, K]
        gram = phi_p.T @ phi_p + self.reg_lambda * self.prior_mat.to(phi_p.device)
        # RHS: [B, T, K]
        rhs = torch.einsum("btp,pk->btk", y_vec, phi_p)
        # Solve: gram * a = rhs
        a = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)  # [B, T, K]
        return a

    def decode_field(
        self,
        a: torch.Tensor,
        w_align: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        a: [B, T, K]
        Returns: [B, T, H, W, 2]
        """
        b, t, k = a.shape
        basis = self.pod_basis
        if w_align is not None:
            basis = basis @ w_align  # (2*M, K)

        flat_u = torch.einsum("btk,mk->btm", a, basis)  # [B, T, 2*M]
        m = self.h * self.w
        u = flat_u[..., :m].reshape(b, t, self.h, self.w)
        v = flat_u[..., m:].reshape(b, t, self.h, self.w)
        return torch.stack([u, v], dim=-1)

    def forward(
        self,
        sensor_values: torch.Tensor,
        w_align: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (a_coefficients: [B, T, K], reconstructed_full_field: [B, T, H, W, 2])
        """
        a = self.solve_coefficients(sensor_values, w_align=w_align)
        field = self.decode_field(a, w_align=w_align)
        return a, field
