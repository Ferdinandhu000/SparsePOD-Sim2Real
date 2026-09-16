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
        mean_flow: Optional[torch.Tensor] = None,
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
        m = h * w

        # pod_basis: (2*M, K)
        self.register_buffer("pod_basis", pod_basis.float())
        self.register_buffer("sensor_indices", sensor_indices.long())

        all_sensor_idx = torch.cat([sensor_indices, sensor_indices + m], dim=0)  # (2*Ps,)
        self.register_buffer("all_sensor_idx", all_sensor_idx)

        # Base flow mean field: (2*M,)
        if mean_flow is not None:
            self.register_buffer("mean_flow", mean_flow.float().reshape(-1))
            self.register_buffer("mean_sensor", self.mean_flow[all_sensor_idx])
        else:
            self.register_buffer("mean_flow", torch.zeros(2 * m))
            self.register_buffer("mean_sensor", torch.zeros(len(all_sensor_idx)))

        # Sub-basis: (2*Ps, K)
        phi_p = pod_basis[all_sensor_idx, :]
        self.register_buffer("phi_p", phi_p)

        # Prior energy matrix (Sigma_prior^-1)
        if prior_energy is not None:
            prior_weights = 1.0 / (prior_energy.float() + 1e-6)
            prior_weights = prior_weights / prior_weights.max()
            self.register_buffer("prior_mat", torch.diag(prior_weights))
        else:
            self.register_buffer("prior_mat", torch.eye(self.k))

    def solve_coefficients(
        self,
        sensor_values: torch.Tensor,
        adapted_basis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sensor_values: [B, T, Ps, 2]
        adapted_basis: optional [2*M, K] adapted basis
        Returns modal coefficients a: [B, T, K]
        """
        b, t, ps, c = sensor_values.shape
        u_val = sensor_values[..., 0]
        v_val = sensor_values[..., 1]
        y_vec = torch.cat([u_val, v_val], dim=-1)  # [B, T, 2*Ps]

        # Subtract sensor-level mean base flow for Reynolds fluctuation
        y_fluc = y_vec - self.mean_sensor.to(y_vec.device)

        if adapted_basis is not None:
            phi_p = adapted_basis[self.all_sensor_idx, :]  # (2*Ps, K)
        else:
            phi_p = self.phi_p

        # Gram matrix: [K, K]
        gram = phi_p.T @ phi_p + self.reg_lambda * self.prior_mat.to(phi_p.device)
        # RHS: [B, T, K]
        rhs = torch.einsum("btp,pk->btk", y_fluc, phi_p)
        # Solve: gram * a = rhs
        a = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)  # [B, T, K]
        return a

    def decode_field(
        self,
        a: torch.Tensor,
        adapted_basis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        a: [B, T, K]
        Returns: [B, T, H, W, 2]
        """
        b, t, k = a.shape
        basis = adapted_basis if adapted_basis is not None else self.pod_basis

        # Fluctuating field: [B, T, 2*M]
        flat_fluc = torch.einsum("btk,mk->btm", a, basis)
        # Add base mean flow
        flat_total = flat_fluc + self.mean_flow.to(flat_fluc.device)

        m = self.h * self.w
        u = flat_total[..., :m].reshape(b, t, self.h, self.w)
        v = flat_total[..., m:].reshape(b, t, self.h, self.w)
        return torch.stack([u, v], dim=-1)

    def forward(
        self,
        sensor_values: torch.Tensor,
        adapted_basis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (a_coefficients: [B, T, K], reconstructed_full_field: [B, T, H, W, 2])
        """
        a = self.solve_coefficients(sensor_values, adapted_basis=adapted_basis)
        field = self.decode_field(a, adapted_basis=adapted_basis)
        return a, field

