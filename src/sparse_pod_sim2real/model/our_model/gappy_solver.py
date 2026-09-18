from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


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

        # Execute linear solve strictly in full float32 with AMP autocast disabled to prevent low-precision Gram degeneration
        device_type = y_vec.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            phi_p_f32 = phi_p.float()
            y_fluc_f32 = y_fluc.float()
            prior_f32 = self.prior_mat.to(phi_p.device).float()

            # Gram matrix: [K, K] in float32
            gram = phi_p_f32.T @ phi_p_f32 + self.reg_lambda * prior_f32
            # RHS: [B, T, K] in float32
            rhs = torch.einsum("btp,pk->btk", y_fluc_f32, phi_p_f32)
            # Solve: gram * a = rhs
            a = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)  # [B, T, K]

        return a.to(sensor_values.dtype)

    def project_full_field(
        self,
        full_field: torch.Tensor,
        basis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project a dense field onto the POD basis without sensor subsampling."""
        if full_field.ndim != 5 or full_field.shape[-1] != 2:
            raise ValueError("full_field must have shape [B, T, H, W, 2].")
        if full_field.shape[2:4] != (self.h, self.w):
            raise ValueError(
                f"Expected spatial shape {(self.h, self.w)}, got {tuple(full_field.shape[2:4])}."
            )

        b, t = full_field.shape[:2]
        u = full_field[..., 0].reshape(b, t, -1)
        v = full_field[..., 1].reshape(b, t, -1)
        flat = torch.cat([u, v], dim=-1)
        active_basis = self.pod_basis if basis is None else basis
        fluctuation = flat - self.mean_flow.to(flat.device, dtype=flat.dtype)
        coefficients = torch.einsum("btm,mk->btk", fluctuation, active_basis.to(flat.dtype))
        reconstruction = self.decode_field(coefficients, adapted_basis=active_basis)
        return coefficients, reconstruction

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
        flat_fluc = torch.einsum("btk,mk->btm", a, basis.to(a.dtype))
        # Add base mean flow
        flat_total = flat_fluc + self.mean_flow.to(flat_fluc.device, dtype=flat_fluc.dtype)

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
