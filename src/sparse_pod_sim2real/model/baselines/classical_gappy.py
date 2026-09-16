from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalGappyPOD(nn.Module):
    """
    Classical Gappy-POD solver for reconstructing full fluid fields from sparse sensor readings.
    u_full(t) = Phi * a(t)
    y_sensor(t) = Phi_p * a(t)
    a_hat(t) = (Phi_p^T * Phi_p + lambda * I)^(-1) * Phi_p^T * y_sensor(t)
    """

    def __init__(
        self,
        pod_basis: torch.Tensor,
        sensor_indices: torch.Tensor,
        mean_flow: torch.Tensor | None = None,
        reg_lambda: float = 1e-4,
        h: int = 64,
        w: int = 128,
        in_time: int = 20,
        out_time: int = 20,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.in_time = in_time
        self.out_time = out_time
        self.reg_lambda = reg_lambda

        # pod_basis: (M*2, K) where M = H*W
        self.register_buffer("pod_basis", pod_basis.float())
        self.register_buffer("sensor_indices", sensor_indices.long())
        self.k = pod_basis.shape[1]

        # Extract sub-basis Phi_p: (Ps*2, K)
        # Spatial flat indices for u and v
        m = h * w
        u_idx = sensor_indices
        v_idx = sensor_indices + m
        all_sensor_idx = torch.cat([u_idx, v_idx], dim=0)  # (2*Ps,)
        self.register_buffer("all_sensor_idx", all_sensor_idx)

        if mean_flow is None:
            mean_flow = torch.zeros(2 * h * w, dtype=pod_basis.dtype, device=pod_basis.device)
        self.register_buffer("mean_flow", mean_flow.float().reshape(-1))
        self.register_buffer("mean_sensor", self.mean_flow[all_sensor_idx])

        phi_p = pod_basis[all_sensor_idx, :]  # (2*Ps, K)
        self.register_buffer("phi_p", phi_p)

        # Precompute projection operator P_inv = (Phi_p^T * Phi_p + lambda * I)^(-1) * Phi_p^T
        gram = phi_p.T @ phi_p + reg_lambda * torch.eye(self.k, device=phi_p.device)
        proj_op = torch.linalg.solve(gram, phi_p.T)  # (K, 2*Ps)
        self.register_buffer("proj_op", proj_op)

    def reconstruct_from_sensors(self, sensor_values: torch.Tensor) -> torch.Tensor:
        """
        sensor_values: [B, T, Ps, 2]
        Returns: [B, T, H, W, 2]
        """
        b, t, ps, c = sensor_values.shape
        # Flatten u and v: [B, T, 2*Ps]
        u_val = sensor_values[..., 0]  # [B, T, Ps]
        v_val = sensor_values[..., 1]  # [B, T, Ps]
        y_vec = torch.cat([u_val, v_val], dim=-1)  # [B, T, 2*Ps]

        # Solve modal coefficients a: [B, T, K]
        # proj_op: [K, 2*Ps]
        a = torch.einsum("btp,kp->btk", y_vec - self.mean_sensor.to(y_vec.device), self.proj_op)

        # Reconstruct full field: [B, T, 2*M]
        u_full_flat = torch.einsum("btk,mk->btm", a, self.pod_basis)
        u_full_flat = u_full_flat + self.mean_flow.to(u_full_flat.device)
        m = self.h * self.w
        u = u_full_flat[..., :m].reshape(b, t, self.h, self.w)
        v = u_full_flat[..., m:].reshape(b, t, self.h, self.w)
        return torch.stack([u, v], dim=-1)  # [B, T, H, W, 2]

    def forward(self, batch: dict) -> torch.Tensor:
        sensor_values = batch["sensor_values"]  # [B, Tin, Ps, 2]
        # Reconstruct historical full field
        u_recon = self.reconstruct_from_sensors(sensor_values)  # [B, Tin, H, W, 2]
        # Persistence forecasting (or repeat last step) for classical baseline
        last_step = u_recon[:, -1:, ...].repeat(1, self.out_time, 1, 1, 1)
        return last_step
