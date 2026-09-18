from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from ..our_model.gappy_solver import DifferentiableGappySolver


class LinearModalAR(nn.Module):
    """One-step linear modal dynamics rolled forward autoregressively."""

    def __init__(self, rank: int, out_time: int):
        super().__init__()
        self.out_time = int(out_time)
        self.transition = nn.Linear(rank, rank)
        with torch.no_grad():
            self.transition.weight.copy_(torch.eye(rank))
            self.transition.bias.zero_()

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        state = coefficients[:, -1]
        future = []
        for _ in range(self.out_time):
            state = self.transition(state)
            future.append(state)
        return torch.stack(future, dim=1)


class GappyLinearAR(nn.Module):
    """Gappy-POD observation model followed by a learned linear modal AR model.

    Dense Sim histories use an exact POD projection during source training. Sparse
    Real histories use the same frozen sensor layout and regularized Gappy solve.
    Only the small modal transition is adapted by the default Real fine-tuning
    scope, making this a strong, low-capacity comparison for neural residual models.
    """

    def __init__(
        self,
        pod_basis: Union[torch.Tensor, dict],
        sensor_indices: torch.Tensor,
        h: int = 64,
        w: int = 128,
        out_time: int = 20,
        rank: int = 64,
        reg_lambda: float = 1e-4,
    ):
        super().__init__()
        if isinstance(pod_basis, dict):
            basis = pod_basis["basis"][:, :rank]
            mean = pod_basis.get("mean")
        else:
            basis = pod_basis[:, :rank]
            mean = None
        self.gappy = DifferentiableGappySolver(
            pod_basis=basis,
            sensor_indices=sensor_indices,
            mean_flow=mean,
            h=h,
            w=w,
            reg_lambda=reg_lambda,
        )
        self.modal_prop = LinearModalAR(basis.shape[1], out_time)

    def forward(self, batch: dict, mode: str = "sparse") -> torch.Tensor:
        if mode == "full":
            if "x_full" not in batch:
                raise ValueError("Dense Sim mode requires x_full.")
            coefficients, _ = self.gappy.project_full_field(batch["x_full"])
        elif mode == "sparse":
            if "sensor_values" not in batch:
                raise ValueError("Sparse mode requires sensor_values.")
            coefficients = self.gappy.solve_coefficients(batch["sensor_values"])
        else:
            raise ValueError(f"Unsupported GappyLinearAR mode: {mode!r}")
        return self.gappy.decode_field(self.modal_prop(coefficients))

