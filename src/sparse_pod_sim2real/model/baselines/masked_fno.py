from __future__ import annotations

import torch
import torch.nn as nn

from ..realpdebench.fno import FNO3d


class MaskedFNO3D(nn.Module):
    """
    3D FNO baseline adapted for sparse masked fluid inputs.
    Input: [B, Tin, H, W, 3] (u, v, mask)
    Output: [B, Tout, H, W, 2] (u, v)
    """

    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 32,
        n_layers: int = 4,
        shape_in: tuple = (20, 64, 128, 3),
        shape_out: tuple = (20, 64, 128, 2),
        **kwargs,
    ):
        super().__init__()
        self.fno = FNO3d(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            shape_in=shape_in,
            shape_out=shape_out,
            n_layers=n_layers,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(x, dict):
            x = x["x_sparse"]
        return self.fno(x)
