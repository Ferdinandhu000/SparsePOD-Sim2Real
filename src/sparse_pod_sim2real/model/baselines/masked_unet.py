from __future__ import annotations

import torch
import torch.nn as nn

from ..realpdebench.unet import Unet3d as UNet


class MaskedUNet3D(nn.Module):
    """
    3D UNet baseline adapted for sparse masked fluid inputs.
    Input: [B, Tin, H, W, 3] (u, v, mask)
    Output: [B, Tout, H, W, 2] (u, v)
    """

    def __init__(
        self,
        in_time: int = 20,
        out_time: int = 20,
        channels: int = 3,
        out_channels: int = 2,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        **kwargs,
    ):
        super().__init__()
        self.in_time = in_time
        self.out_time = out_time
        self.unet = UNet(
            dim=dim,
            channels=channels,
            out_channels=out_channels,
            dim_mults=dim_mults,
            in_time=in_time,
            out_time=out_time,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        x: [B, Tin, H, W, 3] or dict containing 'x_sparse'
        Returns: [B, Tout, H, W, 2]
        """
        if isinstance(x, dict):
            x = x["x_sparse"]
        return self.unet(x)
