from __future__ import annotations

import torch


class IdentityNormalizer:
    def __init__(self, device=None):
        self.device = device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device):
        self.device = device
        return self


class GaussianNormalizer:
    """Zero mean, unit variance normalization per channel."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6, device=None):
        self.eps = eps
        self.mean = mean
        self.std = torch.clamp(std, min=eps)
        self.device = device
        if device is not None:
            self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean

    def to(self, device):
        self.device = device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class RangeNormalizer:
    """Scales data to [-1, 1] using min and max."""

    def __init__(self, min_val: torch.Tensor, max_val: torch.Tensor, eps: float = 1e-6, device=None):
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val
        self.device = device
        if device is not None:
            self.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        min_v = self.min_val.to(x.device)
        max_v = self.max_val.to(x.device)
        return 2.0 * (x - min_v) / (max_v - min_v + self.eps) - 1.0

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        min_v = self.min_val.to(x.device)
        max_v = self.max_val.to(x.device)
        return (x + 1.0) * 0.5 * (max_v - min_v + self.eps) + min_v

    def to(self, device):
        self.device = device
        self.min_val = self.min_val.to(device)
        self.max_val = self.max_val.to(device)
        return self
