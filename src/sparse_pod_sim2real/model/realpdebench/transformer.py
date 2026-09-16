import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model
from .metrics import mse_loss


class SpaceTimeAttention(nn.Module):
    """Divided Space-Time Multi-Head Attention block.
    
    Decouples computation into spatial attention (within each time snapshot)
    and temporal attention (across snapshots for each spatial patch).
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        # Spatial attention
        self.norm_space = nn.LayerNorm(dim)
        self.to_qkv_space = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_space = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Temporal attention
        self.norm_time = nn.LayerNorm(dim)
        self.to_qkv_time = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_time = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # Feed-forward network
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _attention(self, qkv, num_tokens):
        # qkv: [Batch, Tokens, 3 * Heads * DimHead]
        b, n, _ = qkv.shape
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)  # [B, H, N, D]
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B, H, N, D]
        out = out.transpose(1, 2).reshape(b, n, -1)
        return out

    def forward(self, x, num_time: int, num_space: int):
        # x: [B, T * S, dim]
        b, total_tokens, dim = x.shape

        # 1. Spatial Attention (reshape to [B * T, S, dim])
        res = x
        x_space = self.norm_space(x).reshape(b * num_time, num_space, dim)
        qkv_s = self.to_qkv_space(x_space)
        out_s = self._attention(qkv_s, num_space)
        out_s = self.to_out_space(out_s).reshape(b, total_tokens, dim)
        x = res + out_s

        # 2. Temporal Attention (reshape to [B * S, T, dim])
        res = x
        x_time = self.norm_time(x).reshape(b, num_time, num_space, dim).permute(0, 2, 1, 3).reshape(b * num_space, num_time, dim)
        qkv_t = self.to_qkv_time(x_time)
        out_t = self._attention(qkv_t, num_time)
        out_t = self.to_out_time(out_t).reshape(b, num_space, num_time, dim).permute(0, 2, 1, 3).reshape(b, total_tokens, dim)
        x = res + out_t

        # 3. FFN
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Transformer3d(Model):
    """Spatio-Temporal Transformer operator for RealPDEBench 3D PDE solving.
    
    Supports grid coordinate injection (t, x, y), patchified space-time tokens,
    divided space-time attention blocks, and linear projection to future horizon.
    """
    def __init__(
        self,
        n_layers: int = 4,
        width: int = 64,
        heads: int = 8,
        patch_size: tuple[int, int] = (4, 4),
        dropout: float = 0.0,
        shape_in: tuple[int, int, int, int] = (20, 64, 128, 2),
        shape_out: tuple[int, int, int, int] = (20, 64, 128, 2),
    ):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.patch_size = patch_size
        self.width = width
        self.n_layers = n_layers

        time_in, height_in, width_in, c_in = shape_in
        time_out, height_out, width_out, c_out = shape_out

        self.num_time_in = time_in
        self.num_time_out = time_out
        self.ph, self.pw = patch_size
        self.num_space = (height_in // self.ph) * (width_in // self.pw)

        # Injected coordinate grid (t, x, y) -> 3 additional channels
        in_channels = c_in + 3
        patch_dim = in_channels * self.ph * self.pw

        self.patch_embed = nn.Linear(patch_dim, width)
        self.pos_space = nn.Parameter(torch.zeros(1, 1, self.num_space, width))
        self.pos_time = nn.Parameter(torch.zeros(1, time_in, 1, width))
        nn.init.trunc_normal_(self.pos_space, std=0.02)
        nn.init.trunc_normal_(self.pos_time, std=0.02)

        self.blocks = nn.ModuleList([
            SpaceTimeAttention(
                dim=width,
                heads=heads,
                dim_head=max(16, width // heads),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Temporal horizon mapping: maps T_in tokens to T_out tokens
        if time_in != time_out:
            self.time_project = nn.Linear(time_in, time_out)
        else:
            self.time_project = nn.Identity()

        # Output patch unflattening
        out_patch_dim = c_out * self.ph * self.pw
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, out_patch_dim),
        )

    def get_grid(self, shape, device):
        batchsize, size_t, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float, device=device)
        gridt = gridt.reshape(1, size_t, 1, 1, 1).repeat([batchsize, 1, size_x, size_y, 1])
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, size_t, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, 1, size_y, 1).repeat([batchsize, size_t, size_x, 1, 1])
        return torch.cat((gridt, gridx, gridy), dim=-1)

    def forward(self, x):
        # x: [B, T_in, H, W, C_in]
        b, t_in, h, w, c = x.shape
        device = x.device

        # Coordinate injection
        grid = self.get_grid(x.shape, device)
        x_with_grid = torch.cat((x, grid), dim=-1)  # [B, T, H, W, C+3]

        # Patchify: [B, T, H//ph, ph, W//pw, pw, C+3] -> [B, T, S, ph * pw * (C+3)]
        gh, gw = h // self.ph, w // self.pw
        x_patches = x_with_grid.reshape(b, t_in, gh, self.ph, gw, self.pw, -1)
        x_patches = x_patches.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t_in, gh * gw, -1)

        # Linear embedding and spatio-temporal positional encoding
        tokens = self.patch_embed(x_patches) + self.pos_space + self.pos_time  # [B, T, S, width]
        tokens = tokens.reshape(b, t_in * self.num_space, self.width)

        for block in self.blocks:
            tokens = block(tokens, num_time=t_in, num_space=self.num_space)

        # Reshape back to [B, T_in, S, width]
        tokens = tokens.reshape(b, t_in, self.num_space, self.width)

        # Time horizon adaptation: [B, S, width, T_in] -> [B, S, width, T_out]
        tokens = tokens.permute(0, 2, 3, 1)  # [B, S, width, T_in]
        tokens = self.time_project(tokens)   # [B, S, width, T_out]
        tokens = tokens.permute(0, 3, 1, 2)  # [B, T_out, S, width]

        # Decode patches to continuous field
        out_patches = self.head(tokens)  # [B, T_out, S, ph * pw * C_out]
        c_out = self.shape_out[-1]
        out = out_patches.reshape(b, self.num_time_out, gh, gw, self.ph, self.pw, c_out)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_time_out, h, w, c_out)
        return out

    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)
