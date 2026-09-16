import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model
from .metrics import mse_loss


class PhysicsAttention3D(nn.Module):
    """Physics-informed attention mechanism partitioning continuous spatiotemporal fields into physical slices.
    
    Reference: 'Transolver: A Fast Transformer Solver for PDEs on General Geometries' (ICML 2024).
    Linear complexity: O(N * G + G^2), where N is number of mesh tokens and G is number of physical slices.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, slice_num: int = 32, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.scale = dim_head ** -0.5

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, dim]
        b, n, _ = x.shape

        # Step 1: Compute physical slice assignment weights
        fx_mid = self.in_project_fx(x).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)  # [B, H, N, D]
        x_mid = self.in_project_x(x).reshape(b, n, self.heads, self.dim_head).transpose(1, 2)   # [B, H, N, D]

        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_logits = self.in_project_slice(x_mid) / temp  # [B, H, N, G]
        slice_weights = torch.softmax(slice_logits, dim=-1)

        slice_norm = slice_weights.sum(dim=2).clamp_min(1e-5).unsqueeze(-1)  # [B, H, G, 1]
        slice_tokens = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / slice_norm  # [B, H, G, D]

        # Step 2: Attention among physical slice tokens (G x G)
        q = self.to_q(slice_tokens)
        k = self.to_k(slice_tokens)
        v = self.to_v(slice_tokens)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, G, G]
        attn = torch.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)  # [B, H, G, D]

        # Step 3: Deslice back to mesh points
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)  # [B, H, N, D]
        out_x = out_x.transpose(1, 2).reshape(b, n, -1)                    # [B, N, H*D]
        return self.to_out(out_x)


class TransolverBlock3D(nn.Module):
    """Transolver building block: LayerNorm -> PhysicsAttention3D -> LayerNorm -> MLP."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, slice_num: int = 32, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention3D(dim, heads=heads, dim_head=dim_head, slice_num=slice_num, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transolver3d(Model):
    """3D Transolver operator for RealPDEBench fluid mechanics benchmarks.
    
    Applies Physics Attention to structured spatiotemporal meshes,
    grouping points into physical state slices for linear-complexity global modeling.
    """
    def __init__(
        self,
        n_layers: int = 4,
        width: int = 64,
        heads: int = 8,
        slice_num: int = 32,
        patch_size: tuple[int, int] = (2, 2),
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
        self.gh, self.gw = height_in // self.ph, width_in // self.pw
        self.num_space = self.gh * self.gw

        in_channels = c_in + 3  # (u, v) + (t, x, y)
        patch_dim = in_channels * self.ph * self.pw

        self.embed = nn.Linear(patch_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_in * self.num_space, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransolverBlock3D(
                dim=width,
                heads=heads,
                dim_head=max(16, width // heads),
                slice_num=slice_num,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        if time_in != time_out:
            self.time_project = nn.Linear(time_in, time_out)
        else:
            self.time_project = nn.Identity()

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

        grid = self.get_grid(x.shape, device)
        x_with_grid = torch.cat((x, grid), dim=-1)

        # Patchify
        x_patches = x_with_grid.reshape(b, t_in, self.gh, self.ph, self.gw, self.pw, -1)
        x_patches = x_patches.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t_in * self.num_space, -1)

        tokens = self.embed(x_patches) + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        # Reshape to [B, T_in, S, width]
        tokens = tokens.reshape(b, t_in, self.num_space, self.width)

        # Time adaptation
        tokens = tokens.permute(0, 2, 3, 1)  # [B, S, width, T_in]
        tokens = self.time_project(tokens)   # [B, S, width, T_out]
        tokens = tokens.permute(0, 3, 1, 2)  # [B, T_out, S, width]

        out_patches = self.head(tokens)  # [B, T_out, S, ph * pw * c_out]
        c_out = self.shape_out[-1]
        out = out_patches.reshape(b, self.num_time_out, self.gh, self.gw, self.ph, self.pw, c_out)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, self.num_time_out, h, w, c_out)
        return out

    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)
