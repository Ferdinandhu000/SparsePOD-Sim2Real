import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model
from .metrics import mse_loss
from .transolver import PhysicsAttention3D


class iTransolverBlock(nn.Module):
    """Inverted Transolver Block: LayerNorm -> PhysicsAttention across inverted dynamic tokens -> MLP."""
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


class iTransolver3d(Model):
    """SOTA Inverted Transolver Operator for 3D PDE continuous fluid flow forecasting.
    
    Combines iTransformer's inverted trajectory representation with Transolver's
    Physics Attention across coherent fluid dynamic slices.
    
    Key Innovations:
    1. Temporal Trajectory Inversion: Maps entire T_in history of each spatial location
       into high-dimensional dynamic tokens, preventing autoregressive error drift.
    2. Physics Slicing on Dynamic Manifolds: Clusters spatial trajectory tokens into
       intrinsic dynamic modes (wake vortex, shear layer, circulation).
    3. Multi-Variate Triad Coupling: Jointly attends across velocity components (u, v).
    4. Non-stationary normalization per spatial trajectory.
    """
    def __init__(
        self,
        n_layers: int = 4,
        width: int = 64,
        heads: int = 8,
        slice_num: int = 32,
        patch_size: tuple[int, int] = (2, 2),
        dropout: float = 0.0,
        use_norm: bool = True,
        shape_in: tuple[int, int, int, int] = (20, 64, 128, 2),
        shape_out: tuple[int, int, int, int] = (20, 64, 128, 2),
    ):
        super().__init__()
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.patch_size = patch_size
        self.width = width
        self.n_layers = n_layers
        self.use_norm = use_norm

        time_in, height_in, width_in, c_in = shape_in
        time_out, height_out, width_out, c_out = shape_out
        self.num_time_in = time_in
        self.num_time_out = time_out
        self.c_in = c_in
        self.c_out = c_out

        self.ph, self.pw = patch_size
        self.gh, self.gw = height_in // self.ph, width_in // self.pw
        self.num_space = self.gh * self.gw

        # Inverted embedding: time_in steps + spatial patch area
        # Each token represents an individual spatial patch trajectory for a specific channel
        in_traj_dim = time_in * self.ph * self.pw
        self.traj_embed = nn.Linear(in_traj_dim, width)

        # Coordinate and channel embeddings
        self.coord_embed = nn.Linear(2, width)
        self.channel_embed = nn.Embedding(c_in, width)

        self.blocks = nn.ModuleList([
            iTransolverBlock(
                dim=width,
                heads=heads,
                dim_head=max(16, width // heads),
                slice_num=min(slice_num, self.num_space * c_in),
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Inverted direct projection to future horizon
        out_traj_dim = time_out * self.ph * self.pw
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, out_traj_dim),
        )

    def get_spatial_grid(self, device):
        # [gh, gw, 2] normalized spatial coordinates for patch centers
        gx = torch.linspace(0, 1, self.gh, device=device)
        gy = torch.linspace(0, 1, self.gw, device=device)
        mesh_x, mesh_y = torch.meshgrid(gx, gy, indexing="ij")
        grid = torch.stack((mesh_x, mesh_y), dim=-1).reshape(self.num_space, 2)
        return grid

    def forward(self, x):
        # x: [B, T_in, H, W, C_in]
        b, t_in, h, w, c = x.shape
        device = x.device

        # Patchify and invert: [B, C, gh, gw, ph, pw, T_in] -> [B, C * num_space, ph * pw * T_in]
        # First permute to [B, C, gh, ph, gw, pw, T_in]
        x_perm = x.permute(0, 4, 2, 3, 1)  # [B, C, H, W, T_in]
        x_patches = x_perm.reshape(b, c, self.gh, self.ph, self.gw, self.pw, t_in)
        x_patches = x_patches.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, c, self.num_space, self.ph * self.pw * t_in)
        tokens = x_patches.reshape(b, c * self.num_space, self.ph * self.pw * t_in)  # [B, N_tokens, in_traj_dim]

        # Non-stationary temporal trajectory normalization
        if self.use_norm:
            means = tokens.mean(dim=-1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(tokens, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
            norm_tokens = (tokens - means) / stdev
        else:
            norm_tokens = tokens

        # Embed inverted trajectories
        h_tokens = self.traj_embed(norm_tokens)  # [B, c * num_space, width]

        # Add spatial coordinate and channel embeddings
        s_grid = self.get_spatial_grid(device)      # [num_space, 2]
        s_pos = self.coord_embed(s_grid).repeat(c, 1)  # [c * num_space, width]
        c_ids = torch.arange(c, device=device).repeat_interleave(self.num_space)
        c_pos = self.channel_embed(c_ids)           # [c * num_space, width]

        h_tokens = h_tokens + s_pos.unsqueeze(0) + c_pos.unsqueeze(0)

        # Physics Attention across dynamic slices
        for block in self.blocks:
            h_tokens = block(h_tokens)

        # Direct projection to future horizon
        out_traj = self.head(h_tokens)  # [B, c * num_space, ph * pw * t_out]

        # Denormalize
        if self.use_norm:
            out_traj = out_traj * stdev + means

        # Unflatten to [B, T_out, H, W, C_out]
        out_patches = out_traj.reshape(b, c, self.gh, self.gw, self.ph, self.pw, self.num_time_out)
        out_patches = out_patches.permute(0, 6, 2, 4, 3, 5, 1)  # [B, T_out, gh, ph, gw, pw, C]
        out = out_patches.reshape(b, self.num_time_out, h, w, c)
        return out

    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)
