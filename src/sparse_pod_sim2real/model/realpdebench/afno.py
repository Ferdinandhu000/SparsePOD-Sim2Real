import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model
from .metrics import mse_loss


class AdaptiveFourierFilter3d(nn.Module):
    """3D Adaptive Fourier Filter with block-diagonal MLP and softshrink sparsity."""
    def __init__(self, channels: int, blocks: int = 8, sparsity_threshold: float = 0.01):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.block_size = channels // blocks
        self.sparsity_threshold = sparsity_threshold

        scale = 1.0 / (self.block_size * self.block_size)
        self.w1 = nn.Parameter(scale * torch.rand(2, self.blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(scale * torch.rand(2, self.blocks, self.block_size))
        self.w2 = nn.Parameter(scale * torch.rand(2, self.blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(scale * torch.rand(2, self.blocks, self.block_size))

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        orig_dtype = x.dtype
        x_ft = torch.fft.rfftn(x.float(), dim=[-3, -2, -1], norm="ortho")  # [B, C, T, H, W//2+1]
        freq_w = x_ft.size(-1)

        # Reshape to [B, T, H, freq_w, blocks, block_size]
        x_reshaped = x_ft.permute(0, 2, 3, 4, 1).reshape(b, t, h, freq_w, self.blocks, self.block_size)
        real = x_reshaped.real
        imag = x_reshaped.imag

        # Block 1
        first_real = (
            torch.einsum("...gi,gio->...go", real, self.w1[0])
            - torch.einsum("...gi,gio->...go", imag, self.w1[1])
            + self.b1[0]
        )
        first_imag = (
            torch.einsum("...gi,gio->...go", real, self.w1[1])
            + torch.einsum("...gi,gio->...go", imag, self.w1[0])
            + self.b1[1]
        )
        first_real, first_imag = F.gelu(first_real), F.gelu(first_imag)

        # Block 2
        out_real = (
            torch.einsum("...go,goi->...gi", first_real, self.w2[0])
            - torch.einsum("...go,goi->...gi", first_imag, self.w2[1])
            + self.b2[0]
        )
        out_imag = (
            torch.einsum("...go,goi->...gi", first_real, self.w2[1])
            + torch.einsum("...go,goi->...gi", first_imag, self.w2[0])
            + self.b2[1]
        )

        out = torch.stack([out_real, out_imag], dim=-1)
        out = F.softshrink(out, lambd=self.sparsity_threshold)
        out_complex = torch.view_as_complex(out.contiguous()).reshape(b, t, h, freq_w, c)
        out_ft = out_complex.permute(0, 4, 1, 2, 3)

        return torch.fft.irfftn(out_ft, s=(t, h, w), dim=[-3, -2, -1], norm="ortho").to(dtype=orig_dtype)



class AFNO3d(Model):
    """3D AFNO following RealPDEBench FNO3d conventions with adaptive Fourier blocks."""
    def __init__(self, n_layers: int, width: int, shape_in, shape_out, blocks: int = 8, sparsity_threshold: float = 0.01):
        super(AFNO3d, self).__init__()

        self.width = width
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.dim_in = shape_in[-1]
        self.dim_out = shape_out[-1] * shape_out[0] // shape_in[0]
        self.padding = 6

        self.fc0 = nn.Linear(self.dim_in + 3, self.width)

        self.n_layers = n_layers
        actual_blocks = blocks
        while actual_blocks > 1 and (self.width % actual_blocks != 0):
            actual_blocks -= 1

        self.spectral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(AdaptiveFourierFilter3d(self.width, blocks=actual_blocks, sparsity_threshold=sparsity_threshold))
            self.convs.append(nn.Conv3d(self.width, self.width, 1))
            self.bns.append(nn.BatchNorm3d(self.width))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.dim_out)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = self.bns[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x[..., :-self.padding, :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(*x.shape[:-1], self.shape_out[-1], self.shape_out[0] // self.shape_in[0])
        target_shape = (x.shape[0], self.shape_out[0], x.shape[2], x.shape[3], self.shape_out[-1])
        out = x.permute(0, 1, 5, 2, 3, 4).reshape(*target_shape)
        return out



    def train_loss(self, input, target):
        pred = self.forward(input)
        return mse_loss(pred, target)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
