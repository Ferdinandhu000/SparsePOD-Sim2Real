from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .gappy_solver import DifferentiableGappySolver
from .subspace_warping import GrassmannSubspaceAlignment
from ..realpdebench.unet import Unet3d as UNet


class ModalLinearPropagator(nn.Module):
    """Lightweight modal time-stepper predicting future modal coefficients from history."""

    def __init__(self, k: int = 64, in_time: int = 20, out_time: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.k = k
        self.in_time = in_time
        self.out_time = out_time

        self.net = nn.Sequential(
            nn.Linear(in_time * k, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, out_time * k),
        )

    def forward(self, a_in: torch.Tensor) -> torch.Tensor:
        # a_in: [B, Tin, K]
        b = a_in.shape[0]
        flat_in = a_in.reshape(b, -1)
        flat_out = self.net(flat_in)
        return flat_out.reshape(b, self.out_time, self.k)


class PODResUNet3DSparse(nn.Module):
    """
    FLAGSHIP SOTA MODEL: Physics-Lifting Grassmannian Residual Operator.
    
    1. Differentiable Gappy-POD lifts sparse sensor measurements to a globally coherent coarse flow field
       using Reynolds-decomposed mean flow + POD fluctuating modes.
    2. GrassmannSubspaceAlignment moves the basis along orthogonal complement directions Phi_perp on Gr(K, d)
       to capture real-world domain shifts without destroying the physical subspace.
    3. Heavy 23M 3D-UNet operates directly with in_time=20, out_time=20 to predict the future high-frequency
       residual field Delta u = u_real - u_pod.
    """

    def __init__(
        self,
        pod_basis: Union[torch.Tensor, dict],
        sensor_indices: torch.Tensor,
        h: int = 64,
        w: int = 128,
        in_time: int = 20,
        out_time: int = 20,
        k: int = 64,
        reg_lambda: float = 1e-4,
        unet_dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        use_warping: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.in_time = in_time
        self.out_time = out_time
        self.k = k
        self.residual_weight = residual_weight
        self.use_warping = use_warping

        # Parse basis payload (supports both Tensor and Dict with mean & orthogonal complement)
        if isinstance(pod_basis, dict):
            basis_tensor = pod_basis["basis"][:, :k]
            mean_tensor = pod_basis.get("mean", None)
            perp_tensor = pod_basis.get("basis_perp", None)
        else:
            basis_tensor = pod_basis[:, :k]
            mean_tensor = None
            perp_tensor = None

        # 1. Gappy-POD physical lifting module (with Reynolds decomposition)
        self.gappy = DifferentiableGappySolver(
            pod_basis=basis_tensor,
            sensor_indices=sensor_indices,
            mean_flow=mean_tensor,
            h=h,
            w=w,
            reg_lambda=reg_lambda,
        )

        # 2. Grassmann Subspace Alignment on Gr(K, d)
        r = perp_tensor.shape[1] if perp_tensor is not None else 16
        self.warping = (
            GrassmannSubspaceAlignment(k=k, r=r, phi_sim=basis_tensor, phi_perp=perp_tensor)
            if use_warping
            else None
        )

        # 3. Modal temporal propagator
        self.modal_prop = ModalLinearPropagator(k=k, in_time=in_time, out_time=out_time)

        # 4. High-capacity 23M 3D-UNet for spatio-temporal residual refinement
        # Inputs: in_time=20 coarse lifted physical field -> Outputs: out_time=20 high-frequency residual
        self.unet = UNet(
            dim=unet_dim,
            channels=2,
            out_channels=2,
            dim_mults=dim_mults,
            in_time=in_time,
            out_time=out_time,
        )

    def get_adapted_basis(self) -> Optional[torch.Tensor]:
        return self.warping.get_adapted_basis(self.gappy.pod_basis) if self.warping is not None else None

    def compute_principal_angles(self) -> torch.Tensor:
        """Returns principal angles between Sim and Real subspaces in degrees."""
        if self.warping is not None:
            return self.warping.compute_principal_angles(self.gappy.pod_basis)
        return torch.zeros(self.k)

    def forward(self, batch: dict | torch.Tensor, mode: str = "sparse", return_components: bool = False):
        """
        mode:
          - "full": Exact dense POD projection for Sim pre-training. This bypasses
            Gappy inversion while retaining modal dynamics, POD decoding, and residual refinement.
          - "sparse": Gappy-POD physical lifting from sparse sensors -> UNet residual refinement.
        """
        if isinstance(batch, dict):
            if mode == "full" and "x_full" in batch:
                a_in, u_in_pod = self.gappy.project_full_field(batch["x_full"])
                a_out = self.modal_prop(a_in)
                u_out_pod = self.gappy.decode_field(a_out)
                delta_u = self.unet(u_in_pod)
                u_final = u_out_pod + self.residual_weight * delta_u
                if return_components:
                    return {
                        "u_final": u_final,
                        "u_pod": u_out_pod,
                        "delta_u": delta_u,
                        "a_in": a_in,
                        "a_out": a_out,
                    }
                return u_final
            sensor_values = batch.get("sensor_values", None)
            if sensor_values is None and "x_full" in batch:
                return self.unet(batch["x_full"])
        else:
            sensor_values = batch

        adapted_basis = self.get_adapted_basis()

        # Step 1: Solve historical modal coefficients from sparse sensors using Reynolds-decomposed Gappy-POD
        # a_in: [B, Tin, K], u_in_pod: [B, Tin, H, W, 2]
        a_in, u_in_pod = self.gappy(sensor_values, adapted_basis=adapted_basis)

        # Step 2: Propagate modal coefficients into future
        # a_out: [B, Tout, K]
        a_out = self.modal_prop(a_in)

        # Step 3: Decode future coarse physical field
        # u_out_pod: [B, Tout, H, W, 2]
        u_out_pod = self.gappy.decode_field(a_out, adapted_basis=adapted_basis)

        # Step 4: High-capacity 3D UNet receives the in_time=20 coarse physical flow field
        # and directly predicts the out_time=20 spatio-temporal residual
        # delta_u: [B, Tout, H, W, 2]
        delta_u = self.unet(u_in_pod)

        # Step 5: Full-field synthesis (Physics Prior + Neural Residual)
        u_final = u_out_pod + self.residual_weight * delta_u

        if return_components:
            return {
                "u_final": u_final,
                "u_pod": u_out_pod,
                "delta_u": delta_u,
                "a_in": a_in,
                "a_out": a_out,
                "principal_angles": self.compute_principal_angles(),
            }
        return u_final
