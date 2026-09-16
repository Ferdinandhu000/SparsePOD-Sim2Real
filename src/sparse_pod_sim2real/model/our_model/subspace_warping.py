from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class GrassmannSubspaceAlignment(nn.Module):
    """
    Genuine Grassmannian Subspace Alignment on Gr(K, d).
    
    Given the simulation basis Phi_sim in R^(d x K) and its energetic orthogonal
    complement Phi_perp in R^(d x r) (where Phi_perp^T Phi_sim = 0):
        Phi_adapted = qr(Phi_sim + Phi_perp @ A)
    where A in R^(r x K) is a learnable tangent space coordinate matrix initialized to 0.
    
    When A = 0: Phi_adapted = Phi_sim (initial Grassmannian point).
    When A != 0: span(Phi_adapted) genuinely rotates on Gr(K, d) into the orthogonal
    directions of Phi_sim, adapting to real-world viscous/boundary-layer physics.
    
    Also provides exact Principal Angles computation to measure domain shift distance.
    """

    def __init__(
        self,
        k: int = 64,
        r: int = 16,
        phi_sim: Optional[torch.Tensor] = None,
        phi_perp: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.k = k
        self.r = r

        if phi_sim is not None:
            self.register_buffer("phi_sim", phi_sim.float())
        else:
            self.phi_sim = None

        if phi_perp is not None:
            self.register_buffer("phi_perp", phi_perp.float())
            self.has_perp = True
        else:
            self.phi_perp = None
            self.has_perp = False

        # Tangent space transition matrix A in R^(r x K), initialized to 0
        if self.has_perp:
            self.A = nn.Parameter(torch.zeros(r, k))
        else:
            # Fallback coordinate rotation if no complement is provided
            self.s_raw = nn.Parameter(torch.zeros(k, k))

    def get_adapted_basis(self, base_phi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the adapted orthonormal basis Phi_real in R^(d x K).
        """
        phi_0 = self.phi_sim if self.phi_sim is not None else base_phi
        if phi_0 is None:
            raise ValueError("No base POD basis provided for Grassmann adaptation.")

        if self.has_perp and self.phi_perp is not None:
            # Move on Grassmann manifold along tangent direction Phi_perp @ A
            perturbed = phi_0 + self.phi_perp @ self.A
            # Retract to Stiefel/Grassmann manifold via QR decomposition
            q, _ = torch.linalg.qr(perturbed, mode="reduced")
            return q
        else:
            # Skew-symmetric coordinate rotation (Cayley transform)
            s = 0.5 * (self.s_raw - self.s_raw.T)
            eye = torch.eye(self.k, device=s.device, dtype=s.dtype)
            w = torch.linalg.solve(eye + s, eye - s)
            return phi_0 @ w

    @torch.no_grad()
    def compute_principal_angles(self, base_phi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute principal angles between Phi_sim and Phi_adapted in degrees:
            theta_i = arccos(sigma_i(Phi_sim^T Phi_adapted))
        """
        phi_0 = self.phi_sim if self.phi_sim is not None else base_phi
        if phi_0 is None:
            return torch.zeros(self.k)

        phi_adapt = self.get_adapted_basis(phi_0)
        # Cosine matrix: [K, K]
        m = phi_0.T @ phi_adapt
        s = torch.linalg.svdvals(m)
        s_clamped = torch.clamp(s, 0.0, 1.0)
        angles_rad = torch.acos(s_clamped)
        angles_deg = angles_rad * (180.0 / torch.pi)
        return angles_deg

    def regularization_loss(self) -> torch.Tensor:
        """Frobenius norm regularization on tangent displacement A."""
        if self.has_perp and hasattr(self, "A"):
            return torch.norm(self.A, p="fro") ** 2
        return torch.tensor(0.0, device=self.s_raw.device)

