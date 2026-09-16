from __future__ import annotations

import torch
import torch.nn as nn


class SubspaceWarping(nn.Module):
    """
    Grassmannian Subspace Alignment Matrix W_align in R^(K x K).
    Rotates the simulation POD basis Phi_sim to align with real-world flow manifold.
    Can be parameterized via Cayley transform (strictly orthogonal) or unconstrained with orthogonality penalty.
    """

    def __init__(self, k: int = 64, orthogonal: bool = True):
        super().__init__()
        self.k = k
        self.orthogonal = orthogonal

        if orthogonal:
            # Parameterize skew-symmetric matrix S (S^T = -S)
            # Cayley transform: W = (I - S)(I + S)^(-1) is strictly orthogonal: W^T W = I
            self.s_raw = nn.Parameter(torch.zeros(k, k))
        else:
            # Learnable general transformation initialized to identity
            self.weight = nn.Parameter(torch.eye(k))

    def get_matrix(self) -> torch.Tensor:
        if self.orthogonal:
            # S = 0.5 * (s_raw - s_raw^T)
            s = 0.5 * (self.s_raw - self.s_raw.T)
            eye = torch.eye(self.k, device=s.device, dtype=s.dtype)
            # W = (I - S) @ (I + S)^(-1)
            w = torch.linalg.solve(eye + s, eye - s)
            return w
        else:
            return self.weight

    def orthogonality_loss(self) -> torch.Tensor:
        """Returns ||W^T W - I||_F^2 if not strictly orthogonal."""
        if self.orthogonal:
            return torch.tensor(0.0, device=self.s_raw.device)
        w = self.weight
        eye = torch.eye(self.k, device=w.device, dtype=w.dtype)
        return torch.norm(w.T @ w - eye, p="fro") ** 2
