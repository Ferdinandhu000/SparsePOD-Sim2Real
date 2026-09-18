from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import scipy.linalg
import torch

TopologyType = Literal[
    "wall", "uniform", "wake_rake", "random", "q_deim",
    "group_q_deim", "channel_norm_qr",
]


class SensorTopology:
    """Manages sensor placement and sparse sampling operators for 2D flow fields."""

    def __init__(
        self,
        topology_type: TopologyType = "wall",
        num_sensors: int = 64,
        grid_shape: Tuple[int, int] = (64, 128),
        seed: int = 42,
        pod_basis: Optional[np.ndarray | torch.Tensor] = None,
        sensor_indices: Optional[np.ndarray | torch.Tensor] = None,
    ):
        self.topology_type = topology_type
        self.num_sensors = num_sensors
        self.grid_shape = grid_shape
        self.h, self.w = grid_shape
        self.seed = seed
        self.pod_basis = pod_basis

        if sensor_indices is None:
            self.indices_1d, self.coords, self.mask = self._generate_placement()
        else:
            self.indices_1d, self.coords, self.mask = self._placement_from_indices(sensor_indices)
        self._voronoi_assignment = self._build_voronoi_assignment()

    def _placement_from_indices(
        self, sensor_indices: np.ndarray | torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.as_tensor(sensor_indices, dtype=torch.long).reshape(-1)
        if len(indices) != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensor indices, received {len(indices)}."
            )
        if indices.unique().numel() != indices.numel():
            raise ValueError("Sensor manifest contains duplicate spatial indices.")
        if torch.any(indices < 0) or torch.any(indices >= self.h * self.w):
            raise ValueError("Sensor manifest contains out-of-range spatial indices.")
        self.selection_order_1d = indices.clone()
        indices = torch.sort(indices).values
        y_idx = indices // self.w
        x_idx = indices % self.w
        mask = torch.zeros(self.h, self.w, dtype=torch.float32)
        mask[y_idx, x_idx] = 1.0
        coords = torch.stack(
            [y_idx.float() / float(self.h - 1), x_idx.float() / float(self.w - 1)],
            dim=-1,
        )
        return indices, coords, mask

    def _build_voronoi_assignment(self) -> torch.Tensor:
        """Map every grid cell to its nearest sensor in normalized coordinates."""
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.h),
            torch.linspace(0.0, 1.0, self.w),
            indexing="ij",
        )
        grid = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        distances = torch.cdist(grid, self.coords)
        return distances.argmin(dim=-1).reshape(self.h, self.w)

    def _generate_placement(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = self.h, self.w
        p = self.num_sensors
        np_rng = np.random.RandomState(self.seed)

        if self.topology_type == "wall":
            # Exact NACA0025 profile in 64x128 downsampled grid
            # Center: (yc=32.0, xc=24.5), chord=35.0, relative thickness=0.25
            yc, xc = 32.0, 24.5
            chord = 35.0
            t_rel = 0.25

            def naca_half_thickness(x_val: float) -> float:
                s = (x_val - (xc - chord / 2.0)) / chord
                if s <= 0.0 or s >= 1.0:
                    return 0.0
                return 5.0 * t_rel * chord * (
                    0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * (s ** 2) + 0.2843 * (s ** 3) - 0.1015 * (s ** 4)
                )

            # Sample points along top surface and bottom surface
            # Half on upper surface, half on lower surface
            n_side = p // 2
            x_eval = np.linspace(xc - chord / 2.0 + 1.0, xc + chord / 2.0 - 1.0, n_side)

            wall_pts = []
            # Upper wall sensors: fluid cell immediately at/above upper surface
            for x in x_eval:
                yt = naca_half_thickness(x)
                gx = int(np.clip(np.round(x), 0, w - 1))
                gy = int(np.clip(np.ceil(yc + yt), 0, h - 1))
                wall_pts.append((gy, gx))

            # Lower wall sensors: fluid cell immediately at/below lower surface
            for x in reversed(x_eval):
                yt = naca_half_thickness(x)
                gx = int(np.clip(np.round(x), 0, w - 1))
                gy = int(np.clip(np.floor(yc - yt), 0, h - 1))
                wall_pts.append((gy, gx))

            # Deduplicate while preserving order
            seen = set()
            unique_indices = []
            for gy, gx in wall_pts:
                idx = gy * w + gx
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            if len(unique_indices) >= p:
                sel = np.linspace(0, len(unique_indices) - 1, p, dtype=int)
                flat_indices = np.array([unique_indices[i] for i in sel], dtype=np.int64)
            else:
                # If fewer unique surface grid cells than p, sample near-wall trailing/leading cells
                extra_idx = []
                for gy, gx in wall_pts:
                    for dy in [1, -1]:
                        cand_y = np.clip(gy + dy, 0, h - 1)
                        c_idx = cand_y * w + gx
                        if c_idx not in seen:
                            seen.add(c_idx)
                            extra_idx.append(c_idx)
                flat_indices = np.array((unique_indices + extra_idx)[:p], dtype=np.int64)


        elif self.topology_type == "uniform":
            # 2D regular grid subsampling
            step_h = math.sqrt((h * w) / p)
            step_w = step_h
            ny = max(2, int(round(h / step_h)))
            nx = max(2, int(round(w / step_w)))
            y_coords = np.linspace(2, h - 3, ny, dtype=int)
            x_coords = np.linspace(2, w - 3, nx, dtype=int)
            yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
            all_pts = (yy * w + xx).flatten()
            if len(all_pts) >= p:
                selected = np.linspace(0, len(all_pts) - 1, p, dtype=int)
                flat_indices = all_pts[selected]
            else:
                # Retain all uniform grid points and fill remaining deterministically
                remaining = p - len(all_pts)
                seen = set(all_pts.tolist())
                unseen = [i for i in range(h * w) if i not in seen]
                fill_idx = np_rng.choice(unseen, size=remaining, replace=False)
                flat_indices = np.concatenate([all_pts, fill_idx])

        elif self.topology_type == "wake_rake":
            # Vertical sensor lines positioned downstream in wake region (x in [60, 110])
            rake_x = [60, 75, 90, 105]
            pts_per_rake = p // len(rake_x)
            flat_list = []
            for rx in rake_x:
                y_span = np.linspace(8, h - 9, pts_per_rake, dtype=int)
                flat_list.extend([y * w + rx for y in y_span])
            while len(flat_list) < p:
                flat_list.append(np_rng.choice(h * w))
            flat_indices = np.array(flat_list[:p], dtype=np.int64)

        elif self.topology_type == "random":
            flat_indices = np_rng.choice(h * w, size=p, replace=False)

        elif self.topology_type in ("group_q_deim", "q_deim"):
            # Greedy maximization of regularized log-determinant for physical dual-component probes:
            # Objective: log det(Phi_S^T Phi_S + lambda * I)
            if self.pod_basis is None:
                raise ValueError("pod_basis must be provided for group_q_deim topology.")
            basis = self.pod_basis
            if isinstance(basis, dict):
                basis = basis.get("basis")
                if basis is None:
                    raise ValueError("group_q_deim requires a basis tensor in the POD payload.")
            if isinstance(basis, torch.Tensor):
                basis = basis.detach().cpu().numpy()

            # Separate u and v channels: basis shape is (2*H*W, K)
            # Layout: first H*W elements are u, second H*W are v
            k_modes = basis.shape[1]
            phi_u = basis[: h * w, :]  # [HW, K]
            phi_v = basis[h * w :, :]  # [HW, K]

            reg_lambda = 1e-4
            inv_m = np.eye(k_modes, dtype=np.float64) / reg_lambda  # Initial (M_0)^(-1)
            selected_indices = []
            available = set(range(h * w))

            # Greedy block sensor selection
            for _ in range(p):
                best_idx = None

                # Batch evaluate determinant gain across candidates
                # For candidate i, C_i = [phi_u[i]; phi_v[i]] in R^(2 x K)
                # Gain = det(I_2 + C_i @ inv_m @ C_i^T)
                cand_list = sorted(available)
                cu = phi_u[cand_list, :]  # [N_cand, K]
                cv = phi_v[cand_list, :]  # [N_cand, K]

                # cu @ inv_m: [N_cand, K]
                cu_inv = cu @ inv_m
                cv_inv = cv @ inv_m

                # 2x2 elements:
                # v00 = sum(cu_inv * cu, axis=1), v11 = sum(cv_inv * cv, axis=1)
                # v01 = sum(cu_inv * cv, axis=1), v10 = sum(cv_inv * cu, axis=1)
                v00 = np.sum(cu_inv * cu, axis=1)
                v11 = np.sum(cv_inv * cv, axis=1)
                v01 = np.sum(cu_inv * cv, axis=1)
                det_gains = (1.0 + v00) * (1.0 + v11) - (v01 ** 2)

                best_arg = int(np.argmax(det_gains))
                best_idx = cand_list[best_arg]
                selected_indices.append(best_idx)
                available.remove(best_idx)

                # Rank-2 Woodbury update of inv_m:
                # inv_m <- inv_m - inv_m @ C_best^T @ (I_2 + C_best @ inv_m @ C_best^T)^(-1) @ C_best @ inv_m
                c_best = np.stack([phi_u[best_idx], phi_v[best_idx]], axis=0)  # [2, K]
                v_best = c_best @ inv_m @ c_best.T  # [2, 2]
                inv_gain_mat = np.linalg.inv(np.eye(2) + v_best)  # [2, 2]
                w_mat = inv_m @ c_best.T  # [K, 2]
                inv_m = inv_m - w_mat @ inv_gain_mat @ w_mat.T

            flat_indices = np.array(selected_indices, dtype=np.int64)

        elif self.topology_type == "channel_norm_qr":
            # Heuristic QR column pivoting on joint spatial Frobenius norm
            if self.pod_basis is None:
                raise ValueError("pod_basis must be provided for channel_norm_qr topology.")
            basis = self.pod_basis
            if isinstance(basis, dict):
                basis = basis.get("basis")
            if isinstance(basis, torch.Tensor):
                basis = basis.detach().cpu().numpy()

            if basis.shape[0] == h * w * 2:
                # Reshape to (2, H*W, K) and take norm across 2 channels
                spatial_norm = np.linalg.norm(basis.reshape(2, h * w, -1), axis=0)  # (HW, K)
            else:
                spatial_norm = basis.reshape(h * w, -1)
            _, _, piv = scipy.linalg.qr(spatial_norm.T, pivoting=True)
            flat_indices = piv[:p].astype(np.int64)

        else:
            raise ValueError(f"Unknown topology_type: {self.topology_type}")

        self.selection_order_1d = torch.from_numpy(
            np.asarray(flat_indices, dtype=np.int64).copy()
        ).long()
        flat_indices = np.sort(flat_indices)
        indices_tensor = torch.from_numpy(flat_indices).long()

        # Build 2D mask
        mask = torch.zeros(h, w, dtype=torch.float32)
        y_idx = indices_tensor // w
        x_idx = indices_tensor % w
        mask[y_idx, x_idx] = 1.0

        # Normalized coordinates [0, 1] for each sensor
        norm_y = y_idx.float() / float(h - 1)
        norm_x = x_idx.float() / float(w - 1)
        coords = torch.stack([norm_y, norm_x], dim=-1)  # (P, 2)

        return indices_tensor, coords, mask

    def extract_sensor_values(self, full_field: torch.Tensor) -> torch.Tensor:
        """
        Extract sensor values from full field tensor.
        full_field: [..., H, W, C]
        Returns: [..., P, C]
        """
        *batch_dims, h, w, c = full_field.shape
        flat = full_field.reshape(*batch_dims, h * w, c)
        idx = self.indices_1d.to(full_field.device)
        return torch.index_select(flat, dim=-2, index=idx)

    def to_sparse_tensor(
        self,
        full_field: torch.Tensor,
        append_mask: bool = True,
        fill_method: str = "zero",
    ) -> torch.Tensor:
        """
        Zero out unobserved points, optionally appending the binary mask as extra channel.
        full_field: [..., H, W, C]
        Returns: [..., H, W, C or C+1]
        """
        sensor_values = self.extract_sensor_values(full_field)
        return self.sensor_values_to_sparse_tensor(
            sensor_values,
            append_mask=append_mask,
            fill_method=fill_method,
        )

    def sensor_values_to_sparse_tensor(
        self,
        sensor_values: torch.Tensor,
        append_mask: bool = True,
        fill_method: str = "zero",
    ) -> torch.Tensor:
        """Construct a grid input directly from [..., P, C] observations."""
        if sensor_values.shape[-2] != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensors, got {sensor_values.shape[-2]}."
            )
        if fill_method not in ("zero", "voronoi"):
            raise ValueError("fill_method must be 'zero' or 'voronoi'.")

        device = sensor_values.device
        channels = sensor_values.shape[-1]
        leading_shape = sensor_values.shape[:-2]
        if fill_method == "voronoi":
            assignment = self._voronoi_assignment.to(device).reshape(-1)
            flat_filled = sensor_values.index_select(-2, assignment)
        else:
            flat_filled = torch.zeros(
                *leading_shape,
                self.h * self.w,
                channels,
                dtype=sensor_values.dtype,
                device=device,
            )
            flat_filled.index_copy_(-2, self.indices_1d.to(device), sensor_values)
        sparse_field = flat_filled.reshape(*leading_shape, self.h, self.w, channels)

        if append_mask:
            mask_bc = self.mask.to(device=device, dtype=sensor_values.dtype).unsqueeze(-1)
            mask_expanded = mask_bc.expand(*leading_shape, self.h, self.w, 1)
            return torch.cat([sparse_field, mask_expanded], dim=-1)
        return sparse_field


def get_sensor_placement(
    topology_type: TopologyType = "wall",
    num_sensors: int = 64,
    grid_shape: Tuple[int, int] = (64, 128),
    seed: int = 42,
    pod_basis: Optional[np.ndarray | torch.Tensor] = None,
    sensor_indices: Optional[np.ndarray | torch.Tensor] = None,
) -> SensorTopology:
    return SensorTopology(
        topology_type=topology_type,
        num_sensors=num_sensors,
        grid_shape=grid_shape,
        seed=seed,
        pod_basis=pod_basis,
        sensor_indices=sensor_indices,
    )


def plot_sensor_topologies(
    output_path: str = "artifacts/sensor_topologies.png",
    num_sensors: int = 64,
    grid_shape: Tuple[int, int] = (64, 128),
):
    """Generates and saves a high-quality visualization of sensor topologies."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    h, w = grid_shape
    total_points = h * w
    ratio = (num_sensors / total_points) * 100.0

    topologies = ["wall", "uniform", "wake_rake", "random"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
    axes = axes.flatten()

    # NACA0025 contour
    xc, yc, chord, t_rel = 24.5, 32.0, 35.0, 0.25
    x_foil = np.linspace(xc - chord / 2.0, xc + chord / 2.0, 200)
    s = np.clip((x_foil - (xc - chord / 2.0)) / chord, 0.0, 1.0)
    yt = 5.0 * t_rel * chord * (
        0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * (s ** 2) + 0.2843 * (s ** 3) - 0.1015 * (s ** 4)
    )
    foil_upper = np.column_stack([x_foil, yc + yt])
    foil_lower = np.column_stack([x_foil[::-1], (yc - yt)[::-1]])
    foil_polygon = np.vstack([foil_upper, foil_lower])

    for i, top in enumerate(topologies):
        ax = axes[i]
        sensor_op = get_sensor_placement(topology_type=top, num_sensors=num_sensors, grid_shape=grid_shape)
        y_idx = sensor_op.indices_1d.numpy() // w
        x_idx = sensor_op.indices_1d.numpy() % w

        # Draw grid background and foil
        ax.set_facecolor("#f8f9fa")
        patch = Polygon(foil_polygon, closed=True, facecolor="#495057", edgecolor="#212529", linewidth=1.5, zorder=2)
        ax.add_patch(patch)

        # Plot sensors
        ax.scatter(x_idx, y_idx, c="#e63946", s=18, edgecolors="#1d3557", linewidths=0.6, zorder=3, label="Sensors")

        ax.set_xlim(-1, w)
        ax.set_ylim(-1, h)
        ax.set_aspect("equal")
        ax.set_title(f"Topology: '{top}' (Ps={num_sensors}, {ratio:.2f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("X (streamwise)", fontsize=9)
        ax.set_ylabel("Y (transverse)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved sensor layout visualization to {output_path}")


def diagnose_sensor_observability(
    pod_basis: torch.Tensor | np.ndarray | dict,
    sensor_indices_1d: torch.Tensor | np.ndarray,
    grid_shape: Tuple[int, int] = (64, 128),
    reg_lambda: float = 1e-4,
) -> Dict[str, float]:
    """
    Computes spectral observability diagnostics for a given sensor topology:
    - rank(Phi_P): numerical rank of sampled modes
    - sigma_min(Phi_P): smallest singular value
    - cond(Gram): condition number kappa(Phi_P^T Phi_P + lambda * I)
    - logdet(Gram): log det(Phi_P^T Phi_P + lambda * I)
    """
    basis = pod_basis.get("basis", pod_basis) if isinstance(pod_basis, dict) else pod_basis
    if isinstance(basis, torch.Tensor):
        basis = basis.detach().cpu().numpy()
    if isinstance(sensor_indices_1d, torch.Tensor):
        sensor_indices_1d = sensor_indices_1d.detach().cpu().numpy()

    h, w = grid_shape
    p = len(sensor_indices_1d)
    all_sensor_idx = np.concatenate([sensor_indices_1d, sensor_indices_1d + h * w])
    phi_p = basis[all_sensor_idx, :]  # [2*P, K]

    k = phi_p.shape[1]
    # SVD of sampled basis
    s = np.linalg.svd(phi_p, compute_uv=False)
    rank_p = int(np.sum(s > 1e-6))
    sigma_min = float(s[-1]) if len(s) > 0 else 0.0

    gram = phi_p.T @ phi_p + reg_lambda * np.eye(k)
    s_gram = np.linalg.svd(gram, compute_uv=False)
    cond_gram = float(s_gram[0] / max(1e-12, s_gram[-1]))
    sign, logdet = np.linalg.slogdet(gram)

    return {
        "num_sensors": p,
        "rank_phi_p": rank_p,
        "max_possible_rank": min(2 * p, k),
        "sigma_min": sigma_min,
        "cond_gram": cond_gram,
        "logdet_gram": float(logdet) if sign > 0 else -float("inf"),
    }


def compute_q_deim_dof_diagnostics(
    pod_basis: torch.Tensor | np.ndarray | dict,
    p_scalar: int,
    grid_shape: Tuple[int, int] = (64, 128),
    reg_lambda: float = 1e-4,
) -> Dict[str, Any]:
    """
    Offline numerical diagnostic computing standard scalar-DOF Q-DEIM on the POD basis (2P scalar observations).
    """
    import scipy.linalg
    basis = pod_basis.get("basis", pod_basis) if isinstance(pod_basis, dict) else pod_basis
    if isinstance(basis, torch.Tensor):
        basis = basis.detach().cpu().numpy()

    # basis: [2*HW, K]
    _, _, piv = scipy.linalg.qr(basis.T, pivoting=True)
    dof_indices = piv[:p_scalar].astype(np.int64)

    phi_dof = basis[dof_indices, :]  # [P_scalar, K]
    k = phi_dof.shape[1]
    s = np.linalg.svd(phi_dof, compute_uv=False)
    rank_dof = int(np.sum(s > 1e-6))
    sigma_min = float(s[-1]) if len(s) > 0 else 0.0

    gram = phi_dof.T @ phi_dof + reg_lambda * np.eye(k)
    s_gram = np.linalg.svd(gram, compute_uv=False)
    cond_gram = float(s_gram[0] / max(1e-12, s_gram[-1]))
    sign, logdet = np.linalg.slogdet(gram)

    return {
        "p_scalar": p_scalar,
        "dof_indices": dof_indices.tolist(),
        "rank": rank_dof,
        "sigma_min": sigma_min,
        "cond_gram": cond_gram,
        "logdet_gram": float(logdet) if sign > 0 else -float("inf"),
    }
