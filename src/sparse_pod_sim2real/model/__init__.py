from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from .baselines.masked_unet import MaskedUNet3D
from .baselines.masked_fno import MaskedFNO3D
from .baselines.classical_gappy import ClassicalGappyPOD
from .our_model.pod_res_unet import PODResUNet3DSparse
from .our_model.misf_no import MISFNO


def load_model(
    config: Dict[str, Any],
    pod_basis: Optional[torch.Tensor] = None,
    sensor_indices: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Model factory loading configured baseline or proposed model."""
    name = config.get("model_name", "pod_res_unet3d").lower()
    in_time = config.get("in_step", 20)
    out_time = config.get("out_step", 20)
    h = config.get("grid_h", 64)
    w = config.get("grid_w", 128)
    k = config.get("pod_rank", 64)

    if name in ("masked_unet3d", "masked_unet"):
        dim = config.get("unet_dim", 64)
        dim_mults = tuple(config.get("dim_mults", [1, 2, 4]))
        return MaskedUNet3D(
            in_time=in_time,
            out_time=out_time,
            channels=3,
            out_channel=2,
            dim=dim,
            dim_mults=dim_mults,
        )

    elif name in ("masked_fno3d", "masked_fno"):
        modes1 = config.get("modes1", 8)
        modes2 = config.get("modes2", 8)
        modes3 = config.get("modes3", 8)
        width = config.get("fno_width", 32)
        n_layers = config.get("n_layers", 4)
        return MaskedFNO3D(
            modes1=modes1,
            modes2=modes2,
            modes3=modes3,
            width=width,
            n_layers=n_layers,
            shape_in=(in_time, h, w, 3),
            shape_out=(out_time, h, w, 2),
        )

    elif name in ("classical_gappy_pod", "gappy_pod"):
        if pod_basis is None or sensor_indices is None:
            raise ValueError("pod_basis and sensor_indices must be provided for ClassicalGappyPOD.")
        reg_lambda = config.get("reg_lambda", 1e-4)
        return ClassicalGappyPOD(
            pod_basis=pod_basis[:, :k],
            sensor_indices=sensor_indices,
            reg_lambda=reg_lambda,
            h=h,
            w=w,
            in_time=in_time,
            out_time=out_time,
        )

    elif name in ("pod_res_unet3d", "pod_res_unet"):
        if pod_basis is None or sensor_indices is None:
            raise ValueError("pod_basis and sensor_indices must be provided for PODResUNet3DSparse.")
        dim = config.get("unet_dim", 64)
        dim_mults = tuple(config.get("dim_mults", [1, 2, 4]))
        reg_lambda = config.get("reg_lambda", 1e-4)
        use_warping = config.get("use_warping", True)
        residual_weight = config.get("residual_weight", 1.0)
        return PODResUNet3DSparse(
            pod_basis=pod_basis,
            sensor_indices=sensor_indices,
            h=h,
            w=w,
            in_time=in_time,
            out_time=out_time,
            k=k,
            reg_lambda=reg_lambda,
            unet_dim=dim,
            dim_mults=dim_mults,
            use_warping=use_warping,
            residual_weight=residual_weight,
        )

    elif name in ("misf_no", "misfno"):
        if pod_basis is None or sensor_indices is None:
            raise ValueError("pod_basis and sensor_indices must be provided for MISFNO.")
        reg_lambda = config.get("reg_lambda", 1e-4)
        latent_dim = config.get("latent_dim", 128)
        use_warping = config.get("use_warping", True)
        return MISFNO(
            pod_basis=pod_basis,
            sensor_indices=sensor_indices,
            h=h,
            w=w,
            in_time=in_time,
            out_time=out_time,
            k=k,
            reg_lambda=reg_lambda,
            latent_dim=latent_dim,
            use_warping=use_warping,
        )

    else:
        raise ValueError(f"Unknown model_name: {name}")


__all__ = [
    "load_model",
    "MaskedUNet3D",
    "MaskedFNO3D",
    "ClassicalGappyPOD",
    "PODResUNet3DSparse",
    "MISFNO",
]
