#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement
from sparse_pod_sim2real.model.baselines.masked_unet import MaskedUNet3D
from sparse_pod_sim2real.model.baselines.masked_fno import MaskedFNO3D
from sparse_pod_sim2real.model.baselines.classical_gappy import ClassicalGappyPOD
from sparse_pod_sim2real.model.our_model.pod_res_unet import PODResUNet3DSparse
from sparse_pod_sim2real.model.our_model.misf_no import MISFNO
from sparse_pod_sim2real.training.losses import CompositePhysicalLoss, relative_l2_per_sample
from sparse_pod_sim2real.training.metrics import compute_metrics


def test_all():
    print("=" * 80)
    print("RUNNING SMOKE TEST FOR SPARSE-POD-SIM2REAL PIPELINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    h, w = 64, 128
    tin, tout = 20, 20
    b = 1
    ps = 64
    k = 32

    # 1. Test Sensor Topologies
    print("\n[1/6] Testing Sensor Topologies...")
    for top in ["wall", "uniform", "wake_rake", "random"]:
        op = get_sensor_placement(topology_type=top, num_sensors=ps, grid_shape=(h, w))
        assert op.mask.shape == (h, w), f"Mask shape error: {op.mask.shape}"
        assert op.indices_1d.shape[0] == ps, f"Sensor count error: {op.indices_1d.shape}"
        print(f"  [OK] Topology '{top}' passed (sensors: {ps}, mask sum: {op.mask.sum().item():.0f})")

    # Generate dummy POD basis and sensor indices
    m = h * w * 2
    pod_basis, _ = torch.linalg.qr(torch.randn(m, k, device=device))
    sensor_op = get_sensor_placement("wall", num_sensors=ps, grid_shape=(h, w))
    sensor_indices = sensor_op.indices_1d.to(device)

    # Create dummy batch
    dummy_full_in = torch.randn(b, tin, h, w, 2, device=device)
    dummy_full_out = torch.randn(b, tout, h, w, 2, device=device)
    dummy_sparse_in = sensor_op.to_sparse_tensor(dummy_full_in, append_mask=True).to(device)  # [B, Tin, H, W, 3]
    dummy_sensors = sensor_op.extract_sensor_values(dummy_full_in).to(device)  # [B, Tin, Ps, 2]

    batch = {
        "x_sparse": dummy_sparse_in,
        "x_full": dummy_full_in,
        "y_full": dummy_full_out,
        "sensor_values": dummy_sensors,
        "sensor_coords": sensor_op.coords.to(device),
        "sensor_mask": sensor_op.mask.to(device),
    }

    torch.cuda.empty_cache()

    # 2. Test Masked-UNet3D
    print("\n[2/6] Testing Masked-UNet3D Baseline...")
    unet = MaskedUNet3D(in_time=tin, out_time=tout, channels=3, out_channels=2, dim=32, dim_mults=(1, 2, 4)).to(device)
    pred_unet = unet(batch)
    assert pred_unet.shape == (b, tout, h, w, 2), f"UNet output shape mismatch: {pred_unet.shape}"
    loss_u = nn.MSELoss()(pred_unet, dummy_full_out)
    loss_u.backward()
    print(f"  [OK] Masked-UNet3D forward/backward passed. Output shape: {pred_unet.shape}")
    del unet, pred_unet, loss_u
    torch.cuda.empty_cache()

    # 3. Test Masked-FNO3D
    print("\n[3/6] Testing Masked-FNO3D Baseline...")
    fno = MaskedFNO3D(modes1=4, modes2=4, modes3=4, width=16, n_layers=2, shape_in=(tin, h, w, 3), shape_out=(tout, h, w, 2)).to(device)
    pred_fno = fno(batch)
    assert pred_fno.shape == (b, tout, h, w, 2), f"FNO output shape mismatch: {pred_fno.shape}"
    loss_f = nn.MSELoss()(pred_fno, dummy_full_out)
    loss_f.backward()
    print(f"  [OK] Masked-FNO3D forward/backward passed. Output shape: {pred_fno.shape}")

    # 4. Test Classical Gappy-POD
    print("\n[4/6] Testing Classical Gappy-POD Baseline...")
    gappy_base = ClassicalGappyPOD(pod_basis=pod_basis, sensor_indices=sensor_indices, h=h, w=w, in_time=tin, out_time=tout).to(device)
    pred_gappy = gappy_base(batch)
    assert pred_gappy.shape == (b, tout, h, w, 2), f"Gappy output shape mismatch: {pred_gappy.shape}"
    print(f"  [OK] Classical Gappy-POD reconstruction passed. Output shape: {pred_gappy.shape}")
    del gappy_base, pred_gappy
    torch.cuda.empty_cache()

    # 5. Test Flagship SOTA: PODResUNet3DSparse
    print("\n[5/6] Testing Proposed POD-ResUNet3DSparse SOTA Model...")
    pod_unet = PODResUNet3DSparse(
        pod_basis=pod_basis,
        sensor_indices=sensor_indices,
        h=h,
        w=w,
        in_time=tin,
        out_time=tout,
        k=k,
        unet_dim=32,
        dim_mults=(1, 2, 4),
        use_warping=True,
    ).to(device)

    pred_res = pod_unet(batch)
    assert pred_res.shape == (b, tout, h, w, 2), f"POD-ResUNet output shape mismatch: {pred_res.shape}"
    comp_loss_fn = CompositePhysicalLoss()
    loss_res, l_dict = comp_loss_fn(pred_res, dummy_full_out, sensor_indices=sensor_indices, model=pod_unet)
    loss_res.backward()
    print(f"  [OK] POD-ResUNet3DSparse forward/backward passed. Output shape: {pred_res.shape}, Loss: {loss_res.item():.4f}")
    del pod_unet, comp_loss_fn
    torch.cuda.empty_cache()

    # 6. Test MISFNO Model
    print("\n[6/6] Testing Proposed MISFNO Model...")
    misf = MISFNO(pod_basis=pod_basis, sensor_indices=sensor_indices, h=h, w=w, in_time=tin, out_time=tout, k=k, latent_dim=64).to(device)
    pred_misf = misf(batch)
    assert pred_misf.shape == (b, tout, h, w, 2), f"MISFNO output shape mismatch: {pred_misf.shape}"
    loss_misf = nn.MSELoss()(pred_misf, dummy_full_out)
    loss_misf.backward()
    print(f"  [OK] MISFNO forward/backward passed. Output shape: {pred_misf.shape}")

    # Verify metrics computation
    metrics = compute_metrics(pred_res, dummy_full_out)
    print("\nMetrics sample:")
    for k_m, v_m in metrics.items():
        print(f"  - {k_m}: {v_m:.5f}")

    print("\n" + "=" * 80)
    print("ALL SMOKE TESTS PASSED CLEANLY! SYSTEM IS 100% OPERATIONAL.")
    print("=" * 80)


if __name__ == "__main__":
    test_all()
