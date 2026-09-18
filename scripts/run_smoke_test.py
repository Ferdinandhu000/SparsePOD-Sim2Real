#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparse_pod_sim2real.data.sensor_topology import get_sensor_placement, diagnose_sensor_observability
from sparse_pod_sim2real.model.baselines.masked_unet import MaskedUNet3D
from sparse_pod_sim2real.model.baselines.masked_fno import MaskedFNO3D
from sparse_pod_sim2real.model.baselines.classical_gappy import ClassicalGappyPOD
from sparse_pod_sim2real.model.baselines.gappy_linear_ar import GappyLinearAR
from sparse_pod_sim2real.model.our_model.pod_res_unet import PODResUNet3DSparse
from sparse_pod_sim2real.model.our_model.misf_no import MISFNO
from sparse_pod_sim2real.model.our_model.subspace_warping import GrassmannSubspaceAlignment
from sparse_pod_sim2real.training.losses import CompositePhysicalLoss
from sparse_pod_sim2real.training.metrics import StreamingMetricAccumulator


def test_all():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE SMOKE & CORRECTNESS TEST FOR SPARSE-POD-SIM2REAL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Keep the smoke bounded so it completes on an 8 GiB developer GPU. The
    # production-shape coordinate path is covered separately by unit tests.
    h, w = 32, 64
    tin, tout = 4, 4
    b = 1
    ps = 16
    k = 8

    # 1. Test Sensor Topologies including group_q_deim
    print("\n[1/8] Testing Sensor Topologies...")
    m = h * w * 2
    dummy_pod_raw = torch.randn(m, k)
    dummy_pod, _ = torch.linalg.qr(dummy_pod_raw)

    for top in ["wall", "uniform", "wake_rake", "random", "group_q_deim"]:
        op = get_sensor_placement(topology_type=top, num_sensors=ps, grid_shape=(h, w), pod_basis=dummy_pod)
        assert op.mask.shape == (h, w), f"Mask shape error: {op.mask.shape}"
        assert op.indices_1d.shape[0] == ps, f"Sensor count error: {op.indices_1d.shape}"
        diag = diagnose_sensor_observability(dummy_pod, op.indices_1d, grid_shape=(h, w))
        print(f"  [OK] Topology '{top:12s}' passed (sensors: {ps}, rank={diag['rank_phi_p']}/{diag['max_possible_rank']}, cond={diag['cond_gram']:.1f})")

    # 2. Test Grassmann QR Sign Canonicalization
    print("\n[2/8] Testing Grassmann QR Sign Canonicalization...")
    perp = torch.randn(m, 4)
    perp, _ = torch.linalg.qr(perp)
    warping = GrassmannSubspaceAlignment(k=k, r=4, phi_sim=dummy_pod, phi_perp=perp).to(device)
    adapted_at_zero = warping.get_adapted_basis()  # at A=0
    diff_zero = (adapted_at_zero - dummy_pod.to(device)).abs().max().item()
    print(f"  [OK] At A=0, max |Q - Phi_sim| = {diff_zero:.6e}")
    assert diff_zero < 1e-4, f"QR sign canonicalization failed! Discrepancy: {diff_zero}"

    # Build a bounded batched test input.
    sensor_op = get_sensor_placement("wall", num_sensors=ps, grid_shape=(h, w))
    sensor_indices = sensor_op.indices_1d.to(device)

    dummy_full_in = torch.randn(b, tin, h, w, 2, device=device)
    dummy_full_out = torch.randn(b, tout, h, w, 2, device=device)
    dummy_sparse_in = sensor_op.to_sparse_tensor(dummy_full_in, append_mask=True).to(device)  # [B, Tin, H, W, 3]
    dummy_sensors = sensor_op.extract_sensor_values(dummy_full_in).to(device)  # [B, Tin, Ps, 2]
    dummy_y_sensors = sensor_op.extract_sensor_values(dummy_full_out).to(device)  # [B, Tout, Ps, 2]
    # Batched 3D coordinates [B, Ps, 2] as returned by DataLoader
    dummy_coords = sensor_op.coords.unsqueeze(0).expand(b, -1, -1).to(device)

    pod_basis_dict = {
        "basis": dummy_pod.to(device),
        "basis_perp": perp.to(device),
        "mean": torch.randn(m, device=device) * 0.05,
        "singular_values": torch.linspace(10, 1, k + 4, device=device),
    }

    batch = {
        "x_sparse": dummy_sparse_in,
        "x_full": dummy_full_in,
        "y_full": dummy_full_out,
        "y_sensor_values": dummy_y_sensors,
        "sensor_values": dummy_sensors,
        "sensor_coords": dummy_coords,  # Batched [B, Ps, 2]
        "sensor_mask": sensor_op.mask.to(device),
        "traj_stem": ["sample_0"],
        "t_start": [0],
    }

    # 3. Test Masked-UNet3D Baseline
    print("\n[3/8] Testing Masked-UNet3D Baseline...")
    unet = MaskedUNet3D(in_time=tin, out_time=tout, channels=3, out_channels=2, dim=8, dim_mults=(1, 2)).to(device)
    pred_unet = unet(batch)
    assert pred_unet.shape == (b, tout, h, w, 2), f"UNet output shape mismatch: {pred_unet.shape}"
    loss_u = nn.MSELoss()(pred_unet, dummy_full_out)
    loss_u.backward()
    print(f"  [OK] Masked-UNet3D forward/backward passed. Output shape: {pred_unet.shape}")
    del unet, pred_unet, loss_u
    torch.cuda.empty_cache()

    # 4. Test Masked-FNO3D Baseline
    print("\n[4/8] Testing Masked-FNO3D Baseline...")
    fno = MaskedFNO3D(modes1=2, modes2=4, modes3=4, width=8, n_layers=2, shape_in=(tin, h, w, 3), shape_out=(tout, h, w, 2)).to(device)
    pred_fno = fno(batch)
    assert pred_fno.shape == (b, tout, h, w, 2), f"FNO output shape mismatch: {pred_fno.shape}"
    loss_f = nn.MSELoss()(pred_fno, dummy_full_out)
    loss_f.backward()
    print(f"  [OK] Masked-FNO3D forward/backward passed. Output shape: {pred_fno.shape}")
    del fno, pred_fno, loss_f
    torch.cuda.empty_cache()

    # 5. Test Classical Gappy-POD
    print("\n[5/8] Testing Classical Gappy-POD Baseline...")
    gappy_base = ClassicalGappyPOD(pod_basis=dummy_pod.to(device), sensor_indices=sensor_indices, h=h, w=w, in_time=tin, out_time=tout).to(device)
    pred_gappy = gappy_base(batch)
    assert pred_gappy.shape == (b, tout, h, w, 2), f"Gappy output shape mismatch: {pred_gappy.shape}"
    print(f"  [OK] Classical Gappy-POD reconstruction passed. Output shape: {pred_gappy.shape}")
    del gappy_base, pred_gappy
    torch.cuda.empty_cache()

    # 6. Test trainable Gappy-POD linear dynamics baseline
    print("\n[6/8] Testing Gappy-LinearAR Baseline...")
    gappy_ar = GappyLinearAR(
        pod_basis=pod_basis_dict,
        sensor_indices=sensor_indices,
        h=h,
        w=w,
        out_time=tout,
        rank=k,
    ).to(device)
    pred_gappy_ar_dense = gappy_ar(batch, mode="full")
    pred_gappy_ar_sparse = gappy_ar(batch, mode="sparse")
    assert pred_gappy_ar_dense.shape == (b, tout, h, w, 2)
    assert pred_gappy_ar_sparse.shape == pred_gappy_ar_dense.shape
    (pred_gappy_ar_dense.square().mean() + pred_gappy_ar_sparse.square().mean()).backward()
    print("  [OK] Gappy-LinearAR dense/sparse forward/backward passed.")
    del gappy_ar, pred_gappy_ar_dense, pred_gappy_ar_sparse
    torch.cuda.empty_cache()

    # 7. Test Flagship PODResUNet3DSparse
    print("\n[7/8] Testing Flagship POD-ResUNet3DSparse Model...")
    pod_unet = PODResUNet3DSparse(
        pod_basis=pod_basis_dict,
        sensor_indices=sensor_indices,
        h=h,
        w=w,
        in_time=tin,
        out_time=tout,
        k=k,
        unet_dim=8,
        dim_mults=(1, 2),
        use_warping=True,
    ).to(device)

    # Sim pre-training full mode
    pred_full = pod_unet(batch, mode="full")
    assert pred_full.shape == (b, tout, h, w, 2)
    loss_sim = nn.MSELoss()(pred_full, dummy_full_out)
    loss_sim.backward()

    # Sparse Real mode
    res_dict = pod_unet(batch, mode="sparse", return_components=True)
    pred_res = res_dict["u_final"]
    assert pred_res.shape == (b, tout, h, w, 2)

    # Sparse supervision
    sparse_loss_fn = CompositePhysicalLoss(supervision_mode="sparse_sensors")
    loss_res = sparse_loss_fn(
        pred_res, batch, pod_unet, is_real_finetuning=True
    )
    loss_res.backward()
    print(f"  [OK] POD-ResUNet3DSparse full & sparse forward/backward passed. Loss: {loss_res.item():.4f}")
    del pod_unet, sparse_loss_fn
    torch.cuda.empty_cache()

    # 8. Test MISFNO Model with batched coordinates [B, Ps, 2]
    print("\n[8/8] Testing MISFNO Model with Batched Sensor Coordinates...")
    misf = MISFNO(pod_basis=dummy_pod.to(device), sensor_indices=sensor_indices, h=h, w=w, in_time=tin, out_time=tout, k=k, latent_dim=32).to(device)
    pred_misf = misf(batch)
    assert pred_misf.shape == (b, tout, h, w, 2), f"MISFNO output shape mismatch: {pred_misf.shape}"
    loss_misf = nn.MSELoss()(pred_misf, dummy_full_out)
    loss_misf.backward()
    print(f"  [OK] MISFNO forward/backward with batched [B, Ps, 2] coords passed. Output shape: {pred_misf.shape}")
    del misf, pred_misf, loss_misf
    torch.cuda.empty_cache()

    # Test StreamingMetricAccumulator
    print("\nVerifying Streaming Metric Accumulator...")
    accum = StreamingMetricAccumulator(grid_shape=(h, w), sensor_indices=sensor_indices)
    accum.update(dummy_full_out, dummy_full_out, traj_stems=["traj_0"])
    exact_metrics = accum.compute()
    assert exact_metrics["fluid_fullfield_rel_l2"] < 1e-5, f"Expected 0 relative L2 on identical target, got {exact_metrics['fluid_fullfield_rel_l2']}"
    print(f"  [OK] Streaming metric accumulator verified (Self Rel-L2: {exact_metrics['fluid_fullfield_rel_l2']:.6e})")

    print("\n" + "=" * 80)
    print("ALL BOUNDED SMOKE & CORRECTNESS TESTS PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    test_all()
