from __future__ import annotations

import os
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock
import h5py
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sparse_pod_sim2real.data.manifest import (
    assert_split_disjointness,
    build_nested_few_shot_subsets,
    compute_temporal_blocks,
    stratified_physical_split,
    validate_manifest_set,
)
from sparse_pod_sim2real.data.dataset import (
    SparseTrajectoryDataset,
    compute_mmap_cache_handles,
    required_batch_fields,
    TrajectoryLocalityBatchSampler,
)
from sparse_pod_sim2real.data.sensor_topology import (
    get_sensor_placement,
    diagnose_sensor_observability,
    compute_q_deim_dof_diagnostics,
)
from sparse_pod_sim2real.model.our_model.subspace_warping import GrassmannSubspaceAlignment
from sparse_pod_sim2real.model.our_model.gappy_solver import DifferentiableGappySolver
from sparse_pod_sim2real.model.our_model.misf_no import MISFNO
from sparse_pod_sim2real.model.baselines.gappy_linear_ar import GappyLinearAR
from sparse_pod_sim2real.training.metrics import (
    erode_fluid_mask,
    compute_vorticity_2d,
    StreamingMetricAccumulator,
    compute_metrics,
)
from sparse_pod_sim2real.training.losses import (
    CompositePhysicalLoss,
    compute_vorticity_2d as compute_training_vorticity,
)
from sparse_pod_sim2real.training.trainer import Sim2RealTrainer
from sparse_pod_sim2real.training.evaluate import rollout_target_bounds, select_checkpoint_file
from sparse_pod_sim2real.data.preprocess import process_single_h5
from scripts.compute_sim_pod_basis import allocate_snapshot_quotas


class TestProtocolAndCorrectness(unittest.TestCase):
    def test_01_qr_sign_canonicalization_and_modal_coordinates(self):
        """Verify QR signs are canonicalized so Q == Phi_0 at A=0 and modal coordinates match."""
        m, k, r = 1000, 32, 16
        phi_0, _ = torch.linalg.qr(torch.randn(m, k))
        phi_perp, _ = torch.linalg.qr(torch.randn(m, r))

        warping = GrassmannSubspaceAlignment(k=k, r=r, phi_sim=phi_0, phi_perp=phi_perp)
        q = warping.get_adapted_basis()  # A=0

        # Subspace diff
        max_diff = (q - phi_0).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"QR sign canonicalization failed: max diff {max_diff}")

        # Modal coordinate consistency
        u_dummy = torch.randn(1, 1, m)
        coord_orig = torch.einsum("btm,mk->btk", u_dummy, phi_0)
        coord_adapted = torch.einsum("btm,mk->btk", u_dummy, q)
        coord_diff = (coord_orig - coord_adapted).abs().max().item()
        self.assertLess(coord_diff, 1e-4, f"Modal coordinates flipped at A=0: {coord_diff}")

    def test_02_batched_misf_no(self):
        """Verify MISF-NO forward and backward with batched coordinates [B=4, Ps, 2]."""
        h, w = 64, 128
        ps, k, b, tin, tout = 32, 16, 4, 20, 20
        m = h * w * 2
        basis, _ = torch.linalg.qr(torch.randn(m, k))
        sensor_indices = torch.arange(ps)

        misf = MISFNO(pod_basis=basis, sensor_indices=sensor_indices, h=h, w=w, in_time=tin, out_time=tout, k=k)

        sensor_vals = torch.randn(b, tin, ps, 2)
        sensor_coords = torch.rand(b, ps, 2)  # Batched 3D
        batch = {
            "sensor_values": sensor_vals,
            "sensor_coords": sensor_coords,
        }

        pred = misf(batch)
        self.assertEqual(pred.shape, (b, tout, h, w, 2))
        loss = pred.sum()
        loss.backward()
        self.assertIsNotNone(misf.phca.sensor_proj[0].weight.grad)

    def test_03_fail_hard_manifest_disjointness_and_duplicates(self):
        """Verify strict pairwise disjointness assertions and duplicate detection."""
        # Pairwise disjoint
        train = {"traj_01", "traj_02"}
        val = {"traj_03", "traj_04"}
        test_id = {"traj_05"}
        test_ood = {"traj_06"}
        # Should pass
        assert_split_disjointness(train, val, test_id, test_ood)

        # Overlap should fail
        with self.assertRaises(AssertionError):
            assert_split_disjointness(train, {"traj_02", "traj_03"}, test_id, test_ood)

        # Nested few-shot
        subsets = build_nested_few_shot_subsets(["t1", "t2", "t3", "t4", "t5"], seed=42, k_values=(1, 3, 5))
        self.assertEqual(len(subsets[1]), 1)
        self.assertEqual(len(subsets[3]), 3)
        self.assertEqual(len(subsets[5]), 5)
        self.assertTrue(set(subsets[1]).issubset(set(subsets[3])))
        self.assertTrue(set(subsets[3]).issubset(set(subsets[5])))

    def test_04_temporal_boundary_gap(self):
        """Verify that temporal blocks enforce safety gap >= Tin + Tout - 1."""
        tin, tout = 20, 20
        total_frames = 1000
        (t0, t1), (v0, v1) = compute_temporal_blocks(total_frames, train_ratio=0.8, in_step=tin, out_step=tout)

        self.assertGreaterEqual(v0 - t1, tin + tout - 1)
        self.assertEqual(t0, 0)
        self.assertEqual(v1, total_frames)

    def test_05_selective_batch_collator(self):
        """Verify required_batch_fields drops x_full during sparse Real fine-tuning."""
        fields_sparse_real = required_batch_fields("pod_res_unet", stage="real_finetune", supervision_mode="sparse_sensors")
        self.assertNotIn("x_full", fields_sparse_real)
        self.assertNotIn("x_sparse", fields_sparse_real)
        self.assertIn("sensor_values", fields_sparse_real)

        fields_sim_pretrain = required_batch_fields("pod_res_unet", stage="sim_pretrain", supervision_mode="full_field")
        self.assertIn("x_full", fields_sim_pretrain)
        self.assertIn("y_full", fields_sim_pretrain)

        fields_masked = required_batch_fields("masked_unet3d", stage="real_finetune", supervision_mode="sparse_sensors")
        self.assertIn("x_sparse", fields_masked)

        fields_sensor_val = required_batch_fields(
            "pod_res_unet3d", stage="sensor_eval",
            supervision_mode="sparse_sensors", selection_strategy="sensor_val",
        )
        self.assertIn("y_sensor_values", fields_sensor_val)
        self.assertNotIn("y_full", fields_sensor_val)

    def test_06_eroded_fluid_boundary_vorticity(self):
        """Verify that fluid mask erosion correctly removes boundary stencil points."""
        h, w = 64, 128
        fluid_mask = torch.ones(h, w)
        # Create solid foil region in center: y in [25, 39], x in [20, 50]
        fluid_mask[25:40, 20:51] = 0.0

        eroded = erode_fluid_mask(fluid_mask, stencil_radius=1)
        # Point immediately adjacent to solid should be 0 in eroded mask
        self.assertEqual(eroded[24, 25].item(), 0.0)
        self.assertEqual(eroded[40, 25].item(), 0.0)
        # Distant point should remain 1
        self.assertEqual(eroded[5, 5].item(), 1.0)

        # Vorticity computation
        u_field = torch.randn(1, 1, h, w, 2)
        vort = compute_vorticity_2d(u_field, eroded_fluid_mask=eroded)
        # Boundary points must be 0
        self.assertEqual(vort[0, 0, 24, 25].item(), 0.0)

    def test_07_fp32_amp_gappy_solve(self):
        """Verify Gappy solver executes strictly in FP32 without crash or NaN under AMP."""
        h, w, ps, k = 64, 128, 32, 16
        m = h * w * 2
        basis, _ = torch.linalg.qr(torch.randn(m, k))
        sensor_idx = torch.arange(ps)

        solver = DifferentiableGappySolver(pod_basis=basis, sensor_indices=sensor_idx, h=h, w=w)
        dummy_sensor_vals = torch.randn(2, 5, ps, 2)

        # Run under autocast
        with torch.amp.autocast(device_type="cpu", enabled=True):
            a = solver.solve_coefficients(dummy_sensor_vals)
            field = solver.decode_field(a)

        self.assertFalse(torch.isnan(a).any())
        self.assertFalse(torch.isnan(field).any())
        self.assertEqual(field.shape, (2, 5, h, w, 2))

    def test_08_group_q_deim_and_dof_budget(self):
        """Verify group_q_deim optimizes logdet and matches observation budget."""
        h, w, k, p = 32, 64, 16, 20
        m = h * w * 2
        basis, _ = torch.linalg.qr(torch.randn(m, k))

        # Group Q-DEIM: P physical probes -> 2P scalar observations
        op_group = get_sensor_placement("group_q_deim", num_sensors=p, grid_shape=(h, w), pod_basis=basis)
        diag_group = diagnose_sensor_observability(basis, op_group.indices_1d, grid_shape=(h, w))
        self.assertEqual(diag_group["num_sensors"], p)
        self.assertGreater(diag_group["rank_phi_p"], 0)

        # Scalar DOF Q-DEIM: 2P scalar observations
        diag_dof = compute_q_deim_dof_diagnostics(basis, p_scalar=2 * p, grid_shape=(h, w))
        self.assertEqual(diag_dof["p_scalar"], 2 * p)
        self.assertGreater(diag_dof["rank"], 0)

    def test_09_streaming_accumulator_consistency(self):
        """Verify streaming metric accumulator matches batch metrics to within numerical tolerance."""
        b, tout, h, w = 3, 10, 32, 64
        pred = torch.randn(b, tout, h, w, 2)
        target = torch.randn(b, tout, h, w, 2)
        sensor_idx = torch.arange(16)

        accum = StreamingMetricAccumulator(grid_shape=(h, w), sensor_indices=sensor_idx)
        accum.update(pred, target, traj_stems=[f"traj_{i}" for i in range(b)])
        stream_res = accum.compute()

        batch_res = compute_metrics(pred, target)
        self.assertAlmostEqual(stream_res["rel_l2"], batch_res["rel_l2"], delta=0.05)

    def test_10_sensor_only_accumulator_never_needs_dense_target(self):
        b, tout, h, w, ps = 2, 4, 16, 32, 8
        pred = torch.randn(b, tout, h, w, 2)
        sensor_idx = torch.arange(ps)
        target_sensor = pred.reshape(b, tout, h * w, 2)[:, :, sensor_idx].clone()
        accum = StreamingMetricAccumulator(grid_shape=(h, w), sensor_indices=sensor_idx)
        accum.update(pred, None, ["a", "b"], sensor_values_target=target_sensor)
        result = accum.compute()
        self.assertAlmostEqual(result["sensor_rel_l2"], 0.0, places=7)
        self.assertNotIn("fluid_fullfield_rel_l2", result)

    def test_11_sparse_real_loss_does_not_require_y_full(self):
        h, w, ps = 8, 16, 4
        pred = torch.randn(2, 3, h, w, 2, requires_grad=True)
        sensor_idx = torch.arange(ps)
        target_sensor = torch.randn(2, 3, ps, 2)
        loss_fn = CompositePhysicalLoss(supervision_mode="sparse_sensors", ortho_weight=0.0)
        loss = loss_fn(
            pred,
            {"y_sensor_values": target_sensor},
            model=None,
            is_real_finetuning=True,
            sensor_indices=sensor_idx,
        )
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_12_final_source_dataset_uses_all_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensor_dir = root / "tensor_cache_64x128" / "numerical"
            manifest_dir = root / "manifests"
            tensor_dir.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            path = tensor_dir / "traj.h5.pt"
            torch.save(torch.randn(60, 2, 64, 128), path)
            manifest = {
                "trajectories": {
                    "traj.h5": {
                        "num_frames": 60,
                        "train_range": [0, 40],
                        "val_range": [40, 60],
                    }
                }
            }
            (manifest_dir / "sim_source_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            fields = {"traj_stem", "t_start", "x_full", "y_full"}
            dev = SparseTrajectoryDataset(
                root, dataset_type="numerical", mode="train", in_step=20,
                out_step=20, interval=5, manifest_dir=manifest_dir,
                required_fields=fields, use_all_frames=False,
            )
            final = SparseTrajectoryDataset(
                root, dataset_type="numerical", mode="train", in_step=20,
                out_step=20, interval=5, manifest_dir=manifest_dir,
                required_fields=fields, use_all_frames=True,
            )
            self.assertEqual(len(dev), 1)
            self.assertEqual(len(final), 5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP regression test requires a GPU")
    def test_13_misf_fft_non_power_of_two_under_cuda_amp(self):
        device = torch.device("cuda")
        h, w, ps, k, tin, tout = 8, 16, 8, 8, 20, 20
        basis, _ = torch.linalg.qr(torch.randn(2 * h * w, k, device=device))
        model = MISFNO(
            pod_basis=basis,
            sensor_indices=torch.arange(ps, device=device),
            h=h,
            w=w,
            in_time=tin,
            out_time=tout,
            k=k,
            latent_dim=32,
        ).to(device)
        batch = {
            "sensor_values": torch.randn(1, tin, ps, 2, device=device),
            "sensor_coords": torch.rand(1, ps, 2, device=device),
        }
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(batch)
            loss = pred.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(pred).all())

    def test_14_pod_basis_stage_guard(self):
        trainer = object.__new__(Sim2RealTrainer)
        trainer.config = {}
        trainer.pod_basis = {"manifest": {"stage": "dev"}}
        trainer.pod_basis_path = Path("pod_basis_dev.pt")
        trainer.logger = Mock()

        trainer._validate_pod_basis_stage("dev")
        with self.assertRaisesRegex(RuntimeError, "stage mismatch"):
            trainer._validate_pod_basis_stage("final")

    def test_15_physical_stratification_and_adaptive_mmap_lru(self):
        ids = [
            f"{reynolds}_{aoa}_case{rep}.h5"
            for reynolds in (1000, 5000, 10000)
            for aoa in (-5.0, 0.0, 10.0)
            for rep in range(2)
        ]
        train_a, val_a, physical = stratified_physical_split(ids, 0.7, seed=42)
        train_b, val_b, _ = stratified_physical_split(ids, 0.7, seed=42)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a), round(len(ids) * 0.7))
        self.assertTrue(set(train_a).isdisjoint(val_a))
        self.assertEqual(set(train_a) | set(val_a), set(ids))
        self.assertEqual(set(physical), set(ids))

        gib = 1024**3
        self.assertEqual(compute_mmap_cache_handles(gib, 2, 12 * gib), 4)
        self.assertEqual(compute_mmap_cache_handles(gib, 8, 4 * gib), 2)

    def test_16_fast_manifest_verification_checks_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.pt"
            torch.save(torch.zeros(2), path)
            stat = path.stat()
            metadata = {
                path.stem: {
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            }
            validate_manifest_set({path.stem}, [path], recorded_metadata=metadata)
            os.utime(path, (stat.st_atime, stat.st_mtime + 2.0))
            with self.assertRaisesRegex(RuntimeError, "modification-time mismatch"):
                validate_manifest_set({path.stem}, [path], recorded_metadata=metadata)

    def test_17_trajectory_locality_sampler_is_complete_and_deterministic(self):
        samples = [(f"traj_{traj}", time_index) for traj in range(3) for time_index in range(10)]
        sampler_a = TrajectoryLocalityBatchSampler(samples, batch_size=3, seed=11, locality_batches=2)
        sampler_b = TrajectoryLocalityBatchSampler(samples, batch_size=3, seed=11, locality_batches=2)
        batches_a = list(iter(sampler_a))
        batches_b = list(iter(sampler_b))
        self.assertEqual(batches_a, batches_b)
        self.assertEqual(sorted(index for batch in batches_a for index in batch), list(range(30)))
        for batch in batches_a:
            self.assertEqual(len({samples[index][0] for index in batch}), 1)
            times = [samples[index][1] for index in batch]
            self.assertEqual(times, list(range(times[0], times[0] + len(times))))

    def test_18_rollout_targets_advance_without_reuse(self):
        self.assertEqual(rollout_target_bounds(100, 20, 20, 1), (120, 140))
        self.assertEqual(rollout_target_bounds(100, 20, 20, 2), (140, 160))
        self.assertEqual(rollout_target_bounds(100, 20, 20, 3), (160, 180))

    def test_19_sparse_loss_supports_models_without_embedded_sensor_indices(self):
        pred = torch.randn(1, 3, 8, 16, 2, requires_grad=True)
        sensor_indices = torch.tensor([0, 5, 17, 63])
        target = pred.detach().reshape(1, 3, 8 * 16, 2)[:, :, sensor_indices].clone()
        loss_fn = CompositePhysicalLoss(supervision_mode="sparse_sensors", ortho_weight=0.0)
        loss = loss_fn(
            pred,
            {"y_sensor_values": target},
            model=torch.nn.Identity(),
            is_real_finetuning=True,
            sensor_indices=sensor_indices,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_20_pod_snapshot_budget_covers_every_trajectory(self):
        quotas = allocate_snapshot_quotas([100] * 99, budget=5000)
        self.assertEqual(len(quotas), 99)
        self.assertEqual(sum(quotas), 5000)
        self.assertTrue(all(quota > 0 for quota in quotas))
        self.assertLessEqual(max(quotas) - min(quotas), 1)
        with self.assertRaisesRegex(ValueError, "smaller than"):
            allocate_snapshot_quotas([10] * 99, budget=98)

    def test_21_checkpoint_selection_never_confuses_sim_and_real_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            (checkpoint_dir / "sim_pretrained_final.pt").touch()
            (checkpoint_dir / "fixed_budget_step_500.pt").touch()
            config = {
                "model_name": "pod_res_unet3d",
                "sim_epochs": 100,
                "real_epochs": 50,
                "sim_mode": "final",
                "val_selection_strategy": "fixed_budget",
                "fixed_steps": 500,
            }
            selected = select_checkpoint_file(checkpoint_dir, config)
            self.assertEqual(selected, checkpoint_dir / "fixed_budget_step_500.pt")

            (checkpoint_dir / "fixed_budget_step_500.pt").unlink()
            self.assertIsNone(select_checkpoint_file(checkpoint_dir, config))

    def test_22_h5_cache_preserves_official_id_and_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "1000_5.0.h5"
            output = root / "cache"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("u", data=np.zeros((6, 8, 16), dtype=np.float32))
                handle.create_dataset("v", data=np.ones((6, 8, 16), dtype=np.float32))

            result = process_single_h5(source, output, target_res=(4, 8))
            self.assertEqual(result["sim_id"], "1000_5.0.h5")
            cache_file = output / "1000_5.0.h5.pt"
            self.assertTrue(cache_file.exists())
            self.assertEqual(
                tuple(torch.load(cache_file, mmap=True, weights_only=True).shape),
                (6, 2, 4, 8),
            )
            with self.assertRaisesRegex(ValueError, "Cached tensor shape mismatch"):
                process_single_h5(source, output, target_res=(8, 16))

    def test_23_training_vorticity_uses_physical_nonperiodic_differences(self):
        field = torch.zeros(1, 1, 5, 7, 2)
        x = torch.arange(7, dtype=torch.float32) * 0.25
        field[..., 1] = x.view(1, 1, 1, 7)
        vorticity = compute_training_vorticity(field, dx=0.25, dy=0.5)
        self.assertTrue(torch.allclose(vorticity, torch.ones_like(vorticity), atol=1e-6))

    def test_24_direct_sparse_grid_construction_matches_full_field_path(self):
        topology = get_sensor_placement(
            topology_type="random", num_sensors=7, grid_shape=(8, 16), seed=9
        )
        full = torch.randn(3, 8, 16, 2)
        sensors = topology.extract_sensor_values(full)
        for fill_method in ("zero", "voronoi"):
            expected = topology.to_sparse_tensor(
                full, append_mask=True, fill_method=fill_method
            )
            actual = topology.sensor_values_to_sparse_tensor(
                sensors, append_mask=True, fill_method=fill_method
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_25_group_qdeim_selection_order_supports_nested_sensor_budgets(self):
        h, w, rank = 8, 16, 8
        basis, _ = torch.linalg.qr(torch.randn(2 * h * w, rank))
        maximum = get_sensor_placement(
            topology_type="group_q_deim",
            num_sensors=16,
            grid_shape=(h, w),
            pod_basis=basis,
        )
        sets = []
        for count in (4, 8, 16):
            topology = get_sensor_placement(
                topology_type="group_q_deim",
                num_sensors=count,
                grid_shape=(h, w),
                pod_basis=basis,
                sensor_indices=maximum.selection_order_1d[:count],
            )
            sets.append(set(topology.indices_1d.tolist()))
        self.assertTrue(sets[0] < sets[1] < sets[2])

    def test_26_gappy_linear_ar_supports_dense_and_sparse_protocol_paths(self):
        h, w, rank = 4, 8, 5
        basis, _ = torch.linalg.qr(torch.randn(2 * h * w, rank))
        sensors = torch.tensor([0, 3, 11, 20, 31])
        model = GappyLinearAR(
            pod_basis={"basis": basis, "mean": torch.zeros(2 * h * w)},
            sensor_indices=sensors,
            h=h,
            w=w,
            out_time=3,
            rank=rank,
        )
        full = torch.randn(2, 4, h, w, 2)
        sparse_values = full.reshape(2, 4, h * w, 2)[:, :, sensors]
        dense_pred = model({"x_full": full}, mode="full")
        sparse_pred = model({"sensor_values": sparse_values}, mode="sparse")
        self.assertEqual(dense_pred.shape, (2, 3, h, w, 2))
        self.assertEqual(sparse_pred.shape, dense_pred.shape)
        (dense_pred.square().mean() + sparse_pred.square().mean()).backward()
        self.assertIsNotNone(model.modal_prop.transition.weight.grad)

    def test_27_sim_validation_uses_dense_source_contract(self):
        fields = required_batch_fields(
            "gappy_linear_ar",
            stage="sim_pretrain",
            supervision_mode="sparse_sensors",
            selection_strategy="dense_val",
        )
        self.assertIn("x_full", fields)
        self.assertIn("y_full", fields)
        self.assertNotIn("sensor_values", fields)


if __name__ == "__main__":
    unittest.main()
