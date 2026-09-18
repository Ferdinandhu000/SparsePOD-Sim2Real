from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from ..data.dataset import create_dataloaders
from ..data.sensor_topology import get_sensor_placement
from ..data.manifest import compute_sha256, save_manifest, load_manifest
from ..model import load_model
from .losses import CompositePhysicalLoss
from .metrics import StreamingMetricAccumulator, build_naca0025_fluid_mask, erode_fluid_mask


def set_seed(seed: int) -> None:
    """Sets global random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("Sim2RealTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


class Sim2RealTrainer:
    def __init__(self, config: Dict, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = Path(config.get("data_root", "data/foil"))
        self.seed = int(config.get("seed", 42))
        self.model_init_seed = int(config.get("model_init_seed", self.seed))
        self.few_shot_traj_seed = int(config.get("few_shot_traj_seed", self.seed))
        self.sensor_placement_seed = int(config.get("sensor_placement_seed", self.seed))
        self.manifest_dir = Path(config.get("manifest_dir", "manifests"))
        self.config.update({
            "model_init_seed": self.model_init_seed,
            "few_shot_traj_seed": self.few_shot_traj_seed,
            "sensor_placement_seed": self.sensor_placement_seed,
            "manifest_dir": str(self.manifest_dir),
        })
        set_seed(self.model_init_seed)

        # Setup run directories
        self.exp_name = config.get("exp_name", "sim2real_run")
        self.output_dir = Path(config.get("output_dir", "artifacts/runs")) / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_dir = Path(config.get("best_checkpoints_dir", "best_checkpoints")) / self.exp_name
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(self.output_dir / "train.log")
        self.logger.info(
            f"Initialized Sim2RealTrainer on device: {self.device} "
            f"(model={self.model_init_seed}, few-shot={self.few_shot_traj_seed}, "
            f"sensors={self.sensor_placement_seed})"
        )
        self.logger.info(f"Output directory: {self.output_dir}")

        # Load or locate POD basis
        self.pod_basis_path: Optional[Path] = None
        self.pod_basis = self._load_pod_basis()

        # Build sensor topology directly without instantiating full dataset
        self.num_sensors = config.get("num_sensors", 64)
        self.sensor_topology = config.get("sensor_topology", "wall")
        self.grid_shape = (int(config.get("grid_h", 64)), int(config.get("grid_w", 128)))
        canonical_sensor_manifest = (
            self.manifest_dir / f"sensor_manifest_{self.sensor_topology}_{self.num_sensors}.json"
        )
        self.canonical_sensor_manifest_hash: Optional[str] = None
        self.sensor_placement_basis_hash: Optional[str] = None
        canonical_sensor_indices = None
        if canonical_sensor_manifest.exists():
            canonical = load_manifest(canonical_sensor_manifest)
            self.canonical_sensor_manifest_hash = compute_sha256(canonical_sensor_manifest)
            self.sensor_placement_basis_hash = canonical.get("basis_hash")
            expected_provenance = {
                "topology_type": self.sensor_topology,
                "num_sensors": int(self.num_sensors),
                "grid_shape": list(self.grid_shape),
                "sensor_placement_seed": self.sensor_placement_seed,
                "sim_manifest_hash": compute_sha256(self.manifest_dir / "sim_source_manifest.json"),
                "real_manifest_hash": compute_sha256(self.manifest_dir / "real_split_manifest.json"),
            }
            mismatches = {
                key: (canonical.get(key), expected)
                for key, expected in expected_provenance.items()
                if canonical.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"Canonical sensor manifest is stale or incompatible: {canonical_sensor_manifest}; "
                    f"mismatches={mismatches}. Rebuild the sensor manifests."
                )
            canonical_sensor_indices = torch.tensor(canonical.get("sensor_indices", []), dtype=torch.long)
        elif not bool(config.get("allow_manifest_fallback", config.get("smoke_test", False))):
            raise FileNotFoundError(
                f"Formal run requires canonical sensor manifest: {canonical_sensor_manifest}. "
                "Run scripts/build_provenance_manifests.py before training."
            )
        sensor_op = get_sensor_placement(
            topology_type=self.sensor_topology,
            num_sensors=self.num_sensors,
            grid_shape=self.grid_shape,
            seed=self.sensor_placement_seed,
            pod_basis=self.pod_basis,
            sensor_indices=canonical_sensor_indices,
        )
        self.sensor_indices = sensor_op.indices_1d.to(self.device)
        self.sensor_coords = sensor_op.coords.to(self.device)
        self.fluid_mask = build_naca0025_fluid_mask(
            grid_shape=self.grid_shape
        )
        self.grid_dx = float(config.get("grid_dx", 1.0))
        self.grid_dy = float(config.get("grid_dy", 1.0))

        # Archive sensor placement artifacts
        torch.save(self.sensor_indices.cpu(), self.output_dir / "sensor_indices.pt")
        torch.save(self.sensor_indices.cpu(), self.best_checkpoint_dir / "sensor_indices.pt")
        if self.pod_basis is not None:
            torch.save(self.pod_basis, self.best_checkpoint_dir / "pod_basis.pt")
        self.provenance = self._collect_provenance()
        if canonical_sensor_manifest.exists():
            expected_indices = torch.tensor(canonical.get("sensor_indices", []), dtype=torch.long)
            if not torch.equal(expected_indices, self.sensor_indices.detach().cpu()):
                raise RuntimeError(
                    f"Runtime sensor placement disagrees with canonical manifest: "
                    f"{canonical_sensor_manifest}"
                )
        self._save_sensor_manifest(sensor_op)

        # Build model and loss function
        self.model = load_model(config, pod_basis=self.pod_basis, sensor_indices=self.sensor_indices).to(self.device)
        self.loss_fn = CompositePhysicalLoss(
            v_weight=config.get("v_weight", 2.0),
            vorticity_weight=config.get("vorticity_weight", 0.1),
            sensor_weight=config.get("sensor_weight", 1.0),
            ortho_weight=config.get("ortho_weight", 0.01),
            supervision_mode=config.get("supervision_mode", "sparse_sensors"),
            grid_dx=self.grid_dx,
            grid_dy=self.grid_dy,
            vorticity_valid_mask=erode_fluid_mask(self.fluid_mask),
        )
        self.use_amp = bool(config.get("use_amp", (self.device.type == "cuda")))
        self.grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP FP16) enabled.")

        # Save config immediately to both output_dir and best_checkpoint_dir
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)
        with open(self.best_checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)

        self.global_step = 0
        self.best_sim_step = 0

    def _git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _collect_provenance(self) -> Dict[str, Any]:
        paths = {
            "sim_manifest_hash": self.manifest_dir / "sim_source_manifest.json",
            "real_manifest_hash": self.manifest_dir / "real_split_manifest.json",
        }
        provenance: Dict[str, Any] = {"git_commit": self._git_commit()}
        for key, path in paths.items():
            provenance[key] = compute_sha256(path) if path.exists() else None
        provenance["basis_hash"] = (
            compute_sha256(self.pod_basis_path)
            if self.pod_basis_path is not None and self.pod_basis_path.exists()
            else None
        )
        return provenance

    def _save_sensor_manifest(self, sensor_op) -> None:
        payload = {
            "topology_type": self.sensor_topology,
            "num_sensors": self.num_sensors,
            "grid_shape": [int(sensor_op.h), int(sensor_op.w)],
            "flatten_order": "C",
            "channel_order": ["u", "v"],
            "sensor_indices": sensor_op.indices_1d.tolist(),
            "sensor_coordinates": sensor_op.coords.tolist(),
            "sensor_placement_seed": self.sensor_placement_seed,
            "canonical_sensor_manifest_hash": self.canonical_sensor_manifest_hash,
            "sensor_placement_basis_hash": self.sensor_placement_basis_hash,
            **self.provenance,
        }
        save_manifest(payload, self.output_dir / "sensor_manifest.json")
        save_manifest(payload, self.best_checkpoint_dir / "sensor_manifest.json")

    def _load_pod_basis(self) -> Optional[torch.Tensor | dict]:
        pod_path = self.config.get("pod_basis_path", None)
        if pod_path is None:
            candidates = [
                self.data_root / "pod_basis_64x128.pt",
                self.data_root / "foil" / "pod_basis_64x128.pt",
                self.data_root / "pod_basis.pt",
                Path("artifacts/pod_basis_64x128.pt"),
                Path("data/foil/pod_basis_64x128.pt"),
                Path("data/pod_basis_64x128.pt"),
            ]
            for cand in candidates:
                if cand.exists():
                    pod_path = str(cand)
                    break

        if pod_path and Path(pod_path).exists():
            self.pod_basis_path = Path(pod_path)
            self.logger.info(f"Loading POD basis from {pod_path}")
            basis = torch.load(pod_path, map_location=self.device, weights_only=True)
            if isinstance(basis, dict):
                for k in ["basis", "basis_perp", "mean", "singular_values"]:
                    if k in basis and isinstance(basis[k], torch.Tensor):
                        basis[k] = basis[k].to(self.device).float()
                return basis
            return basis.float()

        model_name = self.config.get("model_name", "")
        needs_pod = ("pod" in model_name) or (self.config.get("sensor_topology") in ("group_q_deim", "q_deim", "channel_norm_qr"))
        if needs_pod and not self.config.get("allow_random_pod_fallback", False):
            raise FileNotFoundError(
                f"POD basis not found on disk for model '{model_name}'. "
                f"Please run scripts/compute_sim_pod_basis.py before running formal training."
            )

        self.logger.warning("POD basis not found on disk. Generating temporary random orthonormal basis.")
        m = 64 * 128 * 2
        k = self.config.get("pod_rank", 64)
        q, _ = torch.linalg.qr(torch.randn(m, k, device=self.device))
        return q.float()

    def _validate_pod_basis_stage(self, expected_stage: str) -> None:
        """Fail fast when a development/final run is paired with the wrong POD basis."""
        if self.config.get("allow_basis_stage_mismatch", False):
            self.logger.warning(
                "POD basis stage validation is disabled by allow_basis_stage_mismatch; "
                "this setting is intended only for smoke tests."
            )
            return

        manifest = self.pod_basis.get("manifest", {}) if isinstance(self.pod_basis, dict) else {}
        actual_stage = manifest.get("stage") if isinstance(manifest, dict) else None
        if actual_stage is None:
            raise RuntimeError(
                "The POD basis has no embedded manifest.stage metadata, so its data scope "
                "cannot be audited. Recompute it with scripts/compute_sim_pod_basis.py."
            )
        if actual_stage != expected_stage:
            raise RuntimeError(
                f"POD basis stage mismatch: run expects '{expected_stage}', but "
                f"{self.pod_basis_path or 'the loaded basis'} records '{actual_stage}'."
            )
        self.logger.info(
            f"Validated POD basis stage='{actual_stage}' for this protocol run."
        )

    def _predict(self, batch: Dict[str, Any], is_real_finetuning: bool) -> torch.Tensor:
        sim_full = self.config.get("sim_pretrain_full", True)
        model_name = self.config.get("model_name", "").lower()
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        supports_mode = hasattr(model_ref.forward, "__code__") and "mode" in model_ref.forward.__code__.co_varnames

        if not is_real_finetuning and sim_full:
            if supports_mode:
                return self.model(batch, mode="full")
            if model_name.startswith("masked_") and "x_full" in batch:
                dense_mask = torch.ones_like(batch["x_full"][..., :1])
                dense_batch = dict(batch)
                dense_batch["x_sparse"] = torch.cat([batch["x_full"], dense_mask], dim=-1)
                return self.model(dense_batch)
            return self.model(batch)

        if supports_mode:
            return self.model(batch, mode="sparse")
        return self.model(batch)

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer,
        scheduler=None,
        is_real_finetuning: bool = False,
        max_stage_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        clip_grad = self.config.get("clip_grad_norm", 1.0)
        grad_accum = max(1, self.grad_accum_steps)
        log_every = int(self.config.get("log_every_n_steps", 50))
        max_train_batches = self.config.get("max_train_batches", None)

        for step, batch in enumerate(dataloader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.use_amp and self.device.type == "cuda")):
                pred = self._predict(batch, is_real_finetuning=is_real_finetuning)

                model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                loss = self.loss_fn(
                    pred,
                    batch,
                    model_ref,
                    is_real_finetuning=is_real_finetuning,
                    sensor_indices=self.sensor_indices,
                )
                loss_scaled = loss / grad_accum

            if self.use_amp and self.device.type == "cuda":
                self.scaler.scale(loss_scaled).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    if clip_grad > 0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], clip_grad)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    self.global_step += 1
                    self.stage_step += 1
                    if scheduler is not None:
                        scheduler.step()
            else:
                loss_scaled.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    if clip_grad > 0:
                        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    self.stage_step += 1
                    if scheduler is not None:
                        scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if log_every > 0 and self.stage_step > 0 and self.stage_step % log_every == 0:
                self.logger.info(
                    f"[step {self.stage_step}] running train loss={total_loss / n_batches:.6f}"
                )
            if max_stage_steps is not None and self.stage_step >= max_stage_steps:
                break

        return {
            "train_loss": total_loss / max(1, n_batches),
            "stage_step": float(self.stage_step),
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        is_real_finetuning: bool = False,
        target_mode: str = "dense",
    ) -> Dict[str, float]:
        """
        Streaming evaluation across validation trajectories in O(1) memory.
        Calculates fluid_fullfield_rel_l2, sensor_rel_l2, unobserved_rel_l2, and vorticity_rel_l2.
        """
        self.model.eval()
        accumulator = StreamingMetricAccumulator(
            grid_shape=self.grid_shape,
            fluid_mask=self.fluid_mask,
            sensor_indices=self.sensor_indices,
            unobserved_radius=3.0,
            dx=self.grid_dx,
            dy=self.grid_dy,
        )

        max_val_batches = self.config.get("max_val_batches", None)
        for batch_index, batch in enumerate(dataloader):
            if max_val_batches is not None and batch_index >= int(max_val_batches):
                break
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.use_amp and self.device.type == "cuda")):
                pred = self._predict(batch, is_real_finetuning=is_real_finetuning)

            stems = batch.get("traj_stem", [f"traj_{i}" for i in range(pred.size(0))])
            if target_mode == "dense":
                if "y_full" not in batch:
                    raise RuntimeError("Dense validation requires y_full, but the batch schema omitted it.")
                accumulator.update(pred, batch["y_full"], stems, sensor_values_target=batch.get("y_sensor_values"))
            elif target_mode == "sensor":
                if "y_sensor_values" not in batch or "y_full" in batch:
                    raise RuntimeError(
                        "Sensor-only validation requires y_sensor_values and forbids y_full."
                    )
                accumulator.update(
                    pred, None, stems, sensor_values_target=batch["y_sensor_values"]
                )
            else:
                raise ValueError(f"Unsupported evaluation target_mode: {target_mode}")

        return accumulator.compute()

    def fit_stage(
        self,
        stage_name: str,
        train_dataset_type: str,
        few_shot_k: Optional[int],
        epochs: int,
        lr: float,
        save_name: str,
        fixed_steps: Optional[int] = None,
    ) -> float:
        self.logger.info(f"=== Starting {stage_name}: {train_dataset_type.upper()} (Few-shot K={few_shot_k}) ===")

        # A repeated experiment name must never make a failed/interrupted run
        # appear successful by leaving a checkpoint from an earlier run in place.
        for directory in (self.output_dir, self.best_checkpoint_dir):
            stale_checkpoint = directory / f"{save_name}.pt"
            stale_checkpoint.unlink(missing_ok=True)
            stale_temporary = stale_checkpoint.with_suffix(stale_checkpoint.suffix + ".tmp")
            stale_temporary.unlink(missing_ok=True)

        stage_cfg = dict(self.config)
        stage_cfg["dataset_type"] = train_dataset_type
        stage_cfg["val_dataset_type"] = train_dataset_type
        stage_cfg["few_shot_k"] = few_shot_k
        stage_cfg["seed"] = self.seed
        stage_cfg["model_init_seed"] = self.model_init_seed
        stage_cfg["few_shot_traj_seed"] = self.few_shot_traj_seed
        stage_cfg["sensor_placement_seed"] = self.sensor_placement_seed
        stage_cfg["manifest_dir"] = str(self.manifest_dir)

        stage_id = "sim_pretrain" if train_dataset_type == "numerical" else "real_finetune"
        selector_strategy = (
            "dense_val" if train_dataset_type == "numerical"
            else stage_cfg.get("val_selection_strategy", "dense_val")
        )
        stage_cfg["val_selection_strategy"] = selector_strategy
        final_source = train_dataset_type == "numerical" and stage_cfg.get("sim_mode", "dev") == "final"
        fixed_budget = selector_strategy == "fixed_budget" and train_dataset_type == "real"
        use_validation = not final_source and not fixed_budget

        if final_source and fixed_steps is None:
            raise ValueError("Final source run requires sim_fixed_steps=S*.")
        if fixed_budget and fixed_steps is None:
            raise ValueError("Real fixed-budget selection requires fixed_steps.")

        train_loader, val_loader, _ = create_dataloaders(
            self.data_root,
            stage_cfg,
            pod_basis=self.pod_basis,
            stage=stage_id,
            include_test=False,
            include_val=use_validation,
            sensor_indices_override=self.sensor_indices.detach().cpu(),
        )
        if len(train_loader.dataset) == 0:
            raise RuntimeError(f"{stage_name} has zero training samples.")
        if use_validation and (val_loader is None or len(val_loader.dataset) == 0):
            raise RuntimeError(
                f"{stage_name} selector '{selector_strategy}' requires a non-empty validation set."
            )
        self.logger.info(
            f"Train samples: {len(train_loader.dataset)}, "
            f"Val samples: {len(val_loader.dataset) if val_loader is not None else 0}, "
            f"mmap handles/worker: {getattr(train_loader.dataset, 'mmap_cache_handles', 'unknown')}"
        )

        if train_dataset_type == "real":
            self._configure_real_finetuning()
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)

        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("Selected zero trainable parameters for training.")

        optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=self.config.get("weight_decay", 1e-4))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, self.grad_accum_steps)))
        planned_steps = int(fixed_steps) if fixed_steps is not None else max(1, epochs * steps_per_epoch)
        scheduler_steps = planned_steps
        if train_dataset_type == "numerical" and final_source:
            scheduler_steps = int(self.config.get("sim_scheduler_steps", planned_steps))
            if scheduler_steps < planned_steps:
                raise ValueError(
                    f"sim_scheduler_steps={scheduler_steps} cannot be smaller than "
                    f"sim_fixed_steps={planned_steps}."
                )
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_steps, eta_min=lr * 0.05)
        self.logger.info(
            f"Optimizer-step budget={planned_steps}; cosine scheduler horizon={scheduler_steps}."
        )

        if train_dataset_type == "numerical":
            selection_metric_key = "fluid_fullfield_rel_l2"
        else:
            if selector_strategy == "sensor_val":
                selection_metric_key = "sensor_rel_l2"
            else:
                selection_metric_key = "fluid_fullfield_rel_l2"

        best_metric = float("inf")
        best_epoch = 0
        best_step_record = 0
        patience = self.config.get("patience", 10)
        patience_counter = 0
        self.stage_step = 0
        max_epochs = max(1, math.ceil(planned_steps / steps_per_epoch)) if fixed_steps is not None else epochs

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            is_real = (train_dataset_type == "real")
            train_metrics = self.train_epoch(
                train_loader,
                optimizer,
                scheduler=scheduler,
                is_real_finetuning=is_real,
                max_stage_steps=fixed_steps,
            )
            elapsed = time.time() - t0

            if use_validation:
                target_mode = "sensor" if selector_strategy == "sensor_val" else "dense"
                val_metrics = self.evaluate(
                    val_loader,
                    is_real_finetuning=is_real,
                    target_mode=target_mode,
                )
                current_val_metric = val_metrics[selection_metric_key]
                dense_text = (
                    f"Dense Val Rel-L2: {val_metrics['fluid_fullfield_rel_l2'] * 100:.2f}%"
                    if "fluid_fullfield_rel_l2" in val_metrics else "Dense Val: not accessed"
                )
                sensor_text = (
                    f"Sensor Val Rel-L2: {val_metrics['sensor_rel_l2'] * 100:.2f}%"
                    if "sensor_rel_l2" in val_metrics else "Sensor Val: unavailable"
                )
                self.logger.info(
                    f"[{stage_name}][Epoch {epoch:03d}/{max_epochs:03d}]"
                    f"[Stage Step {self.stage_step}/{planned_steps}][{elapsed:.1f}s] "
                    f"Train Loss: {train_metrics['train_loss']:.5f} | {dense_text} | {sensor_text}"
                )

                if (best_metric - current_val_metric) > 1e-6:
                    best_metric = current_val_metric
                    best_epoch = epoch
                    best_step_record = self.stage_step
                    patience_counter = 0
                    ckpt_payload = self._checkpoint_payload(
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        val_metrics=val_metrics,
                        selection_strategy=selector_strategy,
                        selection_metric=selection_metric_key,
                        selection_value=best_metric,
                    )
                    self._save_checkpoint(save_name, ckpt_payload)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch})."
                        )
                        break
            else:
                self.logger.info(
                    f"[{stage_name}][Epoch {epoch:03d}/{max_epochs:03d}]"
                    f"[Stage Step {self.stage_step}/{planned_steps}][{elapsed:.1f}s] "
                    f"Train Loss: {train_metrics['train_loss']:.5f} | validation not accessed"
                )

            if fixed_steps is not None and self.stage_step >= fixed_steps:
                best_epoch = epoch
                best_step_record = self.stage_step
                ckpt_payload = self._checkpoint_payload(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_metrics={},
                    selection_strategy="final_source_fixed_steps" if final_source else "fixed_budget",
                    selection_metric="optimizer_steps",
                    selection_value=float(self.stage_step),
                )
                self._save_checkpoint(save_name, ckpt_payload)
                self.logger.info(f"Reached exactly {fixed_steps} stage-local optimizer steps.")
                break

        if train_dataset_type == "numerical":
            self.best_sim_step = best_step_record

        if use_validation:
            self.logger.info(
                f"Finished {stage_name}. Best {selection_metric_key}: {best_metric * 100:.2f}% "
                f"at epoch {best_epoch} (stage step {best_step_record})"
            )
        else:
            self.logger.info(
                f"Finished {stage_name} at fixed stage step {best_step_record}; validation was not accessed."
            )
        return best_metric

    def _checkpoint_payload(
        self,
        optimizer,
        scheduler,
        epoch: int,
        val_metrics: Dict[str, float],
        selection_strategy: str,
        selection_metric: str,
        selection_value: float,
    ) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "stage_step": self.stage_step,
            "val_metrics": val_metrics,
            "selection_strategy": selection_strategy,
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "model_init_seed": self.model_init_seed,
            "few_shot_traj_seed": self.few_shot_traj_seed,
            "sensor_placement_seed": self.sensor_placement_seed,
            "sensor_indices": self.sensor_indices.detach().cpu(),
            "sensor_coordinates": self.sensor_coords.detach().cpu(),
            **self.provenance,
        }

    def _save_checkpoint(self, save_name: str, payload: Dict[str, Any]) -> None:
        for directory in (self.output_dir, self.best_checkpoint_dir):
            target = directory / f"{save_name}.pt"
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(target)

    def _configure_real_finetuning(self) -> None:
        scope = self.config.get("real_finetune_scope", "adaptation")
        if scope == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            self.logger.info("Real fine-tuning scope: full model")
            return

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        trainable_modules = []
        if hasattr(self.model, "warping") and self.model.warping is not None:
            for parameter in self.model.warping.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("warping")
        if scope in ("adaptation_dynamics", "adaptation") and hasattr(self.model, "modal_prop"):
            for parameter in self.model.modal_prop.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("modal_prop")
        if scope in ("adaptation_dynamics", "adaptation") and hasattr(self.model, "phca"):
            for parameter in self.model.phca.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("phca")
        if scope in ("adaptation_dynamics", "adaptation") and hasattr(self.model, "ldno"):
            for parameter in self.model.ldno.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("ldno")

        if not trainable_modules:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("full_model_fallback")

        self.logger.info(f"Real fine-tuning scope: {scope} ({', '.join(trainable_modules)})")

    def run(self):
        # Handle training-free classical baseline
        model_name = self.config.get("model_name", "")
        sim_epochs = self.config.get("sim_epochs", 100)
        real_epochs = self.config.get("real_epochs", 50)

        if "classical" in model_name or (sim_epochs == 0 and real_epochs == 0):
            self._validate_pod_basis_stage("final")
            self.logger.info(f"Detected training-free baseline '{model_name}'. Archiving state.")
            payload = {
                "model_state_dict": self.model.state_dict(),
                "selection_strategy": "training_free",
                "epoch": None,
                "global_step": 0,
                "sensor_indices": self.sensor_indices.detach().cpu(),
                "sensor_coordinates": self.sensor_coords.detach().cpu(),
                **self.provenance,
            }
            self._save_checkpoint("classical_gappy_pod", payload)
            with open(self.best_checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True)
            shutil.copy(self.output_dir / "train.log", self.best_checkpoint_dir / "train.log")
            return

        # Stage 1: Sim Pre-training (Dev Run or Final Source Run)
        sim_mode = self.config.get("sim_mode", "dev")
        if sim_mode not in ("dev", "final"):
            raise ValueError(f"sim_mode must be 'dev' or 'final', got {sim_mode!r}.")
        if sim_epochs > 0:
            self._validate_pod_basis_stage(sim_mode)
        elif real_epochs > 0:
            # A Real-only continuation is meaningful only against the frozen all-Sim basis.
            self._validate_pod_basis_stage("final")
        sim_lr = self.config.get("sim_lr", 2e-4)
        sim_save_name = "sim_pretrained_final" if sim_mode == "final" else "best_sim"
        if sim_epochs > 0:
            fixed_s = self.config.get("sim_fixed_steps", None) if sim_mode == "final" else None
            self.fit_stage(
                stage_name="Stage 1 (Sim Pre-training)",
                train_dataset_type="numerical",
                few_shot_k=None,
                epochs=sim_epochs,
                lr=sim_lr,
                save_name=sim_save_name,
                fixed_steps=fixed_s,
            )

        # Load best sim checkpoint before finetuning
        sim_ckpt = self.output_dir / f"{sim_save_name}.pt"
        if sim_epochs > 0:
            if not sim_ckpt.exists():
                raise RuntimeError(
                    f"Sim training finished without the required checkpoint: {sim_ckpt}."
                )
            ckpt_data = torch.load(sim_ckpt, map_location=self.device, weights_only=True)
            state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded best Sim weights from {sim_ckpt} for Real finetuning.")
        elif real_epochs > 0:
            continuation = self.config.get("pretrained_checkpoint")
            if continuation is None:
                raise ValueError(
                    "A Real-only run requires pretrained_checkpoint; refusing to fine-tune a "
                    "randomly initialized model or reuse a stale file from the run directory."
                )
            continuation_path = Path(continuation)
            if not continuation_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {continuation_path}")
            ckpt_data = torch.load(continuation_path, map_location=self.device, weights_only=True)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            self.model.load_state_dict(state_dict, strict=True)
            self.logger.info(f"Loaded explicit pretrained checkpoint: {continuation_path}")

        # Stage 2: Few-shot Real Finetuning
        strategy = self.config.get("val_selection_strategy", "dense_val")
        if strategy == "sensor_val":
            save_target = "best_sensor_val"
        elif strategy == "fixed_budget":
            steps_budget = self.config.get("fixed_steps", 500)
            save_target = f"fixed_budget_step_{steps_budget}"
        else:
            save_target = "best_dense_val"

        real_lr = self.config.get("real_lr", 1e-4)
        few_shot_k = self.config.get("few_shot_k", 3)
        if real_epochs > 0:
            real_fixed_steps = int(self.config.get("fixed_steps", 500)) if strategy == "fixed_budget" else None
            self.fit_stage(
                stage_name="Stage 2 (Real Fine-tuning)",
                train_dataset_type="real",
                few_shot_k=few_shot_k,
                epochs=real_epochs,
                lr=real_lr,
                save_name=save_target,
                fixed_steps=real_fixed_steps,
            )

        # Archive complete assets to best_checkpoints
        with open(self.best_checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)

        shutil.copy(self.output_dir / "train.log", self.best_checkpoint_dir / "train.log")
        self.logger.info(f"Run completed successfully! Archived artifacts to {self.best_checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="SparsePOD-Sim2Real Model Trainer")
    parser.add_argument("--config", type=Path, default=None, help="Path to single config yaml")
    parser.add_argument("--config-dir", type=Path, default=None, help="Directory containing config yamls to run sequentially")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--data-root", type=Path, default=None, help="Override dataset root path")
    parser.add_argument("--pod-basis", type=Path, default=None, help="Override path to pod_basis.pt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override artifacts/runs output directory")
    parser.add_argument("--checkpoints-dir", type=Path, default=None, help="Override best_checkpoints directory")
    parser.add_argument("--sim-epochs", type=int, default=None, help="Override number of sim pre-training epochs")
    parser.add_argument("--real-epochs", type=int, default=None, help="Override number of real fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--few-shot-k", type=int, default=None, help="Override few-shot k")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--selector", type=str, choices=["dense_val", "sensor_val", "fixed_budget"], default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    config_files = []
    if args.config:
        config_files.append(args.config)
    elif args.config_dir and args.config_dir.exists():
        config_files.extend(sorted(args.config_dir.glob("*.yaml")))

    if not config_files:
        print("Error: No valid config files provided via --config or --config-dir.")
        sys.exit(1)

    for cfg_path in config_files:
        print(f"\n{'='*80}\nExecuting Config: {cfg_path}\n{'='*80}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if args.data_root is not None:
            cfg["data_root"] = str(args.data_root)
        elif "DATA_ROOT" in os.environ:
            cfg["data_root"] = os.environ["DATA_ROOT"]

        if args.pod_basis is not None:
            cfg["pod_basis_path"] = str(args.pod_basis)
        if args.output_dir is not None:
            cfg["output_dir"] = str(args.output_dir)
        if args.checkpoints_dir is not None:
            cfg["best_checkpoints_dir"] = str(args.checkpoints_dir)
        if args.sim_epochs is not None:
            cfg["sim_epochs"] = args.sim_epochs
        if args.real_epochs is not None:
            cfg["real_epochs"] = args.real_epochs
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        if args.few_shot_k is not None:
            cfg["few_shot_k"] = args.few_shot_k
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.selector is not None:
            cfg["val_selection_strategy"] = args.selector

        trainer = Sim2RealTrainer(cfg, device=device)
        trainer.run()


if __name__ == "__main__":
    main()
