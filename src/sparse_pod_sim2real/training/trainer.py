from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import yaml

from ..data.dataset import SparseTrajectoryDataset, create_dataloaders
from ..data.normalizer import GaussianNormalizer, IdentityNormalizer
from ..model import load_model
from .losses import CompositePhysicalLoss, relative_l2_per_sample
from .metrics import compute_metrics


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(log_file.stem)
    logger.setLevel(logging.INFO)
    logger.handlers = []

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

        # Setup run directories
        self.exp_name = config.get("exp_name", "sim2real_run")
        self.output_dir = Path(config.get("output_dir", "artifacts/runs")) / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_dir = Path(config.get("best_checkpoints_dir", "best_checkpoints")) / self.exp_name
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(self.output_dir / "train.log")
        self.logger.info(f"Initialized Sim2RealTrainer on device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Save config immediately to both output_dir and best_checkpoint_dir
        with open(self.output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)
        with open(self.best_checkpoint_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)

        # Load or create POD basis
        self.pod_basis = self._load_pod_basis()
        if self.pod_basis is not None:
            torch.save(self.pod_basis, self.best_checkpoint_dir / "pod_basis.pt")

        # Build model and get sensor indices
        self.num_sensors = config.get("num_sensors", 64)
        self.sensor_topology = config.get("sensor_topology", "wall")
        temp_ds = SparseTrajectoryDataset(
            data_root=self.data_root,
            dataset_type="numerical",
            topology_type=self.sensor_topology,
            num_sensors=self.num_sensors,
            pod_basis=self.pod_basis,
        )
        self.sensor_indices = temp_ds.sensor_op.indices_1d.to(self.device)

        self.model = load_model(config, pod_basis=self.pod_basis, sensor_indices=self.sensor_indices).to(self.device)
        self.loss_fn = CompositePhysicalLoss(
            v_weight=config.get("v_weight", 2.0),
            vorticity_weight=config.get("vorticity_weight", 0.1),
            sensor_weight=config.get("sensor_weight", 1.0),
            ortho_weight=config.get("ortho_weight", 0.01),
            supervision_mode=config.get("supervision_mode", "sparse_sensors"),
        )
        self.use_amp = bool(config.get("use_amp", (self.device.type == "cuda")))
        self.grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP FP16) enabled for high throughput and reduced memory.")

    def _load_pod_basis(self) -> Optional[torch.Tensor | dict]:
        pod_path = self.config.get("pod_basis_path", None)
        if pod_path is None:
            # Check default locations
            candidates = [
                self.data_root / "pod_basis_64x128.pt",
                self.data_root / "pod_basis.pt",
                Path("artifacts/pod_basis_64x128.pt"),
                Path("data/pod_basis_64x128.pt"),
                Path("data/foil/pod_basis_64x128.pt"),
                Path("../POD-Sim2Real/data/pod_basis_64x128.pt"),
            ]
            for cand in candidates:
                if cand.exists():
                    pod_path = str(cand)
                    break

        if pod_path and Path(pod_path).exists():
            self.logger.info(f"Loading POD basis from {pod_path}")
            basis = torch.load(pod_path, map_location=self.device)
            # If dictionary payload, move contained tensors to device and return dict
            if isinstance(basis, dict):
                for k in ["basis", "basis_perp", "mean", "singular_values"]:
                    if k in basis and isinstance(basis[k], torch.Tensor):
                        basis[k] = basis[k].to(self.device).float()
                return basis
            return basis.float()

        # Fallback orthonormal basis if not yet computed (for smoke test/dry-run)
        self.logger.warning("POD basis not found on disk. Generating random orthonormal basis.")
        m = 64 * 128 * 2
        k = self.config.get("pod_rank", 64)
        q, _ = torch.linalg.qr(torch.randn(m, k, device=self.device))
        return q.float()

    def train_epoch(self, dataloader, optimizer, is_real_finetuning: bool = False) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        sim_full = self.config.get("sim_pretrain_full", True)
        clip_grad = self.config.get("clip_grad_norm", 1.0)
        grad_accum = max(1, self.grad_accum_steps)

        for step, batch in enumerate(dataloader):
            # Move tensors to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.use_amp and self.device.type == "cuda")):
                # Sim Pre-training uses 100% complete CFD flow fields (sim_pretrain_full=True)
                if not is_real_finetuning and sim_full:
                    if hasattr(self.model, "forward") and "mode" in self.model.forward.__code__.co_varnames:
                        pred = self.model(batch, mode="full")
                    elif "x_sparse" in batch and "x_full" in batch:
                        # Masked baseline: in dense Sim pre-training, provide complete flow + active mask
                        dense_with_mask = torch.cat([batch["x_full"], torch.ones_like(batch["x_sparse"][..., -1:])], dim=-1)
                        b_full = dict(batch)
                        b_full["x_sparse"] = dense_with_mask
                        pred = self.model(b_full)
                    else:
                        pred = self.model(batch)
                else:
                    # Real Fine-tuning: strictly operates on sparse sensor inputs
                    pred = self.model(batch)

                target_full = batch.get("y_full", None)
                target_sensors = batch.get("y_sensor_values", None)

                loss, loss_dict = self.loss_fn(
                    pred,
                    target_full=target_full,
                    target_sensors=target_sensors,
                    sensor_indices=self.sensor_indices,
                    model=self.model,
                    is_real_finetuning=is_real_finetuning,
                )
                loss_scaled = loss / grad_accum

            if self.scaler.is_enabled():
                self.scaler.scale(loss_scaled).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    if clip_grad > 0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], clip_grad)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    if clip_grad > 0:
                        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        return {"train_loss": total_loss / max(1, n_batches)}


    @torch.no_grad()
    def evaluate(self, dataloader, is_real_finetuning: bool = False) -> Dict[str, float]:
        self.model.eval()
        preds = []
        targets = []

        sim_full = self.config.get("sim_pretrain_full", True)
        eval_full = (not is_real_finetuning) and sim_full

        for batch in dataloader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.use_amp and self.device.type == "cuda")):
                if eval_full:
                    if hasattr(self.model, "forward") and "mode" in self.model.forward.__code__.co_varnames:
                        pred = self.model(batch, mode="full")
                    elif "x_sparse" in batch and "x_full" in batch:
                        dense_with_mask = torch.cat([batch["x_full"], torch.ones_like(batch["x_sparse"][..., -1:])], dim=-1)
                        b_full = dict(batch)
                        b_full["x_sparse"] = dense_with_mask
                        pred = self.model(b_full)
                    else:
                        pred = self.model(batch)
                else:
                    if hasattr(self.model, "forward") and "mode" in self.model.forward.__code__.co_varnames:
                        pred = self.model(batch, mode="sparse")
                    else:
                        pred = self.model(batch)

            preds.append(pred.float().cpu())
            targets.append(batch["y_full"].cpu())

        if not preds:
            return {"rel_l2": 1.0, "rmse": 1.0}

        all_preds = torch.cat(preds, dim=0)
        all_targets = torch.cat(targets, dim=0)
        return compute_metrics(all_preds, all_targets)

    def fit_stage(
        self,
        stage_name: str,
        train_dataset_type: str,
        few_shot_k: Optional[int],
        epochs: int,
        lr: float,
        save_name: str,
    ) -> float:
        self.logger.info(f"=== Starting {stage_name}: {train_dataset_type.upper()} (Few-shot K={few_shot_k}) ===")

        stage_cfg = dict(self.config)
        stage_cfg["dataset_type"] = train_dataset_type
        stage_cfg["val_dataset_type"] = train_dataset_type  # strictly isolates validation split by stage
        stage_cfg["few_shot_k"] = few_shot_k

        train_loader, val_loader, _ = create_dataloaders(self.data_root, stage_cfg, pod_basis=self.pod_basis)
        self.logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

        if train_dataset_type == "real":
            self._configure_real_finetuning()
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)

        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("Real fine-tuning selected no trainable parameters.")
        optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=self.config.get("weight_decay", 1e-4))
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.05)

        best_val_rel_l2 = float("inf")
        best_epoch = 0
        patience = self.config.get("patience", 10)
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            is_real = (train_dataset_type == "real")
            train_metrics = self.train_epoch(train_loader, optimizer, is_real_finetuning=is_real)
            scheduler.step()

            val_metrics = self.evaluate(val_loader, is_real_finetuning=is_real)
            elapsed = time.time() - t0

            val_rel_l2 = val_metrics["rel_l2"]
            self.logger.info(
                f"[{stage_name}][Epoch {epoch:03d}/{epochs:03d}][{elapsed:.1f}s] "
                f"Train Loss: {train_metrics['train_loss']:.5f} | "
                f"Val Rel-L2: {val_rel_l2 * 100:.2f}% | "
                f"Val RMSE: {val_metrics['rmse']:.5f} | "
                f"Vorticity Rel-L2: {val_metrics['vorticity_rel_l2'] * 100:.2f}%"
            )

            if val_rel_l2 < best_val_rel_l2:
                best_val_rel_l2 = val_rel_l2
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / f"{save_name}.pt")
                torch.save(self.model.state_dict(), self.best_checkpoint_dir / f"{save_name}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                    break

        self.logger.info(f"Finished {stage_name}. Best Val Rel-L2: {best_val_rel_l2 * 100:.2f}% at epoch {best_epoch}")
        return best_val_rel_l2

    def _configure_real_finetuning(self) -> None:
        """Select an explicit adaptation scope for few-shot Real training."""
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
            # Baselines have no named adaptation modules; retain their full model.
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            trainable_modules.append("full_model_fallback")
        self.logger.info(f"Real fine-tuning scope: {scope} ({', '.join(trainable_modules)})")

    def run(self):
        # Stage 1: Sim Pre-training
        sim_epochs = self.config.get("sim_epochs", 100)
        sim_lr = self.config.get("sim_lr", 2e-4)
        if sim_epochs > 0:
            self.fit_stage("Stage 1 (Sim Pre-training)", "numerical", None, sim_epochs, sim_lr, "best_sim")

        # Load best sim checkpoint before finetuning
        sim_ckpt = self.output_dir / "best_sim.pt"
        if sim_ckpt.exists():
            self.model.load_state_dict(torch.load(sim_ckpt, map_location=self.device))
            self.logger.info(f"Loaded best Sim weights from {sim_ckpt} for Real finetuning.")

        # Stage 2: Few-shot Real Finetuning
        real_epochs = self.config.get("real_epochs", 50)
        real_lr = self.config.get("real_lr", 1e-4)
        few_shot_k = self.config.get("few_shot_k", 3)
        if real_epochs > 0:
            self.fit_stage("Stage 2 (Real Fine-tuning)", "real", few_shot_k, real_epochs, real_lr, "best")

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

        # Apply CLI overrides if provided
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

        trainer = Sim2RealTrainer(cfg, device=device)
        trainer.run()


if __name__ == "__main__":
    main()
