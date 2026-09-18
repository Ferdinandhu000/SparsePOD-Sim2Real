#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sparse_pod_sim2real.training.trainer import Sim2RealTrainer  # noqa: E402


def run_protocol(
    config: Dict,
    device: torch.device,
    dev_basis: Path,
    final_basis: Path,
) -> None:
    exp_name = config.get("exp_name", "sim2real_run")
    model_name = config.get("model_name", "").lower()
    training_free = "classical" in model_name or (
        int(config.get("sim_epochs", 0)) == 0 and int(config.get("real_epochs", 0)) == 0
    )

    if training_free:
        cfg = dict(config)
        cfg["pod_basis_path"] = str(final_basis)
        cfg["protocol_role"] = "final_report"
        Sim2RealTrainer(cfg, device=device).run()
        return

    # Development run: train-block POD and dense Sim validation select S*.
    dev_cfg = dict(config)
    dev_cfg.update({
        "exp_name": f"{exp_name}__development",
        "pod_basis_path": str(dev_basis),
        "sim_mode": "dev",
        "real_epochs": 0,
        "val_selection_strategy": "dense_val",
        "protocol_role": "development_selection",
    })
    dev_trainer = Sim2RealTrainer(dev_cfg, device=device)
    dev_trainer.run()

    dev_checkpoint = dev_trainer.output_dir / "best_sim.pt"
    if not dev_checkpoint.exists():
        raise RuntimeError(f"Development run did not produce {dev_checkpoint}")
    payload = torch.load(dev_checkpoint, map_location="cpu", weights_only=True)
    source_steps = int(payload.get("stage_step", payload.get("global_step", 0)))
    if source_steps <= 0:
        raise RuntimeError("Development checkpoint contains an invalid S* optimizer-step budget.")
    source_scheduler_steps = int(
        payload.get("scheduler_state_dict", {}).get("T_max", source_steps)
    )
    if source_scheduler_steps < source_steps:
        raise RuntimeError("Development checkpoint contains an invalid LR-scheduler horizon.")

    # Final run: rebuild model against the all-frame POD basis and use exactly S*
    # source optimizer steps before Real adaptation.
    final_cfg = dict(config)
    final_cfg.update({
        "exp_name": exp_name,
        "pod_basis_path": str(final_basis),
        "sim_mode": "final",
        "sim_fixed_steps": source_steps,
        # Preserve the development learning-rate schedule as a function of step;
        # only the stopping point changes to S*.
        "sim_scheduler_steps": source_scheduler_steps,
        "protocol_role": "final_report",
    })
    Sim2RealTrainer(final_cfg, device=device).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Development -> Final Source -> Real adaptation protocol")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--dev-basis", type=Path, default=Path("artifacts/pod_basis_dev_64x128.pt"))
    parser.add_argument("--final-basis", type=Path, default=Path("artifacts/pod_basis_final_64x128.pt"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config_files: List[Path] = []
    if args.config is not None:
        config_files = [args.config]
    elif args.config_dir is not None:
        config_files = sorted(args.config_dir.glob("*.yaml"))
    if not config_files:
        raise SystemExit("No configuration files found.")
    if not args.dev_basis.exists() or not args.final_basis.exists():
        raise FileNotFoundError("Both development and final POD basis files are required.")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    for path in config_files:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if args.data_root is not None:
            config["data_root"] = str(args.data_root)
        config["manifest_dir"] = str(args.manifest_dir)
        run_protocol(config, device, args.dev_basis, args.final_basis)


if __name__ == "__main__":
    main()
