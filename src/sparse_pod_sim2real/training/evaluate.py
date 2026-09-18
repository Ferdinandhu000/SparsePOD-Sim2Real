from __future__ import annotations

import argparse
import datetime
from functools import partial
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ..data.dataset import (
    SparseTrajectoryDataset,
    SelectiveBatchCollator,
    required_batch_fields,
    _get_worker_mmap_tensor,
    compute_mmap_cache_handles,
    worker_init_fn,
)
from ..model import load_model
from .metrics import StreamingMetricAccumulator, compute_metrics, build_naca0025_fluid_mask


def rollout_target_bounds(t_start: int, tin: int, tout: int, round_index: int) -> Tuple[int, int]:
    if round_index < 1:
        raise ValueError("round_index is one-based and must be >= 1.")
    start = t_start + tin + (round_index - 1) * tout
    return start, start + tout


def autoregressive_rollout(
    model: torch.nn.Module,
    dataset: SparseTrajectoryDataset,
    traj_stem: str,
    t_start: int,
    rounds: int = 3,
    device: torch.device = torch.device("cpu"),
    grid_dx: float = 1.0,
    grid_dy: float = 1.0,
) -> List[Dict[str, float]]:
    """
    Performs multi-round autoregressive rollout with strictly progressing future ground truth horizons:
    Round r targets slice: [t_start + Tin + (r-1)*Tout : t_start + Tin + r*Tout]
    """
    results = []
    cumulative_predictions = []
    cumulative_targets = []
    tin = dataset.in_step
    tout = dataset.out_step
    total_required = tin + rounds * tout

    path, t_bound_start, t_bound_end = dataset.trajectory_info[traj_stem]
    traj_tensor = _get_worker_mmap_tensor(path)

    # Validate frame boundary
    if t_start + total_required > t_bound_end:
        return results

    # Initial historical context: [Tin, H, W, 2]
    window_init = traj_tensor[t_start : t_start + tin]
    if window_init.shape[1] == 2 and window_init.ndim == 4:
        full_in = window_init.permute(0, 2, 3, 1).contiguous().float()
    else:
        full_in = window_init.contiguous().float()

    sensor_values = dataset.sensor_op.extract_sensor_values(full_in).unsqueeze(0).to(device)  # [1, Tin, Ps, 2]
    curr_batch = {
        "sensor_values": sensor_values,
        "sensor_coords": dataset.sensor_op.coords.unsqueeze(0).to(device),
        "sensor_mask": dataset.sensor_op.mask.to(device),
        "traj_stem": [traj_stem],
        "t_start": [t_start],
    }
    if "x_sparse" in required_batch_fields(model.__class__.__name__, stage="test", supervision_mode="sparse_sensors"):
        curr_batch["x_sparse"] = dataset.sensor_op.to_sparse_tensor(
            full_in,
            append_mask=True,
            fill_method=dataset.sparse_fill_method,
        ).unsqueeze(0).to(device)

    for r in range(1, rounds + 1):
        # Target slice for round r: [t_start + Tin + (r-1)*Tout : t_start + Tin + r*Tout]
        t_target_start, t_target_end = rollout_target_bounds(t_start, tin, tout, r)
        target_raw = traj_tensor[t_target_start:t_target_end]
        if target_raw.shape[1] == 2 and target_raw.ndim == 4:
            target_slice = target_raw.permute(0, 2, 3, 1).contiguous().float()
        else:
            target_slice = target_raw.contiguous().float()
        target_slice = target_slice.unsqueeze(0)  # [1, Tout, H, W, 2]

        with torch.no_grad():
            pred = model(curr_batch)  # [1, Tout, H, W, 2]

        # Compute metric for this horizon
        pred_cpu = pred.cpu()
        target_cpu = target_slice.cpu()
        cumulative_predictions.append(pred_cpu)
        cumulative_targets.append(target_cpu)
        metric_kwargs = {
            "fluid_mask": build_naca0025_fluid_mask((pred.shape[2], pred.shape[3])),
            "dx": grid_dx,
            "dy": grid_dy,
        }
        m = compute_metrics(pred_cpu, target_cpu, **metric_kwargs)
        cumulative = compute_metrics(
            torch.cat(cumulative_predictions, dim=1),
            torch.cat(cumulative_targets, dim=1),
            **metric_kwargs,
        )
        m.update({f"cumulative_{key}": value for key, value in cumulative.items()})
        m["round"] = r
        m["horizon_start"] = (r - 1) * tout
        m["horizon_end"] = r * tout
        results.append(m)

        # Autoregressive update for next round
        # Next historical sensor readings extracted from model predictions
        next_sensors = dataset.sensor_op.extract_sensor_values(pred)  # [1, Tout, Ps, 2]
        if tout == tin:
            curr_batch["sensor_values"] = next_sensors
        elif tout > tin:
            curr_batch["sensor_values"] = next_sensors[:, -tin:]
        else:
            # Shift historical sensor values and append prediction
            curr_batch["sensor_values"] = torch.cat([curr_batch["sensor_values"][:, tout:], next_sensors], dim=1)

        if "x_sparse" in curr_batch:
            next_sparse = dataset.sensor_op.to_sparse_tensor(
                pred[0],
                append_mask=True,
                fill_method=dataset.sparse_fill_method,
            ).unsqueeze(0)
            if tout == tin:
                curr_batch["x_sparse"] = next_sparse
            elif tout > tin:
                curr_batch["x_sparse"] = next_sparse[:, -tin:]
            else:
                curr_batch["x_sparse"] = torch.cat([curr_batch["x_sparse"][:, tout:], next_sparse], dim=1)

    return results


def evaluate_checkpoint_domain(
    model: torch.nn.Module,
    test_ds: SparseTrajectoryDataset,
    batch_size: int,
    device: torch.device,
    grid_dx: float = 1.0,
    grid_dy: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, float]:
    """Runs streaming non-overlapping evaluation across a test dataset domain."""
    test_fields = test_ds.required_fields

    trajectory_sizes = [path.stat().st_size for path in test_ds.tensor_dir.glob("*.pt")]
    cache_handles = compute_mmap_cache_handles(
        max(trajectory_sizes, default=0), max(1, num_workers)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=(
            partial(worker_init_fn, max_cache_handles=cache_handles)
            if num_workers > 0 else None
        ),
        collate_fn=SelectiveBatchCollator(test_fields),
    )

    accumulator = StreamingMetricAccumulator(
        grid_shape=(64, 128),
        fluid_mask=build_naca0025_fluid_mask((64, 128)),
        sensor_indices=test_ds.sensor_op.indices_1d,
        unobserved_radius=3.0,
        dx=grid_dx,
        dy=grid_dy,
    )

    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            if hasattr(model, "forward") and "mode" in model.forward.__code__.co_varnames:
                pred = model(batch, mode="sparse")
            else:
                pred = model(batch)

            stems = batch.get("traj_stem", [f"traj_{i}" for i in range(pred.size(0))])
            accumulator.update(pred, batch["y_full"], stems, sensor_values_target=batch.get("y_sensor_values"))

    metrics = accumulator.compute()
    # Attach frame accounting
    metrics.update({f"fa_{k}": v for k, v in test_ds.frame_accounting.items()})
    return metrics


def select_checkpoint_file(
    checkpoint_dir: Path,
    config: Dict[str, Any],
    selector_preference: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the checkpoint that corresponds to the run's declared protocol.

    A final two-stage run normally contains both ``sim_pretrained_final.pt`` and
    a Real-adapted checkpoint. Selecting by a generic filename priority can
    therefore evaluate the pre-adaptation model by mistake. This resolver uses
    the archived configuration as the source of truth and fails closed when the
    expected adapted checkpoint is absent.
    """
    if selector_preference:
        selector = selector_preference.removesuffix(".pt")
        aliases = {
            "dense_val": "best_dense_val",
            "sensor_val": "best_sensor_val",
        }
        selector = aliases.get(selector, selector)
        if selector in {"fixed_budget", "fixed_budget_step"}:
            steps = int(config.get("fixed_steps", 500))
            target = checkpoint_dir / f"fixed_budget_step_{steps}.pt"
        else:
            target = checkpoint_dir / f"{selector}.pt"
        return target if target.exists() else None

    model_name = str(config.get("model_name", "")).lower()
    sim_epochs = int(config.get("sim_epochs", 0))
    real_epochs = int(config.get("real_epochs", 0))
    if "classical" in model_name or (sim_epochs == 0 and real_epochs == 0):
        target = checkpoint_dir / "classical_gappy_pod.pt"
        return target if target.exists() else None

    if real_epochs > 0:
        strategy = str(config.get("val_selection_strategy", "dense_val"))
        if strategy == "dense_val":
            target = checkpoint_dir / "best_dense_val.pt"
        elif strategy == "sensor_val":
            target = checkpoint_dir / "best_sensor_val.pt"
        elif strategy == "fixed_budget":
            steps = int(config.get("fixed_steps", 500))
            target = checkpoint_dir / f"fixed_budget_step_{steps}.pt"
        else:
            raise ValueError(f"Unsupported checkpoint selection strategy: {strategy!r}")
        return target if target.exists() else None

    sim_mode = str(config.get("sim_mode", "dev"))
    target = checkpoint_dir / (
        "sim_pretrained_final.pt" if sim_mode == "final" else "best_sim.pt"
    )
    return target if target.exists() else None


def evaluate_checkpoint(
    checkpoint_dir: Path,
    data_root: Path,
    device: torch.device,
    selector_preference: Optional[str] = None,
) -> Dict[str, Any]:
    cfg_file = checkpoint_dir / "config.yaml"
    if not cfg_file.exists():
        return {}

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if (
        config.get("protocol_role") == "development_selection"
        or str(config.get("exp_name", checkpoint_dir.name)).endswith("__development")
    ):
        return {}

    weights_file = select_checkpoint_file(
        checkpoint_dir, config, selector_preference=selector_preference
    )
    if weights_file is None:
        return {}

    print(f"Loading checkpoint weights from: {weights_file.name}")

    # Load POD basis if needed
    pod_path = config.get("pod_basis_path", None)
    pod_basis = None
    if pod_path and Path(pod_path).exists():
        pod_basis = torch.load(pod_path, map_location=device, weights_only=True)
    else:
        candidates = [
            checkpoint_dir / "pod_basis.pt",
            data_root / "pod_basis_64x128.pt",
            data_root / "foil" / "pod_basis_64x128.pt",
            Path("artifacts/pod_basis_64x128.pt"),
            Path("data/foil/pod_basis_64x128.pt"),
        ]
        for cand in candidates:
            if cand.exists():
                pod_basis = torch.load(cand, map_location=device, weights_only=True)
                break

    # Sensor indices: prioritize archived sensor_indices.pt for perfect reproducibility
    sensor_indices = None
    sensor_idx_path = checkpoint_dir / "sensor_indices.pt"
    if sensor_idx_path.exists():
        sensor_indices = torch.load(sensor_idx_path, map_location=device, weights_only=True)

    # Build model
    model = load_model(config, pod_basis=pod_basis, sensor_indices=sensor_indices).to(device)
    raw_payload = torch.load(weights_file, map_location=device, weights_only=True)
    state_dict = raw_payload.get("model_state_dict", raw_payload) if isinstance(raw_payload, dict) else raw_payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    batch_size = config.get("batch_size", 8)
    seed = config.get("few_shot_traj_seed", config.get("seed", 42))
    sensor_seed = config.get("sensor_placement_seed", config.get("seed", 42))
    in_step = config.get("in_step", 20)
    out_step = config.get("out_step", 20)
    manifest_dir = config.get("manifest_dir", None)
    test_fields = required_batch_fields(
        config.get("model_name", ""), stage="test",
        supervision_mode=config.get("supervision_mode", "sparse_sensors"),
        selection_strategy="dense_val",
    )

    # 1. Evaluate In-Distribution Test (Subsonic)
    test_id_ds = SparseTrajectoryDataset(
        data_root=data_root,
        dataset_type="real",
        mode="test",
        test_mode="in_dist",
        topology_type=config.get("sensor_topology", "wall"),
        num_sensors=config.get("num_sensors", 64),
        in_step=in_step,
        out_step=out_step,
        interval=out_step,  # Stride = Tout
        seed=seed,
        sensor_seed=sensor_seed,
        pod_basis=pod_basis,
        manifest_dir=manifest_dir,
        required_fields=test_fields,
        allow_manifest_fallback=bool(config.get("allow_manifest_fallback", False)),
        sensor_indices_override=sensor_indices.detach().cpu() if sensor_indices is not None else None,
    )
    id_metrics = evaluate_checkpoint_domain(
        model, test_id_ds, batch_size, device,
        grid_dx=float(config.get("grid_dx", 1.0)),
        grid_dy=float(config.get("grid_dy", 1.0)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", device.type == "cuda")),
    )

    # 2. Evaluate Out-of-Distribution Test (Transonic)
    test_ood_ds = SparseTrajectoryDataset(
        data_root=data_root,
        dataset_type="real",
        mode="test",
        test_mode="out_dist",
        topology_type=config.get("sensor_topology", "wall"),
        num_sensors=config.get("num_sensors", 64),
        in_step=in_step,
        out_step=out_step,
        interval=out_step,  # Stride = Tout
        seed=seed,
        sensor_seed=sensor_seed,
        pod_basis=pod_basis,
        manifest_dir=manifest_dir,
        required_fields=test_fields,
        allow_manifest_fallback=bool(config.get("allow_manifest_fallback", False)),
        sensor_indices_override=sensor_indices.detach().cpu() if sensor_indices is not None else None,
    )
    ood_metrics = evaluate_checkpoint_domain(
        model, test_ood_ds, batch_size, device,
        grid_dx=float(config.get("grid_dx", 1.0)),
        grid_dy=float(config.get("grid_dy", 1.0)),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", device.type == "cuda")),
    )

    # 3. Autoregressive Rollout Evaluation (sample trajectory)
    rollout_results = []
    if len(test_id_ds.samples) > 0:
        sample_stem, sample_t = test_id_ds.samples[0]
        rollout_results = autoregressive_rollout(
            model=model,
            dataset=test_id_ds,
            traj_stem=sample_stem,
            t_start=sample_t,
            rounds=3,
            device=device,
            grid_dx=float(config.get("grid_dx", 1.0)),
            grid_dy=float(config.get("grid_dy", 1.0)),
        )

    summary = {
        "checkpoint": str(weights_file.resolve()),
        "checkpoint_name": weights_file.name,
        "selection_strategy": raw_payload.get("selection_strategy", "unknown") if isinstance(raw_payload, dict) else "unknown",
        "id_test_metrics": id_metrics,
        "ood_test_metrics": ood_metrics,
        "rollout_3rounds": rollout_results,
    }

    # Save results sidecar
    with open(checkpoint_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _print_result(checkpoint_dir: Path, result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY: {checkpoint_dir.name} ({result['checkpoint_name']})")
    print("=" * 80)
    id_m = result["id_test_metrics"]
    ood_m = result["ood_test_metrics"]
    print(
        f"In-Distribution (Subsonic)  Rel-L2: {id_m['fluid_fullfield_rel_l2']*100:.2f}% | "
        f"Sensor: {id_m['sensor_rel_l2']*100:.2f}% | Vorticity: {id_m['vorticity_rel_l2']*100:.2f}%"
    )
    print(
        f"Out-of-Distribution (Trans) Rel-L2: {ood_m['fluid_fullfield_rel_l2']*100:.2f}% | "
        f"Sensor: {ood_m['sensor_rel_l2']*100:.2f}% | Vorticity: {ood_m['vorticity_rel_l2']*100:.2f}%"
    )
    if result["rollout_3rounds"]:
        print("Autoregressive Rollout (3 Rounds):")
        for record in result["rollout_3rounds"]:
            print(
                f"  Round {record['round']} (Frames {record['horizon_start']}-{record['horizon_end']}): "
                f"Rel-L2 = {record['rel_l2']*100:.2f}%, "
                f"Cumulative = {record['cumulative_rel_l2']*100:.2f}%"
            )
    print("=" * 80)


def _batch_summary_row(checkpoint_dir: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run": str(checkpoint_dir),
        "checkpoint": result["checkpoint_name"],
        "selection_strategy": result["selection_strategy"],
    }
    for prefix, metrics in (
        ("id", result["id_test_metrics"]),
        ("ood", result["ood_test_metrics"]),
    ):
        for metric in (
            "fluid_fullfield_rel_l2",
            "sensor_rel_l2",
            "unobserved_rel_l2",
            "vorticity_rel_l2",
            "channel_u_rel_l2",
            "channel_v_rel_l2",
            "rmse",
        ):
            row[f"{prefix}_{metric}"] = metrics.get(metric)
        for metric, value in metrics.items():
            if metric.startswith("fa_"):
                row[f"{prefix}_{metric}"] = value
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint or every archived run")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint-dir", type=Path, help="One run directory")
    target.add_argument("--checkpoints-dir", type=Path, help="Root containing archived runs")
    parser.add_argument("--data-root", type=Path, default=Path("data/foil"))
    parser.add_argument("--output-prefix", type=str, default="evaluation_results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--selector", type=str, default=None, help="Checkpoint selector name")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Evaluating on device: {device}")

    if args.checkpoint_dir is not None:
        result = evaluate_checkpoint(
            args.checkpoint_dir, args.data_root, device, selector_preference=args.selector
        )
        if not result:
            raise SystemExit(f"No evaluable checkpoint found in {args.checkpoint_dir}")
        _print_result(args.checkpoint_dir, result)
        return

    run_dirs = sorted(
        {config_path.parent for config_path in args.checkpoints_dir.rglob("config.yaml")},
        key=lambda path: str(path),
    )
    all_results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for checkpoint_dir in tqdm(run_dirs, desc="Archived runs"):
        result = evaluate_checkpoint(
            checkpoint_dir, args.data_root, device, selector_preference=args.selector
        )
        if result:
            _print_result(checkpoint_dir, result)
            all_results.append({"run": str(checkpoint_dir), **result})
            rows.append(_batch_summary_row(checkpoint_dir, result))
    if not all_results:
        raise SystemExit(f"No evaluable checkpoints found below {args.checkpoints_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = Path(f"{args.output_prefix}_{timestamp}.xlsx")
    json_path = Path(f"{args.output_prefix}_{timestamp}.json")
    pd.DataFrame(rows).to_excel(excel_path, sheet_name="ID_OOD_Summary", index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved summary Excel to: {excel_path}")
    print(f"Saved raw JSON to: {json_path}")


if __name__ == "__main__":
    main()
