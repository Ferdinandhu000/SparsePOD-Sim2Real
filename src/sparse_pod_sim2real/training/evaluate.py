from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml

from ..data.dataset import SparseTrajectoryDataset
from ..model import load_model
from .metrics import compute_metrics


def autoregressive_rollout(
    model: torch.nn.Module,
    initial_batch: dict,
    dataset: SparseTrajectoryDataset,
    rounds: int = 3,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, float]]:
    """Performs multi-round autoregressive rollout and evaluates prediction metrics per round."""
    results = []
    curr_batch = {k: v.to(device) for k, v in initial_batch.items() if isinstance(v, torch.Tensor)}

    for r in range(1, rounds + 1):
        with torch.no_grad():
            pred = model(curr_batch)  # [B, Tout, H, W, 2]

        metrics = compute_metrics(pred.cpu(), curr_batch["y_full"].cpu())
        metrics["round"] = r
        results.append(metrics)

        # Autoregressive update for next round
        # In sparse scenario, next historical sensor values come from the predicted field
        next_sensors = dataset.sensor_op.extract_sensor_values(pred)
        curr_batch["sensor_values"] = next_sensors
        if "x_sparse" in curr_batch:
            curr_batch["x_sparse"] = dataset.sensor_op.to_sparse_tensor(
                pred, append_mask=True, fill_method=dataset.sparse_fill_method
            )

    return results


def evaluate_checkpoint(checkpoint_dir: Path, data_root: Path, device: torch.device) -> Dict:
    cfg_file = checkpoint_dir / "config.yaml"
    weights_file = checkpoint_dir / "best.pt"
    if not weights_file.exists():
        weights_file = checkpoint_dir / "best_sim.pt"

    if not cfg_file.exists() or not weights_file.exists():
        return {}

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load POD basis if needed by model or Q-DEIM
    pod_path = config.get("pod_basis_path", None)
    pod_basis = None
    if pod_path and Path(pod_path).exists():
        pod_basis = torch.load(pod_path, map_location=device)
    else:
        # Search candidate locations
        candidates = [
            data_root / "pod_basis_64x128.pt",
            data_root / "pod_basis.pt",
            checkpoint_dir / "pod_basis.pt",
            Path("data/foil/pod_basis_64x128.pt"),
        ]
        for cand in candidates:
            if cand.exists():
                pod_basis = torch.load(cand, map_location=device)
                break

    model_name = config.get("model_name", "")
    if pod_basis is None and ("pod" in model_name or config.get("sensor_topology") == "q_deim"):
        raise FileNotFoundError(
            f"Missing POD basis for evaluation of {model_name} in {checkpoint_dir}. "
            f"Please ensure scripts/compute_sim_pod_basis.py has been run."
        )

    # Test dataset
    test_ds = SparseTrajectoryDataset(
        data_root=data_root,
        dataset_type="real",
        mode="test",
        test_mode=config.get("test_mode", "all"),
        topology_type=config.get("sensor_topology", "wall"),
        num_sensors=config.get("num_sensors", 64),
        in_step=config.get("in_step", 20),
        out_step=config.get("out_step", 20),
        interval=config.get("interval", 10),
        pod_basis=pod_basis["basis"] if isinstance(pod_basis, dict) else pod_basis,
        sparse_fill_method=config.get("sparse_fill_method", "zero"),
    )

    sensor_indices = test_ds.sensor_op.indices_1d.to(device)
    model = load_model(config, pod_basis=pod_basis, sensor_indices=sensor_indices).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device), strict=True)
    model.eval()


    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.get("batch_size", 8), shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {checkpoint_dir.name}"):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            p = model(batch)
            preds.append(p.cpu())
            targets.append(batch["y_full"].cpu())

    if not preds:
        return {}

    all_preds = torch.cat(preds, dim=0)
    all_targets = torch.cat(targets, dim=0)
    summary_metrics = compute_metrics(all_preds, all_targets)

    return {
        "model_name": checkpoint_dir.name,
        "config": config,
        "summary": summary_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("best_checkpoints"))
    parser.add_argument("--data-root", type=Path, default=Path("data/foil"))
    parser.add_argument("--output-prefix", type=str, default="evaluation_results")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    all_results = []

    sub_dirs = sorted([d for d in args.checkpoints_dir.iterdir() if d.is_dir()])
    for d in sub_dirs:
        res = evaluate_checkpoint(d, args.data_root, device)
        if res:
            all_results.append(res)

    if not all_results:
        print("No evaluation results generated.")
        return

    # Create summary table
    summary_rows = []
    for r in all_results:
        s = r["summary"]
        summary_rows.append({
            "Model Name": r["model_name"],
            "Overall Rel-L2 (%)": f"{s['rel_l2'] * 100:.2f}%",
            "RMSE": f"{s['rmse']:.5f}",
            "MAE": f"{s['mae']:.5f}",
            "R2 Score": f"{s['r2']:.4f}",
            "Vorticity Rel-L2 (%)": f"{s['vorticity_rel_l2'] * 100:.2f}%",
            "KE Error": f"{s['ke_error']:.6f}",
            "MVPE": f"{s['mvpe']:.5f}",
        })

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80 + "\nEVALUATION SUMMARY RESULTS\n" + "=" * 80)
    print(df_summary.to_string(index=False))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = Path(f"{args.output_prefix}_{timestamp}.xlsx")
    json_path = Path(f"{args.output_prefix}_{timestamp}.json")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary Excel to: {excel_path}")
    print(f"Saved raw JSON to:       {json_path}")


if __name__ == "__main__":
    main()
