#!/usr/bin/env bash
# ==============================================================================
# One-click batch training script for SparsePOD-Sim2Real
# Usage:
#   bash scripts/train_all.sh                    # Train all configs in configs/our_models
#   bash scripts/train_all.sh configs/baselines/ # Train specific config directory
#   bash scripts/train_all.sh --gpu 1            # Run on specific GPU
# ==============================================================================

CONFIG_DIR="${1:-configs/our_models}"
GPU="${2:-0}"

echo "=============================================================================="
echo "SparsePOD-Sim2Real: Starting Batch Training"
echo "Config Directory: ${CONFIG_DIR}"
echo "GPU Device:       ${GPU}"
echo "=============================================================================="

export PYTHONPATH="src:$PYTHONPATH"
python -m sparse_pod_sim2real.training.trainer --config-dir "${CONFIG_DIR}" --gpu "${GPU}"

echo "=============================================================================="
echo "Batch training completed! Models and logs archived in best_checkpoints/"
echo "=============================================================================="
