#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# One-click batch training script for SparsePOD-Sim2Real
# Usage:
#   bash scripts/train_all.sh                              # configs/yaml_main_v1 on GPU 0
#   bash scripts/train_all.sh configs/yaml_ablations 1     # ablations on GPU 1
#   bash scripts/train_all.sh configs/yaml_main_v1 0 /data # explicit data root
# ==============================================================================

CONFIG_DIR="${1:-configs/yaml_main_v1}"
GPU="${2:-0}"
DATA_ROOT="${3:-}"

echo "=============================================================================="
echo "SparsePOD-Sim2Real: Starting Batch Training"
echo "Config Directory: ${CONFIG_DIR}"
echo "GPU Device:       ${GPU}"
if [ -n "${DATA_ROOT}" ]; then
    echo "Data Root:        ${DATA_ROOT}"
fi
echo "=============================================================================="

export PYTHONPATH="src:${PYTHONPATH:-}"
ARGS=(
    --config-dir "${CONFIG_DIR}"
    --gpu "${GPU}"
    --manifest-dir manifests
    --dev-basis artifacts/pod_basis_dev_64x128.pt
    --final-basis artifacts/pod_basis_final_64x128.pt
)
if [ -n "${DATA_ROOT}" ]; then
    ARGS+=(--data-root "${DATA_ROOT}")
fi
python scripts/run_two_stage.py "${ARGS[@]}"

echo "=============================================================================="
echo "Batch training completed! Models and logs archived in best_checkpoints/"
echo "=============================================================================="
