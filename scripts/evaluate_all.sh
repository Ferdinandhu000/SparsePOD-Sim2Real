#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# One-click evaluation script across all saved models in best_checkpoints/
# Generates comprehensive Summary Excel table and JSON metrics.
# ==============================================================================

CKPT_DIR="${1:-best_checkpoints}"
DATA_ROOT="${2:-data/foil}"
GPU="${3:-0}"

echo "=============================================================================="
echo "SparsePOD-Sim2Real: Evaluating All Checkpoints in ${CKPT_DIR}"
echo "=============================================================================="

export PYTHONPATH="src:${PYTHONPATH:-}"
python -m sparse_pod_sim2real.training.evaluate \
    --checkpoints-dir "${CKPT_DIR}" \
    --data-root "${DATA_ROOT}" \
    --gpu "${GPU}"
