#!/usr/bin/env bash
set -e

REAL_DIR="${1:-data/foil/hf_dataset/real}"
SIM_DIR="${2:-data/foil/hf_dataset/sim}"
OUT_DIR="${3:-data/tensor_cache_64x128}"

echo "=============================================================================="
echo "SparsePOD-Sim2Real: Server Data Preprocessing Pipeline"
echo "Real Source:   ${REAL_DIR}"
echo "Sim Source:    ${SIM_DIR}"
echo "Target Output: ${OUT_DIR}"
echo "=============================================================================="

# Step 1: Preprocess raw Arrow dataset into downsampled 64x128 .pt files
python scripts/preprocess_to_tensors.py \
    --real-dir "${REAL_DIR}" \
    --sim-dir "${SIM_DIR}" \
    --output-dir "${OUT_DIR}" \
    --resolution 64 128

# Step 2: Compute centered POD basis and orthogonal complement from numerical train split
mkdir -p artifacts
python scripts/compute_sim_pod_basis.py \
    --tensor-dir "${OUT_DIR}/numerical" \
    --output-file "artifacts/pod_basis_64x128.pt" \
    --rank 64 \
    --perp-rank 16

echo "=============================================================================="
echo "Data preprocessing and POD basis extraction completed successfully!"
echo "Tensors saved in:   ${OUT_DIR}"
echo "POD Basis saved in: artifacts/pod_basis_64x128.pt"
echo "=============================================================================="
