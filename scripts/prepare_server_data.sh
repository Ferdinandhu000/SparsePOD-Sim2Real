#!/usr/bin/env bash
set -euo pipefail

REAL_DIR="${1:-data/foil/hf_dataset/real}"
SIM_DIR="${2:-data/foil/hf_dataset/sim}"
OUT_DIR="${3:-data/foil/tensor_cache_64x128}"
DATA_ROOT="$(dirname "${OUT_DIR}")"
MANIFEST_DIR="${4:-manifests}"
REAL_PARENT="$(dirname "${REAL_DIR}")"
DEFAULT_METADATA_DIR="$(dirname "${REAL_PARENT}")"
METADATA_DIR="${5:-${DEFAULT_METADATA_DIR}}"

echo "=============================================================================="
echo "SparsePOD-Sim2Real: Server Data Preprocessing Pipeline"
echo "Real Source:   ${REAL_DIR}"
echo "Sim Source:    ${SIM_DIR}"
echo "Target Output: ${OUT_DIR}"
echo "Split Metadata: ${METADATA_DIR}"
echo "=============================================================================="

# Step 1: Preprocess raw Arrow dataset into downsampled 64x128 .pt files
python scripts/preprocess_to_tensors.py \
    --real-dir "${REAL_DIR}" \
    --sim-dir "${SIM_DIR}" \
    --output-dir "${OUT_DIR}" \
    --resolution 64 128

# Step 2: Freeze provenance manifests and official Real splits
python scripts/build_provenance_manifests.py \
    --data-root "${DATA_ROOT}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --metadata-dir "${METADATA_DIR}" \
    --expected-sim-count 99 \
    --expected-real-count 99

# Step 3: Compute development POD from Sim train-time blocks
mkdir -p artifacts
python scripts/compute_sim_pod_basis.py \
    --tensor-dir "${OUT_DIR}/numerical" \
    --manifest "${MANIFEST_DIR}/sim_source_manifest.json" \
    --stage dev \
    --output-file "artifacts/pod_basis_dev_64x128.pt" \
    --rank 64 \
    --perp-rank 16

# Step 4: Compute final POD from balanced snapshots spanning every trajectory's
# complete temporal range. The final source learner itself consumes all frames.
python scripts/compute_sim_pod_basis.py \
    --tensor-dir "${OUT_DIR}/numerical" \
    --manifest "${MANIFEST_DIR}/sim_source_manifest.json" \
    --stage final \
    --output-file "artifacts/pod_basis_final_64x128.pt" \
    --rank 64 \
    --perp-rank 16

# Step 5: Generate basis-dependent Group-QDEIM sensor manifests without rehashing data
python scripts/build_provenance_manifests.py \
    --manifest-dir "${MANIFEST_DIR}" \
    --sensor-only \
    --pod-basis "artifacts/pod_basis_final_64x128.pt"

echo "=============================================================================="
echo "Data preprocessing and POD basis extraction completed successfully!"
echo "Tensors saved in:   ${OUT_DIR}"
echo "Development POD:    artifacts/pod_basis_dev_64x128.pt"
echo "Final POD:          artifacts/pod_basis_final_64x128.pt"
echo "Manifests saved in: ${MANIFEST_DIR}"
echo "=============================================================================="
