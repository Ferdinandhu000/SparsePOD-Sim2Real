param (
    [string]$RealDir = "data/foil/hf_dataset/real",
    [string]$SimDir = "data/foil/hf_dataset/sim",
    [string]$OutDir = "data/tensor_cache_64x128"
)

$ErrorActionPreference = "Stop"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "SparsePOD-Sim2Real: Data Preprocessing Pipeline (PowerShell)" -ForegroundColor Green
Write-Host "Real Source:   $RealDir"
Write-Host "Sim Source:    $SimDir"
Write-Host "Target Output: $OutDir"
Write-Host "==============================================================================" -ForegroundColor Cyan

# Step 1: Preprocess raw Arrow dataset into downsampled 64x128 .pt files
python scripts/preprocess_to_tensors.py `
    --real-dir "$RealDir" `
    --sim-dir "$SimDir" `
    --output-dir "$OutDir" `
    --resolution 64 128

# Step 2: Compute centered POD basis and orthogonal complement from numerical train split
if (-not (Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts" | Out-Null
}
python scripts/compute_sim_pod_basis.py `
    --tensor-dir "$OutDir/numerical" `
    --output-file "artifacts/pod_basis_64x128.pt" `
    --rank 64 `
    --perp-rank 16

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Data preprocessing and POD basis extraction completed successfully!" -ForegroundColor Green
Write-Host "Tensors saved in:   $OutDir"
Write-Host "POD Basis saved in: artifacts/pod_basis_64x128.pt"
Write-Host "==============================================================================" -ForegroundColor Cyan
