param (
    [string]$RealDir = "data/foil/hf_dataset/real",
    [string]$SimDir = "data/foil/hf_dataset/sim",
    [string]$OutDir = "data/foil/tensor_cache_64x128",
    [string]$ManifestDir = "manifests"
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
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 2: Build local smoke-test manifests. The fallback split is explicit because
# the workstation contains only a tiny subset of the official 99+99 trajectories.
python scripts/build_provenance_manifests.py `
    --data-root "$(Split-Path -Parent $OutDir)" `
    --manifest-dir "$ManifestDir" `
    --metadata-dir "$(Split-Path -Parent (Split-Path -Parent $RealDir))" `
    --allow-fallback-splits
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 3: Compute the development basis from train-time blocks.
if (-not (Test-Path "artifacts")) {
    New-Item -ItemType Directory -Path "artifacts" | Out-Null
}
python scripts/compute_sim_pod_basis.py `
    --tensor-dir "$OutDir/numerical" `
    --manifest "$ManifestDir/sim_source_manifest.json" `
    --stage dev `
    --output-file "artifacts/pod_basis_dev_64x128.pt" `
    --rank 64 `
    --perp-rank 16
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 4: Compute the final basis from every available Sim frame.
python scripts/compute_sim_pod_basis.py `
    --tensor-dir "$OutDir/numerical" `
    --manifest "$ManifestDir/sim_source_manifest.json" `
    --stage final `
    --output-file "artifacts/pod_basis_final_64x128.pt" `
    --rank 64 `
    --perp-rank 16
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Step 5: Freeze basis-dependent Group-QDEIM sensor placements.
python scripts/build_provenance_manifests.py `
    --manifest-dir "$ManifestDir" `
    --sensor-only `
    --pod-basis "artifacts/pod_basis_final_64x128.pt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Data preprocessing and POD basis extraction completed successfully!" -ForegroundColor Green
Write-Host "Tensors saved in:   $OutDir"
Write-Host "Development POD:    artifacts/pod_basis_dev_64x128.pt"
Write-Host "Final POD:          artifacts/pod_basis_final_64x128.pt"
Write-Host "Manifests saved in: $ManifestDir"
Write-Host "==============================================================================" -ForegroundColor Cyan
