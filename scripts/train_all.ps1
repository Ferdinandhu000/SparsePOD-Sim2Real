param (
    [string]$ConfigDir = "configs/yaml_main_v1",
    [int]$Gpu = 0,
    [string]$DataRoot = ""
)

$env:PYTHONPATH = "src;$env:PYTHONPATH"
$ErrorActionPreference = "Stop"
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "SparsePOD-Sim2Real: Starting Batch Training (PowerShell)" -ForegroundColor Green
Write-Host "Config Directory: $ConfigDir"
Write-Host "GPU Device:       $Gpu"
if ($DataRoot -ne "") {
    Write-Host "Data Root:        $DataRoot"
    python scripts/run_two_stage.py --config-dir "$ConfigDir" --gpu $Gpu --data-root "$DataRoot" --manifest-dir manifests --dev-basis artifacts/pod_basis_dev_64x128.pt --final-basis artifacts/pod_basis_final_64x128.pt
} else {
    python scripts/run_two_stage.py --config-dir "$ConfigDir" --gpu $Gpu --manifest-dir manifests --dev-basis artifacts/pod_basis_dev_64x128.pt --final-basis artifacts/pod_basis_final_64x128.pt
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Batch training completed! Models and logs archived in best_checkpoints/" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
