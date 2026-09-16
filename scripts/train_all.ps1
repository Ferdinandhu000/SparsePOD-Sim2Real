param (
    [string]$ConfigDir = "configs/our_models",
    [int]$Gpu = 0
)

$env:PYTHONPATH = "src;$env:PYTHONPATH"
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "SparsePOD-Sim2Real: Starting Batch Training (PowerShell)" -ForegroundColor Green
Write-Host "Config Directory: $ConfigDir"
Write-Host "GPU Device:       $Gpu"
Write-Host "==============================================================================" -ForegroundColor Cyan

python -m sparse_pod_sim2real.training.trainer --config-dir "$ConfigDir" --gpu $Gpu

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Batch training completed! Models and logs archived in best_checkpoints/" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
