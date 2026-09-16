param (
    [string]$CkptDir = "best_checkpoints",
    [string]$DataRoot = "data/foil",
    [int]$Gpu = 0
)

$env:PYTHONPATH = "src;$env:PYTHONPATH"
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "SparsePOD-Sim2Real: Evaluating All Checkpoints in $CkptDir" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan

python -m sparse_pod_sim2real.training.evaluate `
    --checkpoints-dir "$CkptDir" `
    --data-root "$DataRoot" `
    --gpu $Gpu
