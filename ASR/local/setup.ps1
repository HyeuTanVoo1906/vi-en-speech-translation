$ErrorActionPreference = 'Stop'
$asrRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workspaceRoot = Split-Path $asrRoot -Parent
$venvPython = Join-Path $workspaceRoot '.venv-asr\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    # Reuse the installed CUDA torch wheel; pinned ASR packages live in this venv.
    python -m venv --system-site-packages (Join-Path $workspaceRoot '.venv-asr')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create venv' }
}
& $venvPython -m pip install -r (Join-Path $asrRoot 'requirements.txt') 'psutil>=7,<8'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $venvPython -m pip check
Write-Host 'ASR venv reuses global CUDA torch. pip check may report unrelated inherited packages.'
& $venvPython (Join-Path $asrRoot 'scripts\local_run.py') check
if ($LASTEXITCODE -ne 0) { throw 'ASR preflight failed' }
