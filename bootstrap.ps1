#Requires -Version 5.1
# Recreate the repo venv from requirements.lock.txt and run a smoke test.
# Usage (from repo root or any cwd):  powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Host "Creating venv with py -3.14 ..."
    py -3.14 -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (Test-Path $Python)) {
        Write-Error "Failed to create $Python"
        exit 1
    }
}

$Lock = Join-Path $Root 'requirements.lock.txt'
if (-not (Test-Path $Lock)) {
    Write-Error "Missing $Lock"
    exit 1
}

Write-Host "Installing $Lock ..."
& $Python -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Smoke: import pandas, numpy ..."
& $Python -c "import pandas, numpy; print('pandas', pandas.__version__); print('numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Smoke: Adaptive unittests ..."
& $Python -m unittest discover -s Adaptive -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "bootstrap.ps1: OK"
exit 0