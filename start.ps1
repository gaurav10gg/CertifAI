<#
    start.ps1 -- launch the CertifAI backend and frontend together (Windows).

    Each server runs in its own PowerShell window so their logs stay separate and
    either can be restarted independently. Stopping this script does not stop
    them; close the windows instead.

    Usage:  .\start.ps1
#>

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$venvPython = Join-Path $root 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error @"
No virtual environment found at backend\.venv.

    cd backend
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
"@
}

# The API refuses to predict without trained artifacts, so check before starting
# rather than letting the frontend surface a 503.
$artifact = Join-Path $root 'backend\models\certifai_models.joblib'
if (-not (Test-Path $artifact)) {
    Write-Warning "No trained model at backend\models\certifai_models.joblib."
    Write-Warning "Run 'python train_model.py' in backend\ first (about 3 minutes)."
}

if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Error "Frontend dependencies missing. Run 'npm install' in frontend\."
}

Write-Host 'Starting backend on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "Set-Location '$root\backend'; & '$venvPython' -m uvicorn app:app --port 8000 --reload"
)

Write-Host 'Starting frontend on http://localhost:5173 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host ''
Write-Host 'EMC Advisor is starting. Open http://localhost:5173' -ForegroundColor Green
