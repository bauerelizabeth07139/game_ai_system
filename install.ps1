# Game AI Control System - Install
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Game AI Control System - Install" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYTHON = "python"

Write-Host "[1/3] Checking Python..." -ForegroundColor Yellow
try {
    & $PYTHON --version
} catch {
    Write-Host "[ERROR] Python not found: $PYTHON" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
& $PYTHON -m pip install opencv-python numpy torch pyyaml requests pynput --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Done" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Verifying..." -ForegroundColor Yellow
& $PYTHON -c "import cv2, numpy, torch, yaml, requests, pynput; print('      All imports OK')"
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Install complete!" -ForegroundColor Green
Write-Host "  Run: .\run.ps1 or .\run_gui.ps1" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
