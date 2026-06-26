# Game AI Control System - Start
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Game AI Control System - Start" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYTHON = "python"
Set-Location $PSScriptRoot

Write-Host "Starting system..." -ForegroundColor Yellow
Write-Host "  - FPS: 10"
Write-Host "  - Mode: Log only (no real input)"
Write-Host "  - Press Ctrl+C to stop"
Write-Host ""

& $PYTHON main.py --fps 10 --log-level INFO
