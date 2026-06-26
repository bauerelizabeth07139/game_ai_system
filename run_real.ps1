# Game AI Control System - Real Input Mode
Write-Host "============================================================" -ForegroundColor Red
Write-Host "  Game AI Control System - Real Input Mode" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Red
Write-Host ""

$PYTHON = "python"
Set-Location $PSScriptRoot

Write-Host "[WARNING] Real mouse/keyboard control is ENABLED!" -ForegroundColor Yellow
Write-Host "  - Mouse and keyboard will be controlled by AI"
Write-Host "  - Press Ctrl+C to stop immediately"
Write-Host ""

& $PYTHON main.py --fps 10 --real --log-level INFO
