# Game AI Control System - GUI Mode
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Game AI Control System - GUI Mode" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYTHON = "python"
Set-Location $PSScriptRoot

Write-Host "Starting GUI..." -ForegroundColor Yellow
Write-Host ""

& $PYTHON main.py --gui --fps 10
