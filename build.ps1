# Game AI Control System - Build EXE
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Game AI Control System - Build EXE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$PYTHON = "python"
$PYINSTALLER = "pyinstaller"
Set-Location $PSScriptRoot

Write-Host "[1/2] Building... (first build may take several minutes)" -ForegroundColor Yellow
Write-Host ""

& $PYINSTALLER `
    --noconfirm `
    --onedir `
    --name GameAI `
    --add-data "config.yaml;." `
    --add-data "src;src" `
    --hidden-import cv2 `
    --hidden-import numpy `
    --hidden-import yaml `
    --hidden-import requests `
    --hidden-import torch `
    --hidden-import pynput `
    --hidden-import pynput.mouse `
    --hidden-import pynput.keyboard `
    --collect-submodules pynput `
    main.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "[2/2] Copying config..." -ForegroundColor Yellow
Copy-Item -Path "config.yaml" -Destination "dist\GameAI\config.yaml" -Force
Copy-Item -Path "config.yaml" -Destination "dist\GameAI\_internal\config.yaml" -Force

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  Output: dist\GameAI\GameAI.exe" -ForegroundColor Green
Write-Host ""
Write-Host "  Usage:" -ForegroundColor Yellow
Write-Host "    cd dist\GameAI"
Write-Host "    .\GameAI.exe --fps 10 --frames 30"
Write-Host "    .\GameAI.exe --gui"
Write-Host "    .\GameAI.exe --real"
Write-Host "============================================================" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
