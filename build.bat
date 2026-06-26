@echo off
echo ============================================================
echo   Game AI Control System - Build EXE
echo ============================================================
echo.

set PYTHON=python
set PYINSTALLER=pyinstaller
set DIR=%~dp0

cd /d "%DIR%"

echo [1/2] Building... (first build may take several minutes)
echo.

"%PYINSTALLER%" ^
    --noconfirm ^
    --onedir ^
    --name GameAI ^
    --add-data "config.yaml;." ^
    --add-data "src;src" ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import yaml ^
    --hidden-import requests ^
    --hidden-import torch ^
    --hidden-import pynput ^
    --hidden-import pynput.mouse ^
    --hidden-import pynput.keyboard ^
    --collect-submodules pynput ^
    main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo [2/2] Copying config...
copy /Y config.yaml dist\GameAI\config.yaml >nul
copy /Y config.yaml dist\GameAI\_internal\config.yaml >nul

echo.
echo ============================================================
echo   Build complete!
echo   Output: dist\GameAI\GameAI.exe
echo.
echo   Usage:
echo     cd dist\GameAI
echo     GameAI.exe --fps 10 --frames 30
echo     GameAI.exe --gui
echo     GameAI.exe --real
echo ============================================================
pause
