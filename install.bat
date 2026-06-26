@echo off
echo ============================================================
echo   Game AI Control System - Install
echo ============================================================
echo.

set PYTHON=python

echo [1/3] Checking Python...
"%PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] Python not found: %PYTHON%
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
"%PYTHON%" -m pip install opencv-python numpy torch pyyaml requests pynput --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo      Done
echo.

echo [3/3] Verifying...
"%PYTHON%" -c "import cv2, numpy, torch, yaml, requests, pynput; print('      All imports OK')"
echo.

echo ============================================================
echo   Install complete!
echo   Run: double-click run.bat or run_gui.bat
echo ============================================================
pause
