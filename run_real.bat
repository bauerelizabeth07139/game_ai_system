@echo off
echo ============================================================
echo   Game AI Control System - Real Input Mode
echo ============================================================
echo.

set PYTHON=python
set DIR=%~dp0

cd /d "%DIR%"

echo [WARNING] Real mouse/keyboard control is ENABLED!
echo   - Mouse and keyboard will be controlled by AI
echo   - Press Ctrl+C to stop immediately
echo.

"%PYTHON%" main.py --fps 10 --real --log-level INFO

pause
