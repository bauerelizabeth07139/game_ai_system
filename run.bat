@echo off
echo ============================================================
echo   Game AI Control System - Start
echo ============================================================
echo.

set PYTHON=python
set DIR=%~dp0

cd /d "%DIR%"

echo Starting system...
echo   - FPS: 10
echo   - Mode: Log only (no real input)
echo   - Press Ctrl+C to stop
echo.

"%PYTHON%" main.py --fps 10 --log-level INFO

pause
