@echo off
echo ============================================================
echo   Game AI Control System - GUI Mode
echo ============================================================
echo.

set PYTHON=python
set DIR=%~dp0

cd /d "%DIR%"

echo Starting GUI...
echo.

"%PYTHON%" main.py --gui --fps 10

pause
