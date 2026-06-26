@echo off
echo ============================================================
echo   Game AI Control System - Test
echo ============================================================
echo.

set PYTHON=python
set DIR=%~dp0

cd /d "%DIR%"

echo Running all tests...
echo.

"%PYTHON%" test_all.py

echo.
pause
