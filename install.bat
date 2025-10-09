@echo off
REM Copyright (c) 2025 sylcrala | sylcrala.xyz
REM Licensed under the MIT License - see LICENSE.md for details

REM eBay CrossLister - Windows Installation Wrapper
REM Native Windows batch script that calls the Python installer

title eBay CrossLister v1.0.0 - Windows Installer

echo   eBay CrossLister v1.0.0 - Windows Installer
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo  Python not found. Please install Python 3.8+ first.
        echo    Download from: https://www.python.org/downloads/
        echo    Make sure to check "Add Python to PATH" during installation
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=python3
    )
) else (
    set PYTHON_CMD=python
)

echo  Using Python: %PYTHON_CMD%
echo  Launching cross-platform installer...
echo.

REM Pass all arguments to the Python installer
%PYTHON_CMD% install.py %*

