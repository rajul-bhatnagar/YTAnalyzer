@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title YT Creator Studio
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\verify_installation.py --quick
if errorlevel 1 (
    echo.
    echo Environment check failed. Fix the [FAIL] items above, or run setup_windows.bat again.
    pause
    exit /b 1
)
echo.
echo Starting YT Creator Studio. Your browser will open; close this window to stop the app.
python -m streamlit run app.py --server.headless false
pause
