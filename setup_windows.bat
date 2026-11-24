@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title YT Creator Studio - Setup
echo ============================================================
echo  YT Creator Studio - Windows setup
echo  Folder: %CD%
echo ============================================================
echo.

rem ---- 1. Windows / architecture -------------------------------------------
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
if /I not "%ARCH%"=="AMD64" (
    echo [FAIL] 64-bit Windows is required. Detected: %ARCH%
    goto :fail
)
ver | find "10.0." >nul || echo [WARN] Unrecognised Windows version. This app is tested on Windows 11 64-bit.
echo [OK]   Windows 64-bit

rem ---- 2. Python 3.12 --------------------------------------------------------
set "PYEXE="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if not defined PYEXE (
    for /f "delims=" %%i in ('python -c "import sys;print(sys.executable if sys.version_info[:2]==(3,12) else '')" 2^>nul') do set "PYEXE=%%i"
)
if not defined PYEXE (
    echo [FAIL] Python 3.12 was not found. This release is tested on Python 3.12 only.
    echo        Install it with:  winget install Python.Python.3.12
    goto :fail
)
echo [OK]   Python 3.12: !PYEXE!

rem ---- 3. FFmpeg (final check is done by verify_installation.py) -------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [WARN] ffmpeg is not on PATH in this window. If you just installed it, the verifier will
    echo        still find it via the saved user PATH. Otherwise install:  winget install Gyan.FFmpeg
    echo        or set FFMPEG_BIN_DIR in .env
) else (
    echo [OK]   ffmpeg found on PATH
)

rem ---- 4. Virtual environment -------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating virtual environment .venv ...
    "!PYEXE!" -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat" || goto :fail
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)"
if errorlevel 1 (
    echo [FAIL] The existing .venv was created with a different Python version.
    echo        Delete the .venv folder and run this setup again.
    goto :fail
)
echo [OK]   .venv active

echo.
echo Upgrading pip ...
python -m pip install --upgrade pip || goto :fail
echo.
echo Installing core dependencies (requirements.txt) ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [FAIL] Core dependency installation failed. See the messages above.
    goto :fail
)
echo [OK]   Core dependencies installed

rem ---- 5. NVIDIA (optional, only with consent) --------------------------------
set "GPU_LIBS=0"
python scripts\verify_installation.py --check-nvidia
if !errorlevel! equ 0 (
    echo.
    echo An NVIDIA GPU was detected. GPU transcription needs NVIDIA's CUDA 12 cuBLAS and cuDNN 9
    echo libraries from pip. This is a large download, roughly 1 GB or more.
    echo The CPU path keeps working whether or not you install them.
    choice /C YN /M "Install the NVIDIA CUDA runtime libraries now"
    if !errorlevel! equ 1 (
        python -m pip install -r requirements-gpu.txt
        if !errorlevel! neq 0 (
            echo [WARN] GPU library installation failed. CPU mode remains fully available.
            echo        No new virtual environment is needed: these packages do not replace any CPU package.
        ) else (
            set "GPU_LIBS=1"
        )
    )
) else (
    echo [INFO] No NVIDIA GPU detected: CPU mode.
)

rem ---- 6. Optional features ----------------------------------------------------
echo.
echo Installing optional packages (private YouTube upload) ...
python -m pip install -r requirements-optional.txt
if errorlevel 1 echo [WARN] Optional packages failed to install. Private YouTube upload will be disabled.
echo [INFO] Playwright is not used by this app and is not installed.

rem ---- 7. .env -------------------------------------------------------------------
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [OK]   Created .env from .env.example. Add your Gemini key there if you want cloud analysis.
) else (
    echo [OK]   .env already exists; it was NOT modified.
)

rem ---- 8. Verify -----------------------------------------------------------------
echo.
echo Verifying installation and testing hardware capabilities ...
set "VERIFY_ARGS="
if "!GPU_LIBS!"=="1" (
    echo The CUDA Whisper test downloads the small 'tiny' model (about 75 MB^) from Hugging Face.
    choice /C YN /M "Run the CUDA Whisper test now"
    if !errorlevel! equ 1 set "VERIFY_ARGS=--whisper-smoke"
)
python scripts\verify_installation.py !VERIFY_ARGS!
set "VERIFY_RC=!errorlevel!"

echo.
echo ============================================================
if "!VERIFY_RC!"=="0" (
    echo  Setup complete.
    echo  Next steps:
    echo    1. Optional: edit .env to add GEMINI_API_KEY and/or OLLAMA_MODEL
    echo    2. Put recordings in 01_RAW_INPUT  ^(optionally inside a folder named after the game^)
    echo    3. Double-click launch_studio.bat
) else (
    echo  Setup finished, but required checks FAILED. Read the [FAIL] lines above.
)
echo ============================================================
pause
exit /b !VERIFY_RC!

:fail
echo.
echo Setup stopped. Fix the problem above and run setup_windows.bat again.
pause
exit /b 1
