@echo off
REM ============================================================================
REM StreamDiffusion - TensorRT Installation
REM ============================================================================
REM Run this after StreamDiffusionDeps.bat to add TensorRT acceleration support.
REM ============================================================================

setlocal
cd /d "%~dp0"
color 0A

echo.
echo ============================================================================
echo  StreamDiffusion - Installing TensorRT
echo ============================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    color 0C
    echo [ERROR] Virtual environment not found. Run StreamDiffusionDeps.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [INFO] Installing torch-tensorrt...
pip install torch-tensorrt==2.7.0
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] torch-tensorrt installation failed.
    echo.
    pause
    exit /b 1
)
echo [OK] torch-tensorrt installed.

echo.
echo [INFO] Reinstalling pywin32 (required for TensorRT on Windows)...
pip install --force-reinstall pywin32
if %errorlevel% neq 0 (
    color 0E
    echo [WARNING] pywin32 reinstall failed.
) else (
    echo [OK] pywin32 reinstalled.
)

echo.
echo ============================================================================
echo  TensorRT installed successfully.
echo ============================================================================
echo.

color 0A
pause >nul
endlocal
