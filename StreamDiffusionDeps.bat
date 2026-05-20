@echo off
REM ============================================================================
REM StreamDiffusion - Python Dependencies Installation
REM ============================================================================
REM Called by Smode with the CUDA runtime version as %1.
REM Installs all Python dependencies into .venv using Smode's bundled Python.
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
color 0A

set "CUDA_RUNTIME_VERSION=%1"

echo.
echo ============================================================================
echo  StreamDiffusion - Installing Dependencies
echo ============================================================================
echo.

REM ============================================================================
REM Step 0: Locate Python and check CUDA Toolkit
REM ============================================================================

echo [Step 0/3] Checking prerequisites...
echo.

set "PYTHON_EXE=%CD%\..\..\python\python.exe"
set "PATH=%CD%\..\..;%PATH%"

if not exist "%PYTHON_EXE%" (
    color 0C
    echo [ERROR] Smode Python not found:
    echo    %PYTHON_EXE%
    echo.
    echo Make sure Smode Compose is installed and this package is in Packages/.
    echo.
    exit /b 1
)

for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% detected (Smode).

REM Check CUDA Toolkit (required for torch.compile / Triton)
set "CUDA_FOUND=0"
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_FOUND=1"
)
if "%CUDA_FOUND%"=="0" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_FOUND=1"
)
if "%CUDA_FOUND%"=="0" (
    echo [INFO] CUDA Toolkit not detected. Installing via winget...
    winget install Nvidia.CUDA --version 12.9 --accept-source-agreements --accept-package-agreements --silent
    if !errorlevel! equ 0 (
        echo [OK] CUDA Toolkit 12.9 installed. A restart may be required for CUDA_PATH.
    ) else (
        color 0E
        echo [WARNING] Automatic CUDA Toolkit install failed.
        echo    Install manually: https://developer.nvidia.com/cuda-toolkit-archive
        echo    torch.compile() may not work without it.
        echo.
    )
) else (
    echo [OK] CUDA Toolkit detected.
)

echo.
echo ============================================================================
echo [Step 1/3] Creating Python virtual environment
echo ============================================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment already exists, reusing it.
    goto :skip_venv_creation
)
if exist ".venv" (
    echo [INFO] Corrupted virtual environment detected, removing...
    rmdir /s /q .venv
    echo [OK] Old environment removed.
)

echo [INFO] Installing virtualenv...
"%PYTHON_EXE%" -m pip install virtualenv --quiet
echo [INFO] Creating virtual environment...
"%PYTHON_EXE%" -m virtualenv --copies .venv
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to create virtual environment.
    echo.
    exit /b 1
)
echo [OK] Virtual environment created (Python %PYTHON_VERSION%).

:skip_venv_creation

echo.
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to activate virtual environment.
    echo.
    exit /b 1
)
echo [OK] Virtual environment activated.

echo.
echo [INFO] Updating pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip updated.

echo.
echo ============================================================================
echo [Step 2/3] Installing dependencies (requirements.txt)
echo ============================================================================
echo.

if not exist "requirements.txt" (
    color 0C
    echo [ERROR] requirements.txt not found.
    echo.
    exit /b 1
)

echo [INFO] Installing packages from requirements.txt (15-20 min)...
echo.
python -m pip install -r requirements.txt --verbose
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Dependency installation failed.
    echo.
    exit /b 1
)
echo.
echo [OK] All dependencies installed.

REM easy-dwpose has an artificial conflict on huggingface_hub version; install --no-deps
echo.
echo [INFO] Installing easy-dwpose (--no-deps)...
python -m pip install easy-dwpose==1.0.2 --no-deps --quiet
echo [OK] easy-dwpose installed.

REM Sanity-check PyTorch + CUDA
echo.
echo [INFO] Checking PyTorch + CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if %errorlevel% neq 0 (
    color 0E
    echo [WARNING] PyTorch/CUDA check failed. Verify drivers and CUDA installation.
) else (
    echo [OK] PyTorch and CUDA are working.
)

echo.
echo ============================================================================
echo [Step 3/3] Configuring CUDA binaries and Python headers
echo ============================================================================
echo.

echo [INFO] Running setup_venv.py (copies CUDA tools + Python headers for Triton)...
python setup_venv.py
if %errorlevel% neq 0 (
    color 0E
    echo [WARNING] setup_venv.py failed partially. torch.compile() may not work.
) else (
    echo [OK] CUDA binaries and Python headers configured.
)

echo.
echo ============================================================================
echo  Dependencies installed successfully.
echo ============================================================================
echo.

color 0A
endlocal
exit /b 0
