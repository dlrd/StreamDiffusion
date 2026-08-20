@echo off
REM ============================================================================
REM StreamDiffusion-R15 - Automatic Installation
REM ============================================================================
REM This script automatically installs all required dependencies
REM for StreamDiffusion-R15 with ControlNet on Windows.
REM
REM Prerequisites:
REM   - Smode Compose installed (provides Python 3.11.9 in the parent folder)
REM   - NVIDIA GPU (RTX 2000/3000/4000/5000 series) with 8+ GB VRAM
REM   - CUDA 12.9+ installed (recent NVIDIA drivers)
REM   - Internet connection (15-20 GB of downloads required)
REM
REM Recommended GPUs:
REM   - RTX 3060/3070/3080 (12 GB): SD 1.5 + ControlNet
REM   - RTX 4070/4080/4090 (16+ GB): SD 1.5 + SDXL + ControlNet
REM   - RTX 5080/5090 (16+ GB): SD 1.5 + SDXL + ControlNet + StreamV2V
REM ============================================================================

setlocal enabledelayedexpansion
color 0A

REM Status marker polled by Smode Engine to drive the install status icon.
REM See :write_status at the end of this file for the JSON schema.
set "STATUS_FILE=%~dp0install_status.json"
call :write_status installing 0 4 "Starting installation"

echo.
echo ============================================================================
echo  StreamDiffusion-R15 - Automatic Installation
echo ============================================================================
echo.

REM ============================================================================
REM Step 0: Preliminary checks
REM ============================================================================

echo [Step 0/4] Checking prerequisite...
echo.

REM Use Python 3.11.9 provided by Smode (same as StartStreamDiffusion.bat)
set "PYTHON_EXE=%CD%\..\python-3_11_9\python.exe"

if not exist "%PYTHON_EXE%" (
    color 0C
    call :write_status failed 0 4 "Python 3.11.9 not found"
    echo [ERROR] Python 3.11.9 not found:
    echo    %PYTHON_EXE%
    echo.
    echo Make sure Smode Compose / Live is installed and this package
    echo is located in the Packages/ folder of Smode.
    echo.
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found (Smode).

REM Verify that CUDA Toolkit is installed (required for torch.compile / Triton)
set "CUDA_FOUND=0"
REM Verify CUDA_PATH (variable defined by the CUDA installer)
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_FOUND=1"
)
REM Fallback: search for CUDA 12.9 (installed by winget) in the standard path
if "%CUDA_FOUND%"=="0" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_FOUND=1"
)
if "%CUDA_FOUND%"=="0" (
    echo [INFO] CUDA Toolkit not found. Installation via winget...
    winget install Nvidia.CUDA --version 12.9 --accept-source-agreements --accept-package-agreements --silent
    if !errorlevel! equ 0 (
        echo [OK] CUDA Toolkit 12.9.1 installed successfully.
        echo [INFO] Note: a restart may be necessary for CUDA_PATH to be active.
    ) else (
        color 0E
        echo [WARNING] Automatic installation of CUDA Toolkit failed.
        echo    Please install it manually: https://developer.nvidia.com/cuda-toolkit-archive
        echo    The installation will continue, but torch.compile^(^) will not work.
        echo.
    )
) else (
    echo [OK] CUDA Toolkit found.
)
echo [INFO] Verification GPU detailed report will be generated at step 2 (PyTorch CUDA check).

call :write_status installing 1 4 "Creating Python virtual environment"

echo.
echo ============================================================================
echo [Step 1/4] Creation of the Python virtual environment
echo ============================================================================
echo.

REM Verify if .venv already exists and is functional
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Existing virtual environment found and functional.
    echo [INFO] Using existing virtual environment.
    goto :skip_venv_creation
)
if exist ".venv" (
    echo [INFO] Corrupted virtual environment detected, removal...
    rmdir /s /q .venv
    echo [OK] Old virtual environment removed.
)

echo [INFO] Installing virtualenv...
"%PYTHON_EXE%" -m pip install virtualenv --quiet
echo [INFO] Creation of the virtual environment in .venv...
"%PYTHON_EXE%" -m virtualenv --copies .venv
if %errorlevel% neq 0 (
    color 0C
    call :write_status failed 1 4 "Virtual environment creation failed"
    echo [ERROR] Impossible to create the virtual environment.
    echo.
    exit /b 1
)
echo [OK] Virtual environment created successfully (Python %PYTHON_VERSION%).

:skip_venv_creation

echo.
echo [INFO] Activation of the virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    color 0C
    call :write_status failed 1 4 "Virtual environment activation failed"
    echo [ERROR] Impossible to activate the virtual environment.
    echo.
    exit /b 1
)
echo [OK] Virtual environment activated.

REM Update pip (after activation, python = venv Python 3.11.9)
echo.
echo [INFO] Updating pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip updated.

call :write_status installing 2 4 "Installing dependencies from requirements.txt"

echo.
echo ============================================================================
echo [Step 2/4] Installation of dependencies (requirements.txt)
echo ============================================================================
echo.

if not exist "requirements.txt" (
    color 0C
    call :write_status failed 2 4 "requirements.txt not found"
    echo [ERROR] The requirements.txt file is not found.
    echo Make sure you are in the correct directory.
    echo.
    exit /b 1
)

echo [INFO] Installation of all dependencies from requirements.txt...
echo [INFO] This step may take 15-20 minutes depending on your connection...
echo.
echo Main packages that will be installed:
echo   - PyTorch 2.10.0 stable + CUDA 12.8 (runtime PyTorch)
echo   - TensorRT 10.9.0.34
echo   - diffusers 0.36.0 (Hugging Face)
echo   - transformers 4.57.1
echo   - controlnet-aux 0.0.10 (Canny, Depth, OpenPose)
echo   - easy-dwpose 1.0.2 (pose detection)
echo   - triton-windows 3.3.1 (torch.compile)
echo.
echo [INFO] Installation will start. Keep this terminal open...
echo.

python -m pip install -r requirements.txt --verbose

if %errorlevel% neq 0 (
    color 0C
    call :write_status failed 2 4 "Dependency installation failed - pip install -r requirements.txt"
    echo.
    echo [ERROR] Dependencies installation failed.
    echo.
    exit /b 1
)

echo.
echo [OK] All dependencies installed successfully.

REM Install easy-dwpose separately (artificial conflict with huggingface_hub<0.25, API unchanged)
echo.
echo [INFO] Installation of easy-dwpose (--no-deps)...
python -m pip install easy-dwpose==1.0.2 --no-deps --quiet
if %errorlevel% neq 0 (
    color 0E
    echo [WARNING] easy-dwpose installation failed. Pose detection may not work.
    echo [INFO] Installation will continue...
) else (
    echo [OK] easy-dwpose installed.
)

REM insightface (IP-Adapter FaceID) intentionally not installed: feature is
REM unused, and one of its transitive dependencies (stringzilla) needs a
REM source build that fails if the Python NuGet headers aren't in place yet.
REM The FaceID code (ip_adapter_processor.py) stays in place but inert: it
REM degrades gracefully (try/except) if insightface is missing. To re-enable
REM it, reinstall from
REM https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl

REM Verify that PyTorch and CUDA are working
echo.
echo [INFO] Verification of PyTorch and CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [WARNING] PyTorch or CUDA does not seem to be working correctly.
    echo Verify your CUDA installation and NVIDIA drivers.
    echo [INFO] Installation will continue...
) else (
    echo [OK] PyTorch and CUDA are working correctly.
)

call :write_status installing 3 4 "Configuring CUDA binaries and Python headers"

echo.
echo ============================================================================
echo [Step 3/4] Configuration of CUDA binaries and Python headers
echo ============================================================================
echo.

echo [INFO] Configuration for torch.compile() and Triton...
echo [INFO] This step copies the necessary CUDA tools and Python headers.
echo.

python setup_venv.py

if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [WARNING] The configuration of the binaries has failed partially.
    echo StreamDiffusion will work anyway, but torch.compile^(^) might not work.
    echo [INFO] Installation will continue...
) else (
    echo [OK] CUDA binaries and Python headers configured.
)

call :write_status installing 4 4 "Verifying installation"

echo.
echo ============================================================================
echo [Step 4/4] Verification of the installation
echo ============================================================================
echo.

echo [INFO] Test of the installation...
echo.

python verify_install.py

if %errorlevel% neq 0 (
    color 0C
    call :write_status failed 4 4 "Installation verification failed"
    echo.
    echo [ERROR] The installation test has failed.
    echo Verify the error messages above.
    echo.
    exit /b 1
)

call :write_status success 4 4 "Installation completed successfully"

echo.
echo ============================================================================
echo  Installation completed successfully!
echo ============================================================================
echo.
echo Next steps:
echo.
echo 1. The models will be downloaded automatically on the first launch:
echo    - Stable Diffusion 1.5 / SDXL (HuggingFace cache)
echo    - DWPose (dw-ll_ucoco_384.onnx, yolox_l.onnx — for OpenPose)
echo.
echo 2. To use in Smode:
echo    - Load the StreamDiffusion-R15 package in Smode
echo    - The SmodeStreamDiffusion.py script will load automatically
echo.
echo 3. Documentation: see the docs/ folder for usage instructions and examples.
echo.
echo Disk space used: ~12-15 GB total
echo   - Environment .venv: ~3 GB
echo   - Models HuggingFace (cache): ~5-8 GB
echo   - Checkpoints DWPose (OpenPose): ~400 MB (on first launch)
echo.
echo ============================================================================

color 0A
echo.

endlocal
exit /b 0

REM ============================================================================
REM Helper: write install_status.json for Smode Engine to poll.
REM   %1 = status  (installing / success / failed)
REM   %2 = step    (current step index)
REM   %3 = totalSteps
REM   %4 = message (human-readable, must not contain " or parentheses)
REM ============================================================================
:write_status
> "%STATUS_FILE%" (
    echo {
    echo   "status": "%~1",
    echo   "step": %~2,
    echo   "totalSteps": %~3,
    echo   "message": "%~4"
    echo }
)
exit /b 0
