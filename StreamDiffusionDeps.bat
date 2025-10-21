@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ==========================================================
echo   Smode StreamDiffusion Dependency Installer (CUDA Auto)
echo ==========================================================
echo.

if not exist .venv (
    echo Installing pip...
    call "%CD%\..\python-3_11_9\python.exe" "%CD%\..\get-pip.py"
    echo Creating venv environment...
    call "%CD%\..\python-3_11_9\python.exe" -m pip install virtualenv
    call "%CD%\..\python-3_11_9\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

call .\.venv\Scripts\activate

echo Detecting NVIDIA GPU...
set CUDA_AVAILABLE=0
where nvidia-smi >nul 2>nul && set CUDA_AVAILABLE=1

if %CUDA_AVAILABLE%==1 (
    echo GPU detected, using CUDA 12.8 builds.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
) else (
    echo No GPU detected, using CPU-only builds.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
)

echo.
python -m pip install --upgrade pip setuptools wheel

echo Removing old PyTorch / TorchVision / xFormers if any...
pip uninstall -y torch torchvision xformers

echo Installing PyTorch (2.8.0) + TorchVision (0.23.0) + xFormers
pip install torch==2.8.0 torchvision==0.23.0 xformers --index-url %TORCH_INDEX_URL%

echo Installing project requirements...
pip install -r requirements.txt
pip install cuda-python

echo.
echo All dependencies installed successfully.
