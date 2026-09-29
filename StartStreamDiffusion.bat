REM Add CUDA to PATH BEFORE venv activation to ensure Triton can find tools
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
) else (
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
)

REM Create a virtual environment
if not exist .venv (
    echo Creating virtual environment...
    call "%CD%\..\python-3_11_9\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

call .\.venv\Scripts\activate

echo Starting StreamDiffusion with args %1 %2 %3 %4 %5 %6
call python.exe SmodeStreamDiffusion.py --uuid %1 --port %2 --width %3 --height %4 --device %5 --model %6
