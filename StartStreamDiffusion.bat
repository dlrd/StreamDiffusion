@echo off
cd /d "%~dp0"

REM Add Smode SDK root to PATH so python312.dll is found
set "PATH=%CD%\..\..;%PATH%"

REM Add CUDA to PATH BEFORE venv activation so Triton can find tools
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
) else (
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%"
)

REM Create virtual environment if missing
if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    call "..\..\python\python.exe" -m virtualenv --copies .venv
)

call ".venv\Scripts\activate"

echo Starting StreamDiffusion with args %1 %2 %3 %4 %5 %6
call python.exe SmodeStreamDiffusion.py --uuid %1 --port %2 --width %3 --height %4 --device %5 --model %6
