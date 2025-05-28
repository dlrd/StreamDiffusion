REM Create a virtual environment
if not exist .venv (
    echo Creating virtual environment...
    call "%CD%\..\python-3_10_11\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

call .\.venv\Scripts\activate

echo Starting StreamDiffusion with args %1 %2 %3 %4 %5 %6
call python.exe SmodeStreamDiffusion.py --uuid %1 --port %2 --width %3 --height %4 --device %5 --model %6