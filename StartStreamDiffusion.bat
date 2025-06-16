REM Create a virtual environment
if not exist .venv (
    echo Creating virtual environment...
    call "%CD%\..\python-3_11_9\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

call .\.venv\Scripts\activate

set argc=0

for %%A in (%*) do (
    set /a argc+=1
)


if "%~7"=="" (
    echo No HF_TOKEN provided, using default.
) else (
    set HF_TOKEN=%7
)

if !count! EQU 7 (
    set HF_TOKEN=%7
)
echo Starting StreamDiffusion with args %1 %2 %3 %4 %5 %6
call python.exe SmodeStreamDiffusion.py --uuid %1 --port %2 --width %3 --height %4 --device %5 --model %6