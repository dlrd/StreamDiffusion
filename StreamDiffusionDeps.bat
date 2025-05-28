@echo off

if not exist .venv (
    echo Installing pip...
    call "%CD%\..\python-3_10_11\python.exe" "%CD%\..\get-pip.py"
    echo Creating venv environment...
    call "%CD%\..\python-3_10_11\python.exe" -m pip install virtualenv
    call "%CD%\..\python-3_10_11\python.exe" -m virtualenv --copies .venv
    call "%CD%\..\python-3_10_11\python.exe" -m pip install --upgrade pip
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Install required packages

echo Installing requirements...

pip.exe install setuptools==57.4.0 wheel
pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu%1
pip.exe install -r requirements.txt
pip.exe install --upgrade tensorrt-cu12

REM Fix potential issues with Windows compatibility
python.exe -m pip install --force-reinstall pywin32

echo Install done