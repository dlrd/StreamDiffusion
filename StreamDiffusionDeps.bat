@echo off

if not exist .venv (
    echo Installing pip...
    call "%CD%\..\python-3_11_9\python.exe" "%CD%\..\get-pip.py"
    echo Creating venv environment...
    call "%CD%\..\python-3_11_9\python.exe" -m pip install virtualenv
    call "%CD%\..\python-3_11_9\python.exe" -m virtualenv --copies .venv
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
call .\.venv\Scripts\activate

REM Install required packages

echo Installing requirements...

pip.exe install setuptools wheel
pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip.exe install -r requirements.txt
pip.exe install cuda-python


REM For TensorRT
pip install torch-tensorrt==2.7.0
pip install --force-reinstall pywin32

echo Install done