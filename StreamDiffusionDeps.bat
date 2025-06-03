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

pip.exe install setuptools==57.4.0
pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu%1
pip.exe install -r requirements.txt


REM For TensorRT
pip install --index-url https://pypi.nvidia.com tensorrt==9.0.1.post11.dev4
pip install polygraphy==0.47.1 --index-url https://pypi.ngc.nvidia.com
pip install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com
pip install --force-reinstall pywin32

echo Install done