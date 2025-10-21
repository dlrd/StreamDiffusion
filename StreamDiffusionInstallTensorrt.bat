call .\.venv\Scripts\activate

pip install torch-tensorrt

pip uninstall -y cuda-python cuda-bindings cuda-core
pip install cuda-python==12.8 cuda-core

pip install --force-reinstall pywin32
