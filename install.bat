@echo off
REM ============================================================================
REM StreamDiffusion-R13 - Installation Automatique
REM ============================================================================
REM Ce script installe automatiquement toutes les dependances necessaires
REM pour StreamDiffusion-R13 avec ControlNet, YOLO/SAM2 et Wan 2.1 sur Windows.
REM
REM Prerequis:
REM   - Smode Compose installe (fournit Python 3.11.9 dans le dossier parent)
REM   - NVIDIA GPU (RTX 2000/3000/4000/5000) avec 8+ GB VRAM
REM   - CUDA 12.9+ installe (drivers NVIDIA recents)
REM   - Connexion internet (15-20 GB a telecharger)
REM
REM GPU recommandes:
REM   - RTX 3060/3070/3080 (12 GB): SD 1.5 + ControlNet
REM   - RTX 4070/4080/4090 (16+ GB): SD 1.5/SDXL + ControlNet + YOLO/SAM2
REM   - RTX 5080/5090 (16+ GB): Tous les moteurs + Wan 2.1 T2V
REM ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"
color 0A

echo.
echo ============================================================================
echo  StreamDiffusion-R13 - Installation Automatique
echo ============================================================================
echo.

REM ============================================================================
REM Etape 0: Verifications preliminaires
REM ============================================================================

echo [Etape 0/4] Verification des prerequis...
echo.

REM Utiliser le Python fourni par Smode (meme que StartStreamDiffusion.bat)
set "PYTHON_EXE=%CD%\..\..\python\python.exe"
set "PATH=%CD%\..\..;%PATH%"

if not exist "%PYTHON_EXE%" (
    color 0C
    echo [ERREUR] Python 3.11.9 de Smode non trouve:
    echo    %PYTHON_EXE%
    echo.
    echo Assurez-vous que Smode Compose est installe et que ce package
    echo se trouve dans le dossier Packages/ de Smode.
    echo.
    pause
    exit /b 1
)

REM Afficher la version Python
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% detecte (Smode).

REM Verifier que CUDA Toolkit est installe (requis pour torch.compile / Triton)
set "CUDA_FOUND=0"
REM Verifier CUDA_PATH (variable definie par l'installeur CUDA)
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_FOUND=1"
)
REM Fallback: chercher CUDA 12.9 (installe par winget) dans le chemin standard
if "%CUDA_FOUND%"=="0" (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" set "CUDA_FOUND=1"
)
if "%CUDA_FOUND%"=="0" (
    echo [INFO] CUDA Toolkit non detecte. Installation via winget...
    winget install Nvidia.CUDA --version 12.9 --accept-source-agreements --accept-package-agreements --silent
    if !errorlevel! equ 0 (
        echo [OK] CUDA Toolkit 12.9.1 installe avec succes.
        echo [INFO] Note: un redemarrage peut etre necessaire pour que CUDA_PATH soit actif.
    ) else (
        color 0E
        echo [ATTENTION] Installation automatique du CUDA Toolkit echouee.
        echo    Installez-le manuellement: https://developer.nvidia.com/cuda-toolkit-archive
        echo    L'installation va continuer, mais torch.compile^(^) ne fonctionnera pas.
        echo.
    )
) else (
    echo [OK] CUDA Toolkit detecte.
)
echo [INFO] Verification GPU detaillee reportee a l'etape 2 (PyTorch CUDA check).

echo.
echo ============================================================================
echo [Etape 1/4] Creation de l'environnement virtuel Python
echo ============================================================================
echo.

REM Verifier si .venv existe deja et est fonctionnel
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Environnement virtuel existant detecte et fonctionnel.
    echo [INFO] Utilisation de l'environnement existant.
    goto :skip_venv_creation
)
if exist ".venv" (
    echo [INFO] Environnement virtuel corrompu detecte, suppression...
    rmdir /s /q .venv
    echo [OK] Ancien environnement supprime.
)

echo [INFO] Installation de virtualenv...
"%PYTHON_EXE%" -m pip install virtualenv --quiet
echo [INFO] Creation de l'environnement virtuel dans .venv...
"%PYTHON_EXE%" -m virtualenv --copies .venv
if %errorlevel% neq 0 (
    color 0C
    echo [ERREUR] Impossible de creer l'environnement virtuel.
    echo.
    pause
    exit /b 1
)
echo [OK] Environnement virtuel cree avec succes (Python %PYTHON_VERSION%).

:skip_venv_creation

echo.
echo [INFO] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    color 0C
    echo [ERREUR] Impossible d'activer l'environnement virtuel.
    echo.
    pause
    exit /b 1
)
echo [OK] Environnement virtuel active.

REM Mettre a jour pip (apres activation, python = venv Python 3.11.9)
echo.
echo [INFO] Mise a jour de pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip mis a jour.

echo.
echo ============================================================================
echo [Etape 2/4] Installation des dependances (requirements.txt)
echo ============================================================================
echo.

if not exist "requirements.txt" (
    color 0C
    echo [ERREUR] Le fichier requirements.txt est introuvable.
    echo Assurez-vous d'etre dans le bon repertoire.
    echo.
    pause
    exit /b 1
)

echo [INFO] Installation de toutes les dependances depuis requirements.txt...
echo [INFO] Cette etape peut prendre 15-20 minutes selon votre connexion...
echo.
echo Packages principaux qui seront installes:
echo   - PyTorch 2.8.0 stable + CUDA 12.9 (runtime PyTorch)
echo   - TensorRT 10.9.0.34
echo   - diffusers 0.36.0 (Hugging Face)
echo   - transformers 4.57.1
echo   - controlnet-aux 0.0.10 (Canny, Depth, OpenPose)
echo   - easy-dwpose 1.0.2 (pose detection)
echo   - ultralytics 8.4.14 (YOLO segmentation)
echo   - bitsandbytes 0.49.2 (Wan 2.1 NF4 quantization)
echo   - triton-windows 3.3.1 (torch.compile)
echo   - Flask 3.1.2 (interface web)
echo.
echo [INFO] L'installation va commencer. Gardez ce terminal ouvert...
echo.

python -m pip install -r requirements.txt --verbose

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERREUR] Echec de l'installation des dependances.
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Toutes les dependances installees avec succes.

REM Installer easy-dwpose separement (conflit artificiel huggingface_hub<0.25, API inchangee)
echo.
echo [INFO] Installation de easy-dwpose (--no-deps)...
python -m pip install easy-dwpose==1.0.2 --no-deps --quiet
echo [OK] easy-dwpose installe.

REM Verifier que PyTorch et CUDA fonctionnent
echo.
echo [INFO] Verification de PyTorch et CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [ATTENTION] PyTorch ou CUDA ne semble pas fonctionner correctement.
    echo Verifiez votre installation CUDA et les drivers NVIDIA.
    echo [INFO] L'installation continue...
) else (
    echo [OK] PyTorch et CUDA fonctionnent correctement.
)

echo.
echo ============================================================================
echo [Etape 3/4] Configuration des binaires CUDA et headers Python
echo ============================================================================
echo.

echo [INFO] Configuration pour torch.compile() et Triton...
echo [INFO] Cette etape copie les outils CUDA et headers Python necessaires.
echo.

python setup_venv.py

if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [ATTENTION] La configuration des binaires a echoue partiellement.
    echo StreamDiffusion fonctionnera quand meme, mais torch.compile^(^) pourrait ne pas marcher.
    echo [INFO] L'installation continue...
) else (
    echo [OK] Binaires CUDA et headers Python configures.
)

echo.
echo ============================================================================
echo [Etape 4/4] Verification de l'installation
echo ============================================================================
echo.

echo [INFO] Test de l'installation...
echo.

python verify_install.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERREUR] Le test d'installation a echoue.
    echo Verifiez les messages d'erreur ci-dessus.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo  Installation Terminee avec Succes!
echo ============================================================================
echo.
echo Prochaines etapes:
echo.
echo 1. Les modeles seront telecharges automatiquement au premier lancement:
echo    - Stable Diffusion 1.5 / SDXL (HuggingFace cache)
echo    - DWPose (dw-ll_ucoco_384.onnx, yolox_l.onnx)
echo    - YOLO segmentation (yolo11m-seg.pt)
echo    - SAM2 hiera-tiny (HuggingFace cache)
echo.
echo 2. Pour lancer l'interface web de configuration ControlNet:
echo    .venv\Scripts\activate
echo    python controlnet_web.py
echo.
echo 3. Pour utiliser dans Smode:
echo    - Chargez le package StreamDiffusion-R13 dans Smode Compose
echo    - Le script SmodeStreamDiffusion.py se chargera automatiquement
echo.
echo 4. Documentation: voir le dossier docs/
echo.
echo Espace disque utilise: ~12-15 GB total
echo   - Environnement .venv: ~3 GB
echo   - Modeles HuggingFace (cache): ~5-8 GB
echo   - Checkpoints DWPose/YOLO: ~400 MB (au premier lancement)
echo.
echo ============================================================================

color 0A
echo.
echo Appuyez sur une touche pour quitter...
pause >nul

endlocal
