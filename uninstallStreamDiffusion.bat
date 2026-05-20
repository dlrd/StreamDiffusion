@echo off
REM ============================================================================
REM StreamDiffusion - Uninstall StreamDiffusion
REM ============================================================================
REM Removes the .venv virtual environment (Python packages).
REM The StreamDiffusion scripts themselves are kept; only the installed
REM dependencies are deleted. Smode's bundled Python is NOT touched.
REM ============================================================================

setlocal
cd /d "%~dp0"
color 0E

echo.
echo ============================================================================
echo  StreamDiffusion - Uninstall Dependencies
echo ============================================================================
echo.

REM Terminate any running SmodeStreamDiffusion Python process
echo [INFO] Stopping any running StreamDiffusion process...
taskkill /f /fi "WINDOWTITLE eq StreamDiffusion*" /im python.exe >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq python*" /im python.exe >nul 2>&1

REM Uninstall ControlNet first (cannot live without StreamDiffusion)
if exist "ControlNet" (
    echo [INFO] Removing ControlNet directory...
    rmdir /s /q "ControlNet"
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to remove ControlNet directory.
        echo.
        pause
        exit /b 1
    )
    echo [OK] ControlNet removed.
)

if exist ".venv" (
    echo [INFO] Removing virtual environment (.venv)...
    rmdir /s /q ".venv"
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to remove .venv.
        echo    Make sure no Python process is still running and try again.
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment removed.
) else (
    echo [INFO] Virtual environment (.venv) not found. Nothing to remove.
)

echo.
echo [OK] StreamDiffusion dependencies uninstalled.
echo     Run install.bat or use 'Auto Install' in Smode to reinstall.
echo.

color 0A
pause >nul
endlocal
