@echo off
REM ============================================================================
REM StreamDiffusion - Uninstall ControlNet
REM ============================================================================
REM Removes the ControlNet subfolder. StreamDiffusion itself is kept intact.
REM ============================================================================

setlocal
cd /d "%~dp0"
color 0E

echo.
echo ============================================================================
echo  StreamDiffusion - Uninstall ControlNet
echo ============================================================================
echo.

if not exist "ControlNet" (
    echo [INFO] ControlNet is not installed. Nothing to remove.
    echo.
    pause
    exit /b 0
)

echo [INFO] Removing ControlNet directory...
rmdir /s /q "ControlNet"
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Failed to remove ControlNet directory.
    echo    Close any applications using it and try again.
    echo.
    pause
    exit /b 1
)

echo [OK] ControlNet removed.
echo.
echo Restart Smode to apply the change.
echo.

color 0A
pause >nul
endlocal
