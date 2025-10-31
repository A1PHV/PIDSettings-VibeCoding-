@echo off
REM Windows batch script to build .exe

echo ============================================================
echo Building PID Tuner .exe
echo ============================================================
echo.

REM Check Python
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

REM Build
echo.
echo Building executable...
python build_exe.py

echo.
echo ============================================================
echo Done! Check the 'dist' folder for pid_tuner.exe
echo ============================================================
pause
