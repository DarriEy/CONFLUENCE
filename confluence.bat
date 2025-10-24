@echo off
REM CONFLUENCE Windows Launcher

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Check if --install flag is present
set "INSTALL_MODE="
for %%a in (%*) do (
    if "%%a"=="--install" set "INSTALL_MODE=1"
)

if defined INSTALL_MODE (
    echo Installing CONFLUENCE...
    python CONFLUENCE.py --get_executables
    python CONFLUENCE.py --validate_binaries
    exit /b 0
)

REM Run CONFLUENCE normally
python CONFLUENCE.py %*
