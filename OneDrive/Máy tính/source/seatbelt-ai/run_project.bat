@echo off
setlocal
cd /d "%~dp0"


set "TCL_LIBRARY=%LOCALAPPDATA%\Programs\Python\Python313\tcl\tcl8.6"
set "TK_LIBRARY=%LOCALAPPDATA%\Programs\Python\Python313\tcl\tk8.6"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay Python trong .venv
    pause
    exit /b 1
)

echo 
echo.

".venv\Scripts\python.exe" main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Chuong trinh gap loi.
)

echo.
pause
endlocal
