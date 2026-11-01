@echo off
rem ================================================================
rem  Image Compressive Sensing Reconstruction Demo - launcher
rem  
rem  - locates the project by its own folder (%~dp0), no absolute
rem    path is hard coded, so the project can be moved/copied freely
rem  - keeps this console open so the log and any error stay visible
rem  - enables auto reload on source change, so edits to app.py or
rem    src/*.py take effect without restarting the server
rem
rem  Usage: double click the desktop shortcut, or run this file.
rem ================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Image CS Reconstruction Demo System

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Not found: .venv\Scripts\streamlit.exe
    echo         Create the virtual environment first ^(README section 2^).
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr ":8501 " | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [WARN] Port 8501 is already in use by another process.
    echo        Streamlit will start on the next free port instead.
    echo        To avoid two instances, close the old window first.
    echo.
)

echo [INFO] Project dir : %CD%
echo [INFO] Entry file  : app.py
echo [INFO] Default URL : http://127.0.0.1:8501  ^(opened automatically^)
echo [INFO] Stop server : press Ctrl+C in this window
echo.

".venv\Scripts\streamlit.exe" run app.py --server.runOnSave true

echo.
echo [INFO] Streamlit exited. Exit code = %ERRORLEVEL%
pause
