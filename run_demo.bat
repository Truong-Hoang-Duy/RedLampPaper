@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   RedLamp Web Demo (demo_redlamp) - Windows Runner
echo ========================================================

set "PY_CMD=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
    echo [INFO] Su dung moi truong ao: .venv\Scripts\python.exe
) else (
    echo [INFO] Su dung Python he thong: python
)

echo [INFO] Dang mo trinh duyet tai http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

echo [INFO] Khoi chay FastAPI Server (demo_redlamp\server.py) tai cong 8000...
%PY_CMD% demo_redlamp\server.py
