@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   RedLamp Vibrant Web Studio (KDD '25) - Windows Runner
echo ========================================================

set "PY_CMD=python"
if exist ".venv\Scripts\python.exe" (
    set "PY_CMD=.venv\Scripts\python.exe"
    echo [INFO] Su dung moi truong ao: .venv\Scripts\python.exe
) else (
    echo [INFO] Su dung Python he thong: python
)

%PY_CMD% -c "import fastapi, uvicorn, torch, numpy, pandas, sklearn, scipy" 2>nul
if errorlevel 1 (
    echo [INFO] Dang cai dat cac thu vien con thieu tu requirements.txt...
    %PY_CMD% -m pip install -r requirements.txt
)

echo [INFO] Dang mo trinh duyet tai http://localhost:8080 ...
start "" http://localhost:8080

echo [INFO] Khoi chay FastAPI Server tai cong 8080 (Nhan Ctrl+C de dung)...
%PY_CMD% -m uvicorn demo.server:app --host 0.0.0.0 --port 8080 --reload

