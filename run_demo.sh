#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "========================================================"
echo "  RedLamp Vibrant Web Studio (KDD '25) - macOS / Linux"
echo "========================================================"

if [ -x ".venv/bin/python" ]; then
    PY_CMD=".venv/bin/python"
    echo "[INFO] Sử dụng môi trường ảo: .venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
    echo "[INFO] Sử dụng Python hệ thống: $(command -v python3)"
else
    PY_CMD="python"
    echo "[INFO] Sử dụng Python hệ thống: $(command -v python)"
fi

if ! "$PY_CMD" -c "import fastapi, uvicorn, torch, numpy, pandas, sklearn, scipy" >/dev/null 2>&1; then
    echo "[INFO] Đang cài đặt các thư viện còn thiếu từ requirements.txt..."
    "$PY_CMD" -m pip install -r requirements.txt
fi

echo "[INFO] Đang mở trình duyệt tại http://localhost:8080 ..."
if command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://localhost:8080") &
elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 1.5 && xdg-open "http://localhost:8080") &
fi

echo "[INFO] Khởi chạy FastAPI Server tại cổng 8080 (Nhấn Ctrl+C để dừng)..."
exec "$PY_CMD" -m uvicorn demo.server:app --host 0.0.0.0 --port 8080 --reload

