#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "========================================================"
echo "  RedLamp Web Demo (demo_redlamp) - macOS / Linux"
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

echo "[INFO] Đang mở trình duyệt tại http://127.0.0.1:8000 ..."
if command -v open >/dev/null 2>&1; then
    (sleep 1.5 && open "http://127.0.0.1:8000") &
elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 1.5 && xdg-open "http://127.0.0.1:8000") &
fi

echo "[INFO] Khởi chạy FastAPI Server (demo_redlamp/server.py) tại cổng 8000..."
exec "$PY_CMD" demo_redlamp/server.py
