FROM python:3.11-slim

WORKDIR /app

# Cài đặt PyTorch CPU nhẹ và các thư viện cần thiết cho Web Studio
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch numpy pandas scikit-learn scipy tqdm requests matplotlib patool fastapi uvicorn

COPY . /app

EXPOSE 7860 8080

CMD ["sh", "-c", "uvicorn demo.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
