FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy pandas scikit-learn xgboost && \
    pip install --no-cache-dir opencv-python mediapipe && \
    pip install --no-cache-dir fastapi uvicorn pyyaml joblib tqdm

COPY . .

RUN mkdir -p models config

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "src.inference.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
