# ============ Stage 1: Builder ============
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-inference.txt .
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements-inference.txt

# ============ Stage 2: Runtime ============
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

COPY --from=builder /install /usr/local

WORKDIR /app

COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

EXPOSE 8000

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["uvicorn", "src.inference.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]