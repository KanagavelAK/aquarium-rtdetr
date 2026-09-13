FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_WEIGHTS=/app/artifacts/best.pt \
    WEIGHTS_URL=https://github.com/KanagavelAK/aquarium-rtdetr/releases/download/v1.0/best.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# opencv and torch need these; keep the layer small
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image at a size a reviewer will actually pull
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
# artifacts/ may be empty; the API fetches best.pt on first start if it is missing
RUN mkdir -p artifacts
COPY artifacts/ ./artifacts/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
