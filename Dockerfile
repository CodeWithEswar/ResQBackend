# Use official lightweight Python 3.12 image
FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PORT=8000

# Install minimal OS dependencies for network operations, healthchecks, PyTorch, and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import cv2; _ = cv2.HOGDescriptor(); print('OpenCV', cv2.__version__, 'and HOGDescriptor OK')" && \
    python -c "import torch; print('PyTorch successfully loaded:', torch.__version__)"

# Copy application files, models, and artifacts
COPY app/ ./app/
COPY models/ ./models/
COPY ml/ ./ml/
COPY api/ ./api/

# Expose standard port (Render will override via $PORT at runtime)
EXPOSE 8000

# Health check to ensure service is ready before traffic routing
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start Uvicorn ASGI server bound to all interfaces
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
