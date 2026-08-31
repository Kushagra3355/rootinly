# Multi-stage production Dockerfile
FROM python:3.11-slim

# System dependencies for OpenCV & YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY configs/ ./configs/
COPY src/ ./src/
COPY models/ ./models/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY main.py .

# Environment variables
ENV HOST=0.0.0.0 \
    PORT=5000 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/best.pt \
    STAGE_MODEL_PATH=/app/models/best_norwood.pt

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

CMD ["python", "main.py", "--no-browser"]
