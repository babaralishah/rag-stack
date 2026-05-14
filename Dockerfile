# ========================
# Production Dockerfile for Google Cloud Run
# ========================

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://127.0.0.1:8000 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x start-cloud.sh

# Cloud Run will provide $PORT environment variable
CMD ["./start-cloud.sh"]