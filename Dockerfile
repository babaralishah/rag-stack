# ========================
# Production Dockerfile for Hugging Face Spaces (CPU-optimized)
# Single container: FastAPI (background) + Streamlit on port 7860
# ========================

FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://127.0.0.1:8000 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf \
    TORCH_CPU_ONLY=1

# Install dependencies with CPU-only torch (critical for speed & size)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x /app/start.sh

EXPOSE 7860

# Use the start script
CMD ["/app/start.sh"]

# FROM python:3.11-slim

# WORKDIR /app

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_NO_CACHE_DIR=1
# ENV API_BASE_URL=http://127.0.0.1:8000

# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt

# COPY . .

# RUN chmod +x /app/start.sh

# EXPOSE 7860

# CMD ["/app/start.sh"]