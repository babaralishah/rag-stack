# ========================
# Production Dockerfile - Hugging Face Spaces (CPU only)
# ========================

FROM python:3.11-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables (HF requires /tmp for everything writable)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://127.0.0.1:8000 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_DATASETS_CACHE=/tmp/hf \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf

# Install CPU-only torch first (greatly reduces build time & size)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/start.sh

# Run as non-root (HF best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["/app/start.sh"]