#!/bin/bash
echo "🚀 Starting RAG App on Google Cloud Run..."

# Start FastAPI backend in background
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait for FastAPI to start
sleep 4

# Start Streamlit on the PORT given by Cloud Run (very important)
streamlit run ui.py \
  --server.port ${PORT:-8080} \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --logger.level error