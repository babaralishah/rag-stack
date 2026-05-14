#!/bin/bash
echo "🚀 Starting RAG App on Google Cloud Run..."

# Start FastAPI in background
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level warning &

# Give FastAPI time to start
sleep 6

# Start Streamlit on Cloud Run's required PORT
echo "Starting Streamlit on port ${PORT:-8080}..."
exec streamlit run ui.py \
  --server.port ${PORT:-8080} \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --logger.level error