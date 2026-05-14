#!/bin/bash
set -e

echo "🚀 Starting RAG App on Google Cloud Run..."

# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level warning &

# Wait and check if FastAPI is ready
echo "Waiting for FastAPI to be ready..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null; then
        echo "✅ FastAPI is ready!"
        break
    fi
    echo "Waiting... ($i/15)"
    sleep 2
done

# Start Streamlit on Cloud Run's PORT
echo "Starting Streamlit on port ${PORT:-8080}..."
exec streamlit run ui.py \
  --server.port ${PORT:-8080} \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --logger.level error
