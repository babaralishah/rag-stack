#!/bin/bash
set -e

echo "🚀 Starting RAG App on Hugging Face Spaces..."

# Start FastAPI in background
echo "Starting FastAPI on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level info &

# Better readiness check
echo "Waiting for FastAPI to be ready..."
for i in {1..40}; do
    if curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1 || \
       curl -s -f http://127.0.0.1:8000/docs >/dev/null 2>&1; then
        echo "✅ FastAPI is ready!"
        break
    fi
    sleep 3
done

# Start Streamlit in foreground (required by HF)
echo "Starting Streamlit on port 7860..."
exec streamlit run ui.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --logger.level error \
    --server.enableCORS false \
    --server.enableXsrfProtection false