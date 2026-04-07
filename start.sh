#!/bin/bash
set -e

echo "🚀 Starting RAG App on Hugging Face Spaces..."

# Start FastAPI in background
echo "Starting FastAPI on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level info &

# Better readiness check with timeout
echo "Waiting for FastAPI to be ready..."
timeout=30
for i in $(seq 1 $timeout); do
    if curl -s -f http://127.0.0.1:8000/health > /dev/null 2>&1 || \
       curl -s -f http://127.0.0.1:8000/docs > /dev/null 2>&1; then
        echo "✅ FastAPI is ready!"
        break
    fi
    sleep 2
done

if [ $i -eq $timeout ]; then
    echo "⚠️ Warning: FastAPI did not respond in time. Continuing anyway..."
fi

# Start Streamlit in foreground (required by HF)
echo "Starting Streamlit on port 7860..."
exec streamlit run ui.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --logger.level error \
    --server.enableCORS false \
    --server.enableXsrfProtection false