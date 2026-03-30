#!/bin/bash
set -e

echo "Starting Local RAG App on Hugging Face Spaces..."

# Start FastAPI in background
echo "Starting FastAPI server on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready (better than fixed sleep)
echo "Waiting for FastAPI to start..."
for i in {1..30}; do
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1 || \
       curl -s -f http://localhost:8000/docs > /dev/null 2>&1; then
        echo "FastAPI is ready!"
        break
    fi
    sleep 2
done

# Start Streamlit (must run in foreground on port 7860 for HF)
echo "Starting Streamlit on port 7860..."
streamlit run ui.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.headless true \
    --logger.level error

# #!/bin/bash
# set -e

# uvicorn src.api:app --host 0.0.0.0 --port 8000 &
# sleep 5
# streamlit run ui.py --server.port 7860 --server.address 0.0.0.0