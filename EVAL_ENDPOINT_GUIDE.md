# FastAPI Evaluation Endpoint Integration Guide

## Overview

Your local Streamlit app (`ui.py`) now launches a background **FastAPI server** that exposes a dedicated `/eval` endpoint for retrieval-only (no LLM generation) evaluation and ablation testing.

## What Changed

### 1. **New `/eval` Endpoint in `src/api.py`**
   - **Purpose**: Retrieval-only endpoint for automated benchmarking
   - **URL**: `POST /eval`
   - **Accepts**:
     ```python
     {
       "query": str,                      # Required: Search question
       "chunks": int,                     # Optional: Number of chunks to retrieve (default: 3)
       "rerank": str,                     # Optional: "true"/"false" for cross-encoder (default: "true")
       "hybrid": str,                     # Optional: "true"/"false" for BM25+semantic (default: "true")
       "ablation": str,                   # Optional: Phase identifier (e.g., "V1", "V2")
       "rewriting_strategy": str          # Optional: "none"/"keyword_expansion"/"hyde" (default: "none")
     }
     ```
   - **Returns**:
     ```json
     {
       "status": "success",
       "retrieved_context_keys": ["doc_key_1", "doc_key_2", "doc_key_3"],
       "message": null
     }
     ```
     OR on error:
     ```json
     {
       "status": "error",
       "retrieved_context_keys": [],
       "message": "Error description"
     }
     ```

### 2. **Background FastAPI Server in `ui.py`**
   - Automatically launches on a **background thread** when you start `ui.py`
   - Runs on **port 8001** by default (configurable via `EVAL_SERVER_PORT` env var)
   - Uses thread-safe singleton pattern (`@st.cache_resource`) to ensure it starts only once
   - No changes needed to your Streamlit UI logic

### 3. **Updated `script.py` (Evaluation Script)**
   - Now queries the `/eval` endpoint directly via **POST JSON**
   - No longer tries to parse Streamlit HTML
   - Cleaner error handling and logging
   - Better support for both local and HuggingFace Spaces deployment

## How to Use Locally

### Step 1: Run the Streamlit App
```bash
# From the project root
streamlit run ui.py
```

This will:
- Start the Streamlit UI on `http://localhost:8501`
- Auto-launch the FastAPI server on `http://localhost:8001` (background thread)

### Step 2: Run Your Evaluation Script
```bash
# In another terminal, from docs/3.0/ (or wherever script.py is)
python script.py
```

The script will:
- Load your Excel benchmark
- Query the FastAPI `/eval` endpoint on `http://127.0.0.1:8001`
- Return retrieval metrics without LLM generation
- Print accuracy scores

### Example Query
```python
import requests

payload = {
    "query": "What is machine learning?",
    "chunks": 5,
    "rerank": "true",
    "hybrid": "false",
    "ablation": "V2",
    "rewriting_strategy": "hyde"
}

response = requests.post(
    "http://127.0.0.1:8001/eval",
    json=payload,
    timeout=30
)

result = response.json()
print(result["retrieved_context_keys"])  # List of doc IDs
```

## Deployment on HuggingFace Spaces

When deploying on HuggingFace Spaces, you get **one public URL** that serves Streamlit. The background FastAPI server runs internally and should be accessible via:

```
https://your-space-url.hf.space
```

However, HuggingFace Spaces typically only exposes port 8501 (Streamlit). The `/eval` endpoint will be accessible through the same Streamlit proxy.

### Option A: Query via Streamlit Proxy (Recommended)
```python
# In script.py for HuggingFace Spaces:
EVAL_SERVER_URL = "https://your-space-name-your-huggingface-username.hf.space"

# Query directly
response = requests.post(f"{EVAL_SERVER_URL}/eval", json=payload)
```

### Option B: Use Environment Variables
```bash
# In your HuggingFace Space deployment settings:
EVAL_SERVER_HOST=0.0.0.0
EVAL_SERVER_PORT=8001
```

Then in script.py:
```python
EVAL_SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://127.0.0.1:8001")
```

## Configuration

### Environment Variables
- **`EVAL_SERVER_HOST`** (default: `127.0.0.1`)
  - Change to `0.0.0.0` to accept external requests
- **`EVAL_SERVER_PORT`** (default: `8001`)
  - Port where FastAPI server listens
- **`API_BASE_URL`** (default: `http://127.0.0.1:8000`)
  - Backend API URL (for Streamlit to communicate with separate FastAPI if needed)

### Example with Custom Port
```bash
# Run on port 9000 instead
export EVAL_SERVER_PORT=9000
streamlit run ui.py
```

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│          Your Local Machine                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  ui.py (Streamlit App)              │   │
│  │  - Main UI on port 8501             │   │
│  │  - Evaluation mode handling         │   │
│  └──────────────────────────────────────┘   │
│              ↓                              │
│  ┌──────────────────────────────────────┐   │
│  │  FastAPI Server (Background Thread)  │   │
│  │  - Launched automatically on 8001    │   │
│  │  - Serves /eval endpoint            │   │
│  │  - Thread-safe (cache_resource)     │   │
│  └──────────────────────────────────────┘   │
│              ↑                              │
│  ┌──────────────────────────────────────┐   │
│  │  script.py (Evaluation Script)       │   │
│  │  - Queries /eval endpoint            │   │
│  │  - Returns retrieved_context_keys    │   │
│  │  - Calculates ablation metrics       │   │
│  └──────────────────────────────────────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

## Troubleshooting

### "Connection Error: Is FastAPI running?"
**Solution**: Make sure `ui.py` is running. The FastAPI server starts automatically when Streamlit starts.

### "Port 8001 already in use"
**Solution**: Either:
1. Kill the process: `lsof -ti:8001 | xargs kill -9` (Mac/Linux) or `netstat -ano | findstr :8001` (Windows)
2. Change the port: `export EVAL_SERVER_PORT=8002 && streamlit run ui.py`

### "Retrieved keys don't match expected"
**Debugging steps**:
1. Check if documents are indexed: Visit Streamlit UI and verify sources in sidebar
2. Verify Excel file path and column names in script.py
3. Check logs for query rewriting/retrieval details: Look at terminal output

### Performance Issues
- **Reduce `chunks` parameter** to retrieve fewer results (default: 3)
- **Disable reranking** (`"rerank": "false"`) to skip cross-encoder inference
- **Use `"rewriting_strategy": "none"`** to skip query rewriting

## API Endpoint Reference

### POST /eval
Retrieve documents without LLM generation.

**Example cURL**:
```bash
curl -X POST "http://localhost:8001/eval" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "chunks": 5,
    "rerank": "true",
    "hybrid": "true",
    "ablation": "V1",
    "rewriting_strategy": "hyde"
  }'
```

**Response**:
```json
{
  "status": "success",
  "retrieved_context_keys": ["doc_1", "doc_2", "doc_3"],
  "message": null
}
```

## Files Modified

1. **`src/api.py`**
   - Added `EvalRequest` and `EvalResponse` Pydantic models
   - Added `POST /eval` endpoint
   - Added helper function `_parse_bool_param()`

2. **`ui.py`**
   - Added imports: `threading`, `time`, `logging`, `app from src.api`
   - Added `EVAL_SERVER_PORT` and `EVAL_SERVER_HOST` config
   - Added `launch_fastapi_server()` function with background thread launcher
   - Called `launch_fastapi_server()` at startup

3. **`docs/3.0/script.py`**
   - Changed from GET to POST requests
   - Changed from query string parameters to JSON payload
   - Updated payload structure to match `/eval` endpoint
   - Better error handling and connection detection
   - Clearer logging and output

## Next Steps

1. ✅ Test locally: Run `streamlit run ui.py` and then `python script.py`
2. ✅ Verify ablation matrices are working with different configurations
3. ✅ Deploy to HuggingFace Spaces when ready
4. ✅ Update any other evaluation scripts to use the `/eval` endpoint

## Support

For issues or questions:
- Check the logs in the Streamlit terminal (FastAPI logs appear there)
- Verify documents are indexed by visiting the Streamlit UI sidebar
- Test the endpoint manually with cURL or Postman before running the full script
