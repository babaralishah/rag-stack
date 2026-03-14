from pathlib import Path

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"

STORE_DIR = Path("storage") / "faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 3
MIN_SCORE = 0.35
MAX_CHARS = 1500

EMBED_DIM = 384

MIN_SIMILARITY = 0.30

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
