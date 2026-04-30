from pathlib import Path
import os


DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"

STORE_DIR = Path("storage") / "faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 3
MIN_SCORE = 0.35
MAX_CHARS = 1500

CHUNK_OVERLAP = 120
CHUNK_SIZE = 600

EMBED_DIM = 384

MIN_SIMILARITY = 0.30

# Reranker Settings
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_K = 12         # How many chunks to retrieve initially
RERANKER_KEEP_TOP_K = 6      # How many best chunks to keep after reranking
USE_RERANKER = True          # Easy toggle

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

