"""Global configuration constants used throughout the Local RAG app.

This module centralizes model names, vector store locations, default retrieval
settings, and environment-driven endpoints.
"""

from pathlib import Path
import os

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"

STORE_DIR = Path("storage") / "faiss"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBED_MODEL = "BAAI/bge-large-en-v1.5"
OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 3
MIN_SCORE = 0.35
MAX_CHARS = 1500

CHUNK_OVERLAP = 120
CHUNK_SIZE = 600

# EMBED_DIM = 384
# EMBED_DIM = 1024
# Dynamic dimensions based on the active model
EMBED_DIM = 1024 if "bge-large" in EMBED_MODEL else 384

MIN_SIMILARITY = 0.30
CHAT_HISTORY_TURNS = 8

# Reranker Settings
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_K = 12  # How many chunks to retrieve initially
RERANKER_KEEP_TOP_K = 6  # How many best chunks to keep after reranking
USE_RERANKER = True  # Easy toggle
RERANKER_FUSION_ALPHA = 0.65  # 0.65 reranker + 0.35 embedding

# Ablation study phase toggles
PHASE_FLAGS = {
    "V1": {  # Absolute Baseline Floor
        # "embedding_model": "all-MiniLM-L6-v2 (384d)",
        "use_query_rewriter": False,
        "use_hybrid": False,
        "use_chat_history": False,
        "use_reranker": False,
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V2": {  # Isolate Algorithmic Optimization: Query Rewriting
        # "embedding_model": "all-MiniLM-L6-v2 (384d)",
        "use_query_rewriter": True, 
        "use_hybrid": False,
        "use_chat_history": False,
        "use_reranker": False,
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V3": {  # Isolate Representation Capacity: Vector Density Upgrade
        # "embedding_model": "bge-large-en-v1.5 (1024d)",
        "use_query_rewriter": True,                      # No other flags change
        "use_hybrid": False,
        "use_chat_history": False, 
        "use_reranker": False,
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V4": {  # Isolate Context Awareness: Window Memory
        # "embedding_model": "bge-large-en-v1.5 (1024d)",
        "use_query_rewriter": True,
        "use_hybrid": False,
        "use_chat_history": True,                        # Swapped here!
        "use_reranker": False,
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V5": {  # Isolate Retrieval Multi-Modality: BM25 + RRF
        # "embedding_model": "bge-large-en-v1.5 (1024d)",
        "use_query_rewriter": True,
        "use_hybrid": True,                              # Swapped here!
        "use_chat_history": True,
        "use_reranker": False,
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V6": {  # Isolate Document Re-scoring: Cross-Encoder Reranking
        # "embedding_model": "bge-large-en-v1.5 (1024d)",
        "use_query_rewriter": True,
        "use_hybrid": True,
        "use_chat_history": True,
        "use_reranker": True,                            # Swapped here!
        "use_guardrails": False,
        "use_cache": False,
        "use_cot": False,
        "use_few_shot": False,
    },
    "V7": {  # Maximum Production Optimization Engine
        # "embedding_model": "bge-large-en-v1.5 (1024d)",
        "use_query_rewriter": True,
        "use_hybrid": True,
        "use_chat_history": True,
        "use_reranker": True,
        "use_guardrails": True,
        "use_cache": True,
        "use_cot": True,
        "use_few_shot": True,
    },
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Document Management
DOCUMENTS_DIR = UPLOADS_DIR / "documents"  # We'll store metadata
