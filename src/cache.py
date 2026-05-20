"""In-memory caching for embeddings and query results.

This module holds small TTL caches to reduce repeat embedding calls and
repeated query work in the Local RAG application.
"""

import logging
import hashlib
import json
from datetime import datetime
from cachetools import TTLCache
from typing import Dict

logger = logging.getLogger("rag")

# ==================== CACHES ====================
embedding_cache = TTLCache(maxsize=1000, ttl=3600)
query_cache = TTLCache(maxsize=500, ttl=1800)

cache_stats = {
    "embedding_hits": 0,
    "embedding_misses": 0,
    "query_hits": 0,
    "query_misses": 0,
    "last_cleared": datetime.now().isoformat(),
}


def _normalize_history_for_cache(history):
    if not history:
        return None

    normalized = []
    for message in history:
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
        else:
            role = str(getattr(message, "role", "")).strip().lower()
            content = str(getattr(message, "content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized if normalized else None


def get_cache_key(question: str, use_hybrid=True, use_reranker=True, top_k=6, history=None):
    """Create a stable cache key for a query configuration."""
    data = {
        "q": question.strip().lower(),
        "hybrid": bool(use_hybrid),
        "rerank": bool(use_reranker),
        "k": int(top_k),
    }

    normalized_history = _normalize_history_for_cache(history)
    if normalized_history:
        history_hash = hashlib.md5(
            json.dumps(
                normalized_history,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        data["history_hash"] = history_hash

    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def clear_all_caches(reason="manual"):
    """Clear all in-memory caches and record the clearing timestamp."""
    embedding_cache.clear()
    query_cache.clear()
    cache_stats["last_cleared"] = datetime.now().isoformat()
    logger.info(f"🧹 All caches cleared. Reason: {reason}")


def get_cached_query(key):
    """Return a cached query result if present, otherwise None."""
    if key in query_cache:
        cache_stats["query_hits"] += 1
        return query_cache[key]

    cache_stats["query_misses"] += 1
    return None


def set_cached_query(key, value):
    """Store a query response in cache for later reuse."""
    query_cache[key] = value
    return value


def get_cache_stats() -> Dict:
    """Return cache hit/miss statistics and current entry counts."""
    total_emb = cache_stats["embedding_hits"] + cache_stats["embedding_misses"]
    total_q = cache_stats["query_hits"] + cache_stats["query_misses"]

    return {
        "embedding_hit_rate": (
            round((cache_stats["embedding_hits"] / total_emb * 100), 1)
            if total_emb > 0
            else 0
        ),
        "query_hit_rate": (
            round((cache_stats["query_hits"] / total_q * 100), 1) if total_q > 0 else 0
        ),
        "embedding_entries": len(embedding_cache),
        "query_entries": len(query_cache),
        "last_cleared": cache_stats["last_cleared"],
    }
