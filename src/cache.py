import logging
import hashlib
import json
from datetime import datetime
from cachetools import TTLCache
from typing import Dict

logger = logging.getLogger("rag")

# Caches
embedding_cache = TTLCache(maxsize=1000, ttl=3600)
query_cache = TTLCache(maxsize=500, ttl=1800)

cache_stats = {
    "embedding_hits": 0,
    "embedding_misses": 0,
    "query_hits": 0,
    "query_misses": 0,
    "last_cleared": datetime.now().isoformat()
}


def get_cache_key(question: str, use_hybrid=True, use_reranker=True, top_k=6):
    data = {
        "q": question.strip().lower(),
        "hybrid": bool(use_hybrid),
        "rerank": bool(use_reranker),
        "k": int(top_k)
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def clear_all_caches(reason="manual"):
    embedding_cache.clear()
    query_cache.clear()
    cache_stats["last_cleared"] = datetime.now().isoformat()
    logger.info(f"🧹 All caches cleared. Reason: {reason}")


def get_cache_stats() -> Dict:
    total_emb = cache_stats["embedding_hits"] + cache_stats["embedding_misses"]
    total_q = cache_stats["query_hits"] + cache_stats["query_misses"]
    
    return {
        "embedding_hit_rate": round((cache_stats["embedding_hits"] / total_emb * 100), 1) if total_emb > 0 else 0,
        "query_hit_rate": round((cache_stats["query_hits"] / total_q * 100), 1) if total_q > 0 else 0,
        "embedding_entries": len(embedding_cache),
        "query_entries": len(query_cache),
        "last_cleared": cache_stats["last_cleared"]
    }