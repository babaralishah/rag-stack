import logging
import hashlib
import json
from functools import wraps
from datetime import datetime
from cachetools import TTLCache
from typing import Dict

logger = logging.getLogger("rag")

# Global Caches
embedding_cache = TTLCache(maxsize=1000, ttl=3600)   # 1 hour
query_cache = TTLCache(maxsize=500, ttl=1800)         # 30 minutes

cache_stats = {
    "embedding_hits": 0,
    "embedding_misses": 0,
    "query_hits": 0,
    "query_misses": 0,
    "last_cleared": datetime.now().isoformat()
}


def generate_cache_key(question: str, **kwargs) -> str:
    data = {
        "question": question.strip().lower(),
        "use_hybrid": bool(kwargs.get("use_hybrid", True)),
        "use_reranker": bool(kwargs.get("use_reranker", True)),
        "top_k": int(kwargs.get("top_k", 6))
    }
    key_string = json.dumps(data, sort_keys=True)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cache_embedding(func):
    """Safe decorator for embedding cache"""
    @wraps(func)
    def wrapper(self, text: str, *args, **kwargs):
        if not isinstance(text, str) or not text.strip():
            return func(self, text, *args, **kwargs)

        key = f"emb_{hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()}"

        if key in embedding_cache:
            cache_stats["embedding_hits"] += 1
            return embedding_cache[key]

        cache_stats["embedding_misses"] += 1
        result = func(self, text, *args, **kwargs)
        
        if result is not None:
            embedding_cache[key] = result
            
        return result
    return wrapper


def cache_query(func):
    """Safe decorator for full query cache"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract question safely
        question = kwargs.get("question") or (args[1] if len(args) > 1 else "")
        if not question:
            return func(*args, **kwargs)

        key = generate_cache_key(question, **kwargs)

        if key in query_cache:
            cache_stats["query_hits"] += 1
            logger.info(f"🔄 QUERY CACHE HIT: {question[:60]}...")
            return query_cache[key]

        cache_stats["query_misses"] += 1
        result = func(*args, **kwargs)

        if isinstance(result, dict) and "answer" in result:
            query_cache[key] = result
            logger.info(f"✅ Query cached: {question[:60]}...")

        return result
    return wrapper


def clear_all_caches(reason: str = "manual"):
    embedding_cache.clear()
    query_cache.clear()
    cache_stats["last_cleared"] = datetime.now().isoformat()
    logger.info(f"🧹 All caches cleared. Reason: {reason}")


def get_cache_stats() -> Dict:
    total_emb = cache_stats["embedding_hits"] + cache_stats["embedding_misses"]
    total_query = cache_stats["query_hits"] + cache_stats["query_misses"]

    return {
        "embedding_hit_rate": round((cache_stats["embedding_hits"] / total_emb * 100), 1) if total_emb > 0 else 0,
        "query_hit_rate": round((cache_stats["query_hits"] / total_query * 100), 1) if total_query > 0 else 0,
        "embedding_entries": len(embedding_cache),
        "query_entries": len(query_cache),
        "last_cleared": cache_stats["last_cleared"]
    }