import logging
import hashlib
import json
from functools import wraps
from datetime import datetime
from cachetools import TTLCache, LRUCache
from typing import Any, Dict, Optional

logger = logging.getLogger("rag")

# ==================== GLOBAL CACHES ====================
embedding_cache = TTLCache(maxsize=1000, ttl=3600)   # 1 hour
query_cache = TTLCache(maxsize=500, ttl=1800)         # 30 minutes

# Cache statistics
cache_stats = {
    "embedding_hits": 0,
    "embedding_misses": 0,
    "query_hits": 0,
    "query_misses": 0,
    "last_cleared": datetime.now().isoformat()
}


def generate_cache_key(question: str, **kwargs) -> str:
    """Generate stable and unique cache key"""
    data = {
        "question": question.strip().lower(),
        "use_hybrid": kwargs.get("use_hybrid", True),
        "use_reranker": kwargs.get("use_reranker", True),
        "top_k": kwargs.get("top_k", 6)
    }
    key_string = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cache_embedding(func):
    """Decorator for embedding cache with stats"""
    @wraps(func)
    def wrapper(text: str, *args, **kwargs):
        if not text:
            return func(text, *args, **kwargs)
            
        key = f"emb_{hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()}"
        
        if key in embedding_cache:
            cache_stats["embedding_hits"] += 1
            logger.debug(f"Embedding cache HIT: {text[:50]}...")
            return embedding_cache[key]
        
        cache_stats["embedding_misses"] += 1
        result = func(text, *args, **kwargs)
        embedding_cache[key] = result
        logger.debug(f"Embedding cached: {text[:50]}...")
        return result
    return wrapper


def cache_query(func):
    """Decorator for full RAG query cache"""
    @wraps(func)
    def wrapper(question: str, **kwargs):
        key = generate_cache_key(question, **kwargs)
        
        if key in query_cache:
            cache_stats["query_hits"] += 1
            logger.info(f"🔄 QUERY CACHE HIT: {question[:70]}...")
            return query_cache[key]
        
        cache_stats["query_misses"] += 1
        result = func(question, **kwargs)
        
        if result and isinstance(result, dict) and "answer" in result:
            query_cache[key] = result
            logger.info(f"✅ Query cached successfully: {question[:60]}...")
        
        return result
    return wrapper


def clear_all_caches(reason: str = "manual"):
    """Clear all caches - call after document changes"""
    embedding_cache.clear()
    query_cache.clear()
    cache_stats["last_cleared"] = datetime.now().isoformat()
    logger.info(f"🧹 All caches cleared. Reason: {reason}")


def get_cache_stats() -> Dict:
    """Return cache statistics for UI"""
    total_embedding = cache_stats["embedding_hits"] + cache_stats["embedding_misses"]
    total_query = cache_stats["query_hits"] + cache_stats["query_misses"]
    
    return {
        "embedding_hit_rate": round(cache_stats["embedding_hits"] / total_embedding * 100, 1) if total_embedding > 0 else 0,
        "query_hit_rate": round(cache_stats["query_hits"] / total_query * 100, 1) if total_query > 0 else 0,
        "embedding_entries": len(embedding_cache),
        "query_entries": len(query_cache),
        "last_cleared": cache_stats["last_cleared"]
    }