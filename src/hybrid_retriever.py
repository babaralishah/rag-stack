import logging
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger("rag")

class HybridRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []  # Store original chunk data

    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from all chunks"""
        self.chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"✅ BM25 index built with {len(chunks)} chunks")

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Pure keyword search using BM25"""
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])  # copy
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)
        return results