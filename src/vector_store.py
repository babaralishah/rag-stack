import logging
from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.config import TOP_K

logger = logging.getLogger("rag")


class FaissVectorStore:
    def __init__(self, dim: int, store_dir: str):
        self.dim = dim
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "meta.jsonl"

        self.index = faiss.IndexFlatIP(dim)   # Semantic Search
        self.bm25 = None                      # Keyword Search (BM25)
        self.records: List[Dict[str, Any]] = []  # All chunks

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]]):
        if len(texts) != len(metadatas) or len(texts) != embeddings.shape[0]:
            raise ValueError("Lengths mismatch")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)

        for t, m in zip(texts, metadatas):
            self.records.append({"text": t, "metadata": m})

        self._build_bm25()   # Build keyword index

        logger.info(f"Added {len(texts)} vectors. Total now: {len(self.records)}")

    def _build_bm25(self):
        """Build BM25 keyword index"""
        if not self.records:
            return
        tokenized = [rec["text"].lower().split() for rec in self.records]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self.records)} documents")

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """Hybrid Search: Semantic + BM25"""
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if self.index.ntotal == 0:
            return []

        # 1. Semantic Search (FAISS)
        semantic_scores, semantic_idxs = self.index.search(query_vec, top_k * 2)
        results = []
        for score, idx in zip(semantic_scores[0], semantic_idxs[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            results.append({
                "text": rec["text"],
                "metadata": rec["metadata"],
                "score": float(score),           # FAISS semantic score
                "semantic_score": float(score)
            })

        # 2. Keyword Search (BM25) if hybrid enabled
        if use_hybrid and self.bm25 and len(results) > 0:
            try:
                # Use top semantic results as context for better keyword search
                context_text = " ".join([r["text"] for r in results[:8]])
                tokenized_query = context_text.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)

                top_bm25_idx = np.argsort(bm25_scores)[-top_k:][::-1]

                for idx in top_bm25_idx:
                    rec = self.records[idx]
                    results.append({
                        "text": rec["text"],
                        "metadata": rec["metadata"],
                        "score": float(bm25_scores[idx]),
                        "bm25_score": float(bm25_scores[idx])
                    })
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # Remove duplicates and return top_k
        seen = set()
        unique_results = []
        for r in results:
            key = r["text"][:150]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results[:top_k]

    def save(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Saved FAISS + BM25 index")

    @classmethod
    def load(cls, store_dir: str) -> "FaissVectorStore":
        store_dir = Path(store_dir)
        index_path = store_dir / "index.faiss"
        meta_path = store_dir / "meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found in {store_dir}")

        index = faiss.read_index(str(index_path))
        dim = index.d

        obj = cls(dim=dim, store_dir=str(store_dir))
        obj.index = index

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj.records.append(json.loads(line))

        obj._build_bm25()   # Rebuild BM25 index after loading

        logger.info(f"Loaded FAISS + BM25 index ({obj.index.ntotal} vectors)")
        return obj