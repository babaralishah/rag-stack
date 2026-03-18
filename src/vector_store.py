# Stores & searches vectors using Faiss. Each vector is associated with a text chunk and metadata. The store can be saved to disk and loaded later for retrieval.


from dataclasses import asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import logging
import numpy as np

import faiss

from src.config import TOP_K

logger = logging.getLogger(__name__)

class FaissVectorStore:
    def __init__(self, dim: int, store_dir: str):
        self.dim = dim
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "meta.jsonl"

        self.index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
        self.records: List[Dict[str, Any]] = []  # each: {"text":..., "metadata":...}

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]]):
        if len(texts) != len(metadatas) or len(texts) != embeddings.shape[0]:
            raise ValueError("texts/metadatas/embeddings lengths do not match")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        for t, m in zip(texts, metadatas):
            self.records.append({"text": t, "metadata": m})

        logger.info("Added %d vectors. Total now: %d", len(texts), len(self.records))

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if self.index.ntotal == 0:
            return []

        scores, idxs = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            results.append({
                "score": float(score),
                "text": rec["text"],
                "metadata": rec["metadata"],
            })
        return results

    def save(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Saved FAISS index to %s", self.store_dir)

    @classmethod
    def load(cls, store_dir: str) -> "FaissVectorStore":
        store_dir = Path(store_dir)
        index_path = store_dir / "index.faiss"
        meta_path = store_dir / "meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing index files in: {store_dir}")

        index = faiss.read_index(str(index_path))
        dim = index.d
        obj = cls(dim=dim, store_dir=str(store_dir))
        obj.index = index

        records = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        obj.records = records

        logger.info("Loaded FAISS index (%d vectors) from %s", obj.index.ntotal, store_dir)
        return obj