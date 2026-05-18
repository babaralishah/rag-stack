import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import faiss
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from src.config import TOP_K

logger = logging.getLogger("rag")


class FaissVectorStore:
    def __init__(self, dim: int, store_dir: str):
        self.dim = dim
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "meta.jsonl"

        self.index = faiss.IndexFlatIP(dim)
        self.bm25: Optional[BM25Okapi] = None
        self.records: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]]):
        if len(texts) != len(metadatas) or len(texts) != embeddings.shape[0]:
            raise ValueError("Lengths of embeddings, texts, and metadatas must match")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)

        for t, m in zip(texts, metadatas):
            self.records.append({"text": t, "metadata": m})

        self._build_bm25()

        logger.info(f"Added {len(texts)} documents. Total vectors: {len(self.records)}")

    def _build_bm25(self):
        if not self.records:
            return
        if BM25Okapi is None:
            logger.debug("rank_bm25 not available; skipping BM25 build")
            return
        tokenized = [rec["text"].lower().split() for rec in self.records]
        self.bm25 = BM25Okapi(tokenized)
        logger.debug(f"BM25 index built with {len(self.records)} documents")

    # ====================== INTERNAL SEARCH ======================

    def _semantic_search(self, query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype("float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            rec = self.records[idx]
            results.append({
                "text": rec["text"],
                "metadata": rec["metadata"],
                "semantic_score": float(score),
                "score": float(score),
                "source": "semantic"
            })
        return results

    def _bm25_search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        if not self.bm25 or not query_text or not query_text.strip():
            return []

        try:
            tokenized_query = query_text.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[-k:][::-1]

            results = []
            for idx in top_indices:
                rec = self.records[idx]
                results.append({
                    "text": rec["text"],
                    "metadata": rec["metadata"],
                    "bm25_score": float(bm25_scores[idx]),
                    "score": float(bm25_scores[idx]),
                    "source": "bm25"
                })
            return results
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    @staticmethod
    def _reciprocal_rank_fusion(
        lists: List[List[Dict[str, Any]]], 
        k: int = 60, 
        top_n: int = 50
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (Industry Standard)"""
        score_dict: Dict[str, float] = {}
        item_map = {}

        for rank_list in lists:
            for rank, item in enumerate(rank_list):
                key = item["text"][:200]
                item_map[key] = item
                score_dict[key] = score_dict.get(key, 0.0) + 1.0 / (rank + k)

        # Sort by RRF score
        sorted_keys = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

        fused = []
        for key, rrf_score in sorted_keys:
            if key in item_map:
                result = item_map[key].copy()
                result["score"] = float(rrf_score)
                result["rrf_score"] = float(rrf_score)
                fused.append(result)

        return fused

    # ====================== PUBLIC SEARCH API ======================

    def search(
        self,
        query_vec: np.ndarray,
        query_text: Optional[str] = None,
        top_k: int = TOP_K,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        # Semantic search (always performed)
        semantic_results = self._semantic_search(query_vec, k=top_k * 4)

        if not use_hybrid or not query_text:
            return semantic_results[:top_k]

        # BM25 + RRF Hybrid
        bm25_results = self._bm25_search(query_text, k=top_k * 4)

        fused_results = self._reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            k=60,
            top_n=top_k * 3
        )

        logger.debug(f"Hybrid RRF: {len(semantic_results)} semantic + "
                    f"{len(bm25_results)} BM25 → {len(fused_results)} fused")

        return fused_results[:top_k]

    # ====================== PERSISTENCE ======================

    def save(self):
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(f"Vector store saved ({len(self.records)} documents)")

    @classmethod
    def load(cls, store_dir: str) -> "FaissVectorStore":
        store_dir = Path(store_dir)
        index_path = store_dir / "index.faiss"
        meta_path = store_dir / "meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index not found in {store_dir}")

        index = faiss.read_index(str(index_path))
        obj = cls(dim=index.d, store_dir=str(store_dir))
        obj.index = index

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj.records.append(json.loads(line))

        obj._build_bm25()
        logger.info(f"Loaded {obj.index.ntotal} vectors with BM25 index")
        return obj
    
    
    def delete_by_file_hash(self, file_hash: str) -> int:
        """Delete document and rebuild indexes"""
        if not file_hash:
            return 0

        kept_records = [
            rec for rec in self.records 
            if rec.get("metadata", {}).get("file_hash") != file_hash
        ]

        deleted_count = len(self.records) - len(kept_records)

        if deleted_count == 0:
            logger.info(f"No chunks found for file_hash: {file_hash}")
            return 0

        self.records = kept_records

        # Rebuild FAISS
        if self.records:
            try:
                from src.embedder import HFEmbedder
                from src.config import EMBED_MODEL
                
                embedder = HFEmbedder(model_name=EMBED_MODEL)
                texts = [r["text"] for r in self.records]
                embeddings = embedder.embed_texts(texts)

                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(embeddings.astype("float32"))
                logger.info(f"Rebuilt FAISS with {len(self.records)} chunks")
            except Exception as e:
                logger.error(f"FAISS rebuild failed: {e}")
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        # Rebuild BM25
        self._build_bm25()

        logger.info(f"✅ Deleted document {file_hash} ({deleted_count} chunks)")
        return deleted_count
    
    def get_all_documents(self) -> List[Dict]:
        """Return list of unique documents with metadata"""
        from collections import defaultdict
        docs = defaultdict(lambda: {
            "file_hash": "",
            "filename": "",
            "chunk_count": 0,
            "uploaded_at": ""
        })

        for rec in self.records:
            meta = rec["metadata"]
            fhash = meta.get("file_hash")
            if not fhash:
                continue
                
            if not docs[fhash]["file_hash"]:
                docs[fhash]["file_hash"] = fhash
                docs[fhash]["filename"] = meta.get("source_file", "unknown.pdf")
                docs[fhash]["uploaded_at"] = meta.get("uploaded_at", "N/A")
            
            docs[fhash]["chunk_count"] += 1

        return list(docs.values())