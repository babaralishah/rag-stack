import logging
from typing import List
import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag")


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.apply_sigmoid = "bge-reranker" in model_name.lower()
        logger.info(f"Reranker loaded. Apply sigmoid: {self.apply_sigmoid}")

        """Cross-encoder based reranker.

        - Uses a `CrossEncoder` to score (query, candidate) pairs.
        - Optionally applies sigmoid normalization for BGE models.
        - Produces `rerank_score` and `final_score` for each candidate and
            returns the top_k items sorted by `final_score`.
        """

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 6,
        fusion_alpha: float = 0.65,
    ) -> List[dict]:
        if not candidates:
            return []

        pairs = [(query, cand["text"]) for cand in candidates]

        with torch.no_grad():
            raw_scores = self.model.predict(pairs, batch_size=16)

        # Apply sigmoid for BGE models
        if self.apply_sigmoid:
            rerank_scores = 1 / (1 + np.exp(-np.array(raw_scores)))
        else:
            rerank_scores = np.array(raw_scores)

        # Normalize both scores
        rerank_norm = self._normalize(rerank_scores)
        embed_scores = np.array([cand.get("score", 0.5) for cand in candidates])
        embed_norm = self._normalize(embed_scores)

        # Score Fusion
        fused_scores = fusion_alpha * rerank_norm + (1 - fusion_alpha) * embed_norm

        # Attach scores
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(rerank_scores[i])
            cand["final_score"] = float(fused_scores[i])

        # Sort by final fused score
        reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)

        return reranked[:top_k]

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - min_s) / (max_s - min_s)


# Singleton
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from src.config import RERANKER_MODEL

        _reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
    return _reranker
