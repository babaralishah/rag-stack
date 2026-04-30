import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger("reranker")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Reranker loaded on {self.device}")

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """
        Rerank retrieved chunks using cross-encoder.
        
        Args:
            query: User question
            candidates: List of retrieved chunks from FAISS (each has 'text', 'score', 'metadata')
            top_k: Number of top results to return after reranking
        """
        if not candidates:
            return []

        # Prepare pairs for cross-encoder: (query, document)
        pairs = [(query, cand["text"]) for cand in candidates]

        # Get relevance scores
        with torch.no_grad():
            scores = self.model.predict(pairs)   # Returns numpy array of scores

        # Add reranker score and sort
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        # Sort by reranker score (descending)
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Keep only top_k
        return reranked[:top_k]


# Global singleton (similar to your embedder)
reranker: Optional[CrossEncoderReranker] = None

def get_reranker() -> CrossEncoderReranker:
    global reranker
    if reranker is None:
        from src.config import RERANKER_MODEL
        reranker = CrossEncoderReranker(model_name=RERANKER_MODEL)
    return reranker