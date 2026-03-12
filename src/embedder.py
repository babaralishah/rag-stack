# Converts text → numbers (vectors) using a Hugging Face SentenceTransformer model. The vectors are normalized to unit length for cosine similarity search in Faiss.


from typing import List
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class HFEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")