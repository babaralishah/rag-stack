import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")

class HFEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self.model

    def embed_query(self, text: str):
        """Simple cached embedding"""
        if not text:
            return None
            
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding

    def embed_texts(self, texts: list):
        if not texts:
            return None
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True, batch_size=32)