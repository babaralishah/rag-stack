import logging
from src.cache import cache_embedding

logger = logging.getLogger("rag")

class HFEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        logger.info(f"Loading embedding model: {model_name}")

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self.model

    @cache_embedding
    def embed_query(self, text: str):
        """Embed a single query (cached)"""
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding

    def embed_texts(self, texts: list[str]):
        """Embed multiple texts (used during indexing - less caching needed)"""
        if not texts:
            return None
        model = self._load_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings