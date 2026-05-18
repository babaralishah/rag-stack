import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")


class HFEmbedder:
    """Light-weight wrapper around `SentenceTransformer` for embeddings.

    - Lazily loads the model on first use to avoid startup cost.
    - Exposes `embed_query` and `embed_texts` helpers used by the pipeline.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self.model

    def embed_query(self, text: str):
        """Return a single query embedding (numpy array).

        Returns `None` for empty input.
        """
        if not text:
            return None

        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding

    def embed_texts(self, texts: list):
        """Embed a list of texts and return numpy array of embeddings.

        Returns `None` for empty input lists.
        """
        if not texts:
            return None
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True, batch_size=32)
