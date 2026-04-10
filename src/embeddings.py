# src/langchain_version/embeddings.py
# LangChain version of embeddings

from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import logging

logger = logging.getLogger(__name__)

class LangChainEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        logger.info("Embedding {len(texts)} texts...")
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        logger.info("Embedding query...")
        return self.embeddings.embed_query(text)
