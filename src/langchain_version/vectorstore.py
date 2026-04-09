# src/langchain_version/vectorstore.py
# LangChain version of vector store using FAISS

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class LangChainVectorStore:
    def __init__(self, embeddings: HuggingFaceEmbeddings, store_dir: str = "storage/faiss"):
        self.embeddings = embeddings
        self.store_dir = store_dir
        self.vectorstore = None
        os.makedirs(store_dir, exist_ok=True)
        logger.info(f"Vector store initialized at {store_dir}")

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add texts to the vector store"""
        if not texts:
            return

        logger.info(f"Adding {len(texts)} texts to vector store...")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        else:
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )

        # Save to disk
        self.vectorstore.save_local(self.store_dir)
        logger.info("Vector store saved successfully")

    def similarity_search(self, query: str, k: int = 4):
        """Search for similar documents"""
        if self.vectorstore is None:
            logger.warning("Vector store is empty. No documents indexed yet.")
            return []

        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.0
            })
        return results

    def load(self):
        """Load existing vector store from disk"""
        if os.path.exists(self.store_dir):
            try:
                self.vectorstore = FAISS.load_local(
                    self.store_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded vector store with {self.vectorstore.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
        else:
            logger.info("No existing vector store found. Will create new one.")
