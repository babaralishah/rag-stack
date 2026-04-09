# test_langchain_rag.py
# Updated test script - loads .env file automatically

import sys
import os

# Add project paths
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

# Load .env file
from dotenv import load_dotenv
load_dotenv()   # This loads GROQ_API_KEY from .env

from langchain_version.embeddings import LangChainEmbedder
from langchain_version.rag_chain import LangChainRAG

import logging

logging.basicConfig(level=logging.INFO)

def test_langchain_rag():
    print("=== Testing LangChain RAG ===")
    
    try:
        # 1. Initialize Embedder
        embedder = LangChainEmbedder()
        print("✅ Embedder loaded successfully")
        
        # 2. Initialize RAG (it will read GROQ_API_KEY from .env)
        rag = LangChainRAG()
        print("✅ RAG initialized successfully")
        
        # 3. Load existing vector store
        rag.load_vectorstore()
        
        # 4. Test question
        question = "What is RAG?"
        print(f"\nTesting question: {question}")
        
        result = rag.get_answer(question)
        
        print("\n=== Answer ===")
        print(result["answer"])
        
        if result.get("sources"):
            print("\n=== Sources ===")
            for i, src in enumerate(result["sources"], 1):
                print(f"{i}. {src.get('file', 'unknown')} (page {src.get('page', '?')})")
        else:
            print("\nNo sources returned.")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_langchain_rag()
