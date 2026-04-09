# index_with_langchain.py
# One-time script to index your documents into LangChain version

import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from dotenv import load_dotenv
load_dotenv()

from langchain_version.rag_chain import LangChainRAG

# Reuse your original loader and chunker
from document_loader import load_pdf
from chunker import chunk_text

import logging

logging.basicConfig(level=logging.INFO)

def index_documents():
    print("=== Indexing Documents into LangChain RAG ===")
    
    rag = LangChainRAG()
    
    # Change this path if your PDFs are elsewhere
    data_folder = "data"
    
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in '{data_folder}/' folder")
        print("Please put your sample.pdf or other PDFs in the data/ folder")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s). Starting indexing...\n")
    
    total_chunks = 0
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f"→ Processing: {pdf_file}")
        
        # Load PDF using your existing loader
        pages = load_pdf(pdf_path)
        
        # Prepare for chunker
        page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages]
        
        # Chunk using your existing chunker (with overlap)
        chunks = chunk_text(page_dicts, chunk_size=800, chunk_overlap=150)
        
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Add to LangChain vector store
        rag.add_documents(texts=texts, metadatas=metadatas)
        
        total_chunks += len(texts)
        print(f"   Indexed {len(texts)} chunks from {pdf_file}\n")
    
    print(f"✅ Indexing completed!")
    print(f"Total chunks added: {total_chunks}")
    print("You can now run the test script.")

if __name__ == "__main__":
    index_documents()
