from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from .langchain_version.rag_chain import LangChainRAG

# Reuse your existing loader and chunker for upload
from .document_loader import load_pdf
from .chunker import chunk_text

app = FastAPI(title="LangChain RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
async def startup_event():
    global rag
    print("🚀 Starting LangChain RAG API")
    rag = LangChainRAG()
    rag.load_vectorstore()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        os.makedirs("data/uploads", exist_ok=True)
        temp_path = f"data/uploads/{file.filename}"
        
        with open(temp_path, "wb") as f:
            f.write(content)

        print(f"Processing uploaded file: {file.filename}")

        # Load and chunk using your existing tools
        pages = load_pdf(temp_path)
        page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages]
        chunks = chunk_text(page_dicts, chunk_size=800, chunk_overlap=150)

        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Add to LangChain RAG
        rag.add_documents(texts=texts, metadatas=metadatas)

        return {"message": f"Successfully uploaded and indexed {file.filename}", "chunks": len(texts)}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(question: str):
    try:
        if rag is None:
            return {"answer": "System not initialized yet.", "sources": []}
        
        result = rag.get_answer(question)
        return result

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "LangChain RAG"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

