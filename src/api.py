import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

from src.cache import cache_query, get_cache_stats, clear_all_caches

from pathlib import Path
from typing import Optional, List, Dict, Any
    
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import json

from src.config import RERANKER_TOP_K, UPLOADS_DIR, STORE_DIR, EMBED_MODEL, EMBED_DIM, TOP_K, MIN_SCORE, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import file_sha256
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer
from src.query_rewriter import rewrite_query

app = FastAPI(title="Local RAG API", version="0.1")

@app.get("/")
def home():
    return {"message": "RAG app is running 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}

# ---- Global singletons (simple + fine for learning)
embedder: Optional[HFEmbedder] = None
vs: Optional[FaissVectorStore] = None

def get_embedder() -> HFEmbedder:
    global embedder
    if embedder is None:
        embedder = HFEmbedder(model_name=EMBED_MODEL)
    return embedder

def load_or_create_store(dim: int) -> FaissVectorStore:
    global vs
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = STORE_DIR / "index.faiss"
    meta_path = STORE_DIR / "meta.jsonl"

    if vs is not None:
        return vs

    if index_path.exists() and meta_path.exists():
        vs = FaissVectorStore.load(str(STORE_DIR))
        return vs

    vs = FaissVectorStore(dim=dim, store_dir=str(STORE_DIR))
    return vs

def already_ingested(file_hash: str) -> bool:
    meta_path = STORE_DIR / "meta.jsonl"
    if not meta_path.exists():
        return False

    hashes = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                fh = record.get("metadata", {}).get("file_hash")
                if fh:
                    hashes.add(fh)
            except:
                continue

    return file_hash in hashes

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_reranker: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported right now.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / file.filename

    try:
        content = await file.read()
        save_path.write_bytes(content)
    except Exception as e:
        logger.exception("Failed saving upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # Hash for dedupe
    fhash = file_sha256(save_path)
    if already_ingested(fhash):
        return {"status": "skipped", "reason": "File already ingested", "file": file.filename}

    try:
        pages = load_pdf(str(save_path))
        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found. PDF might be scanned. Use a text-based PDF for now."
            )

        page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages]
        chunks = chunk_text(page_dicts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        texts = [c.text for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c.metadata)
            m["file_hash"] = fhash
            metas.append(m)

        emb = get_embedder().embed_texts(texts)
        store = load_or_create_store(dim=emb.shape[1])
        store.add(emb, texts, metas)
        store.save()
        clear_all_caches()
        return {"status": "ingested", "file": file.filename, "chunks_added": len(texts)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload/indexing failed: %s", e)
        raise HTTPException(status_code=500, detail="Indexing failed. Check server logs.")

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from src.cache import get_cache_key, query_cache

    # Check Cache
    cache_key = get_cache_key(
        question=q,
        use_hybrid=getattr(req, 'use_hybrid', True),
        use_reranker=getattr(req, 'use_reranker', True),
        top_k=req.top_k
    )

    if cache_key in query_cache:
        logger.info(f"🔄 QUERY CACHE HIT: {q[:70]}...")
        cached = query_cache[cache_key]
        return QueryResponse(answer=cached["answer"], sources=cached.get("sources", []))

    # Normal processing
    try:
        from src.query_rewriter import rewrite_query
        rewritten_query = rewrite_query(q)

        emb = get_embedder()
        index_path = STORE_DIR / "index.faiss"
        meta_path = STORE_DIR / "meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            return QueryResponse(answer="No documents indexed yet. Upload a PDF first.", sources=[])

        store = load_or_create_store(dim=EMBED_DIM)
        qv = emb.embed_query(rewritten_query)

        retrieve_k = RERANKER_TOP_K if getattr(req, 'use_reranker', True) else req.top_k

        retrieved = store.search(
            query_vec=qv,
            top_k=retrieve_k,
            use_hybrid=getattr(req, 'use_hybrid', True)
        )

        out = rag_answer(
            question=q,
            retrieved=retrieved,
            min_score=MIN_SCORE,
            use_reranker=getattr(req, 'use_reranker', True),
            final_top_k=req.top_k
        )

        result = {"answer": out["answer"], "sources": out.get("sources", [])}
        query_cache[cache_key] = result

        logger.info(f"✅ Query cached: {q[:60]}...")
        return QueryResponse(answer=out["answer"], sources=out["sources"])

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error")

class DocumentResponse(BaseModel):
    file_hash: str
    filename: str
    chunk_count: int
    uploaded_at: str

@app.get("/documents", response_model=List[DocumentResponse])
def list_documents():
    """List all uploaded documents"""
    try:
        store = load_or_create_store(dim=EMBED_DIM)
        docs = store.get_all_documents()
        return docs
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []

@app.delete("/documents/{file_hash}")
def delete_document(file_hash: str):
    """Delete a document and its chunks"""
    try:
        store = load_or_create_store(dim=EMBED_DIM)
        deleted = store.delete_by_file_hash(file_hash)
        
        if deleted > 0:
            store.save()
            clear_all_caches()
            return {"status": "success", "message": f"Deleted {deleted} chunks"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.get("/cache/stats")
def get_cache_statistics():
    """Return cache statistics for monitoring"""
    try:
        return get_cache_stats()
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"error": str(e)}

@app.post("/cache/clear")
def clear_cache_endpoint():
    """Manual cache clear"""
    try:
        clear_all_caches(reason="manual_user_request")
        return {"status": "success", "message": "All caches cleared"}
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")