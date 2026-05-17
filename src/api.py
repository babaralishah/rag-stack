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

from src.cache import get_cache_stats, clear_all_caches

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
    
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, HttpUrl

import json

from src.config import RERANKER_TOP_K, UPLOADS_DIR, STORE_DIR, EMBED_MODEL, EMBED_DIM, TOP_K, MIN_SCORE, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import file_sha256
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer
from src.query_rewriter import rewrite_query
from src.source_loader import fetch_web_text, fetch_youtube_transcript, load_sqlite_table

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


def compute_source_hash(source_value: str) -> str:
    return hashlib.sha256(source_value.encode("utf-8")).hexdigest()


def ingest_pages(
    pages: List[Dict[str, Any]],
    source_file: str,
    source_type: str,
    source_hash: str,
) -> Dict[str, Any]:
    if not pages:
        raise HTTPException(status_code=400, detail="No text content extracted from source.")

    page_dicts = [{"text": p["text"], "metadata": p["metadata"]} for p in pages]
    chunks = chunk_text(page_dicts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    texts = [c.text for c in chunks]
    metas = []
    for c in chunks:
        m = dict(c.metadata)
        m["file_hash"] = source_hash
        m["source_file"] = source_file
        m["source_type"] = source_type
        m["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
        metas.append(m)

    emb = get_embedder().embed_texts(texts)
    store = load_or_create_store(dim=emb.shape[1])
    store.add(emb, texts, metas)
    store.save()
    clear_all_caches()

    return {"status": "ingested", "source": source_file, "chunks_added": len(texts)}


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_reranker: bool = True
    use_hybrid: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    supported_text = {".md", ".txt", ".py", ".js", ".ts", ".json", ".csv"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext != ".pdf" and file_ext not in supported_text:
        raise HTTPException(status_code=400, detail="Supported uploads are PDF, markdown, text, and code files.")

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
        if file_ext == ".pdf":
            pages = load_pdf(str(save_path))
            if not pages:
                raise HTTPException(
                    status_code=400,
                    detail="No extractable text found. PDF might be scanned. Use a text-based PDF for now."
                )
        else:
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1", errors="replace")

            if not text_content.strip():
                raise HTTPException(status_code=400, detail="Uploaded text file is empty or could not be decoded.")

            pages = [{
                "text": text_content,
                "metadata": {
                    "source_type": "code" if file_ext in {".py", ".js", ".ts"} else "markdown" if file_ext == ".md" else "text",
                    "file_extension": file_ext,
                }
            }]

        page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages] if file_ext == ".pdf" else [{"text": p["text"], "metadata": p["metadata"]} for p in pages]
        chunks = chunk_text(page_dicts, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        texts = [c.text for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c.metadata)
            m["file_hash"] = fhash
            m["source_file"] = file.filename
            m["source_type"] = "pdf" if file_ext == ".pdf" else m.get("source_type", "text")
            m["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
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


@app.post("/ingest/url")
def ingest_url(url: HttpUrl = Form(...)):
    try:
        pages = fetch_web_text(str(url))
        source_hash = compute_source_hash(str(url))
        return ingest_pages(pages, source_file=str(url), source_type="web", source_hash=source_hash)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("URL ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest URL source.")


@app.post("/ingest/youtube")
def ingest_youtube(url: HttpUrl = Form(...)):
    try:
        pages = fetch_youtube_transcript(str(url))
        source_hash = compute_source_hash(str(url))
        return ingest_pages(pages, source_file=str(url), source_type="youtube", source_hash=source_hash)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("YouTube ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest YouTube source.")


@app.post("/ingest/text")
def ingest_text(
    content: str = Form(...),
    source_name: str = Form("manual_text"),
    source_type: str = Form("text")
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    pages = [{
        "text": content,
        "metadata": {
            "source_url": source_name,
            "source_type": source_type,
        }
    }]

    source_hash = compute_source_hash(source_name + content[:1000])
    return ingest_pages(
        pages,
        source_file=source_name,
        source_type=source_type,
        source_hash=source_hash
    )


@app.post("/ingest/sqlite")
async def ingest_sqlite(file: UploadFile = File(...), table_name: str = Form("user_history")):
    if not file.filename.lower().endswith((".db", ".sqlite")):
        raise HTTPException(status_code=400, detail="Upload a .db or .sqlite file.")

    try:
        content = await file.read()
    except Exception as e:
        logger.exception("Failed reading SQLite upload: %s", e)
        raise HTTPException(status_code=500, detail="Could not read SQLite upload.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / file.filename
    temp_path.write_bytes(content)

    try:
        pages = load_sqlite_table(str(temp_path), table_name=table_name)
        source_hash = compute_source_hash(str(file.filename) + table_name)
        result = ingest_pages(
            pages,
            source_file=f"{file.filename}:{table_name}",
            source_type="sqlite",
            source_hash=source_hash
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("SQLite ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest SQLite source.")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    from src.cache import get_cache_key, get_cached_query, set_cached_query

    # Check Cache
    cache_key = get_cache_key(
        question=q,
        use_hybrid=getattr(req, 'use_hybrid', True),
        use_reranker=getattr(req, 'use_reranker', True),
        top_k=req.top_k
    )

    cached = get_cached_query(cache_key)
    if cached is not None:
        logger.info(f"🔄 QUERY CACHE HIT: {q[:70]}...")
        return QueryResponse(answer=cached["answer"], sources=cached.get("sources", []))

    # Normal processing
    try:
        from src.query_rewriter import rewrite_query
        rewritten_query = rewrite_query(q)

        emb = get_embedder()
        index_path = STORE_DIR / "index.faiss"
        meta_path = STORE_DIR / "meta.jsonl"

        if not index_path.exists() or not meta_path.exists():
            return QueryResponse(answer="No documents indexed yet. Upload at least one source first.", sources=[])

        store = load_or_create_store(dim=EMBED_DIM)
        qv = emb.embed_query(rewritten_query)

        retrieve_k = RERANKER_TOP_K if getattr(req, 'use_reranker', True) else req.top_k

        retrieved = store.search(
            query_vec=qv,
            query_text=rewritten_query,
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
        set_cached_query(cache_key, result)

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
    """List all uploaded documents and sources"""
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