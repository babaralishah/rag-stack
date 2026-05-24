"""FastAPI backend for Local RAG ingestion, retrieval, and document management.

This module exposes endpoints for uploading files, ingesting web/youtube/sqlite
sources, performing queries, retrieving document metadata, deleting sources,
and clearing caches.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, HttpUrl

from src.cache import (
    get_cache_key,
    get_cache_stats,
    get_cached_query,
    clear_all_caches,
    set_cached_query,
)
from src.config import (
    RERANKER_TOP_K,
    UPLOADS_DIR,
    STORE_DIR,
    EMBED_MODEL,
    EMBED_DIM,
    TOP_K,
    MIN_SCORE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHAT_HISTORY_TURNS,
)
from src.utils import file_sha256
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer
from src.evaluator import compute_ragas_metrics
from src.source_loader import (
    fetch_web_text,
    fetch_youtube_transcript,
    load_sqlite_table,
    get_sqlite_table_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

SUPPORTED_UPLOAD_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".csv"}

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
    """Return a shared embedder instance, creating it lazily on first use."""
    global embedder
    if embedder is None:
        embedder = HFEmbedder(model_name=EMBED_MODEL)
    return embedder


def load_or_create_store(dim: int) -> FaissVectorStore:
    """Load an existing FAISS vector store from disk, or create a new one."""
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
    """Return True if the source hash is already present in the stored metadata."""
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
            except Exception:
                continue

    return file_hash in hashes


def compute_source_hash(source_value: str) -> str:
    """Compute a stable SHA256 hash for a source identifier or content snippet."""
    return hashlib.sha256(source_value.encode("utf-8")).hexdigest()


def ingest_pages(
    pages: List[Dict[str, Any]],
    source_file: str,
    source_type: str,
    source_hash: str,
) -> Dict[str, Any]:
    """Convert extracted pages into chunks, embed them, and persist them to FAISS."""
    if not pages:
        raise HTTPException(
            status_code=400, detail="No text content extracted from source."
        )

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


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_reranker: bool = True
    use_hybrid: bool = True
    history: Optional[List[ChatMessage]] = None


class EvaluationMetrics(BaseModel):
    ragas_score: float
    source_confidence: float
    max_source_score: float
    source_support: float
    source_count: int
    label: str
    warnings: List[str]
    reference_scores: Optional[Dict[str, float]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    evaluation: Optional[EvaluationMetrics] = None


def normalize_chat_history(history: Optional[List[ChatMessage]]) -> List[Dict[str, str]]:
    """Validate, normalize, and trim chat history for the query payload."""
    if not history:
        return []

    normalized: List[Dict[str, str]] = []
    for message in history:
        role = message.role.strip().lower()
        if role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail="Chat history roles must be 'user' or 'assistant'.",
            )

        content = message.content.strip()
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized[-CHAT_HISTORY_TURNS:]


def validate_upload_extension(file_ext: str) -> None:
    """Raise an HTTP exception when the uploaded file extension is unsupported."""
    if file_ext != ".pdf" and file_ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported uploads are PDF, markdown, text, JSON, CSV, and code files.",
        )


async def save_upload_file(upload_file: UploadFile) -> tuple[Path, bytes]:
    """Save the uploaded file to disk and return its content bytes."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / upload_file.filename

    try:
        content = await upload_file.read()
        save_path.write_bytes(content)
        return save_path, content
    except Exception as e:
        logger.exception("Failed saving upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")


def parse_upload_file_to_pages(
    file_ext: str, content: bytes, save_path: Path
) -> List[Dict[str, Any]]:
    """Convert uploaded content into page dictionaries for chunking."""
    if file_ext == ".pdf":
        pages = load_pdf(str(save_path))
        if not pages:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found. PDF might be scanned. Use a text-based PDF for now.",
            )
        return [{"text": p.text, "metadata": p.metadata} for p in pages]

    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content.decode("latin-1", errors="replace")

    if not text_content.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded text file is empty or could not be decoded.",
        )

    source_type = (
        "code"
        if file_ext in {".py", ".js", ".ts"}
        else "markdown"
        if file_ext == ".md"
        else "text"
    )

    return [
        {
            "text": text_content,
            "metadata": {
                "source_type": source_type,
                "file_extension": file_ext,
            },
        }
    ]


def ingest_upload_pages(
    pages: List[Dict[str, Any]], file_hash: str, filename: str, file_ext: str
) -> Dict[str, Any]:
    """Chunk, embed, and store uploaded pages with upload-specific metadata."""
    if not pages:
        raise HTTPException(status_code=400, detail="No upload content found.")

    chunks = chunk_text(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    texts = [c.text for c in chunks]
    metas: List[Dict[str, Any]] = []
    for c in chunks:
        m = dict(c.metadata)
        m["file_hash"] = file_hash
        m["source_file"] = filename
        m["source_type"] = "pdf" if file_ext == ".pdf" else m.get("source_type", "text")
        m["uploaded_at"] = datetime.utcnow().isoformat() + "Z"
        metas.append(m)

    emb = get_embedder().embed_texts(texts)
    store = load_or_create_store(dim=emb.shape[1])
    store.add(emb, texts, metas)
    store.save()
    clear_all_caches()

    return {"status": "ingested", "file": filename, "chunks_added": len(texts)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a document, deduplicate, parse content, and index new chunks."""
    file_ext = Path(file.filename).suffix.lower()
    validate_upload_extension(file_ext)

    save_path, content = await save_upload_file(file)

    # Deduplicate uploads by content hash
    fhash = file_sha256(save_path)
    if already_ingested(fhash):
        return {
            "status": "skipped",
            "reason": "File already ingested",
            "file": file.filename,
        }

    try:
        pages = parse_upload_file_to_pages(file_ext, content, save_path)
        return ingest_upload_pages(pages, file_hash=fhash, filename=file.filename, file_ext=file_ext)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload/indexing failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Indexing failed. Check server logs."
        )


@app.post("/ingest/url")
def ingest_url(url: HttpUrl = Form(...)):
    try:
        pages = fetch_web_text(str(url))
        source_hash = compute_source_hash(str(url))
        return ingest_pages(
            pages, source_file=str(url), source_type="web", source_hash=source_hash
        )
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
        return ingest_pages(
            pages, source_file=str(url), source_type="youtube", source_hash=source_hash
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("YouTube ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest YouTube source.")


@app.post("/ingest/text")
def ingest_text(
    content: str = Form(...),
    source_name: str = Form("manual_text"),
    source_type: str = Form("text"),
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    pages = [
        {
            "text": content,
            "metadata": {
                "source_url": source_name,
                "source_type": source_type,
            },
        }
    ]

    source_hash = compute_source_hash(source_name + content[:1000])
    return ingest_pages(
        pages, source_file=source_name, source_type=source_type, source_hash=source_hash
    )


@app.post("/ingest/sqlite")
async def ingest_sqlite(
    file: UploadFile = File(...), table_name: str = Form("user_history")
):
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
        if not table_name.strip():
            detected_tables = get_sqlite_table_names(str(temp_path))
            table_name = detected_tables[0]

        pages = load_sqlite_table(str(temp_path), table_name=table_name)
        source_hash = compute_source_hash(str(file.filename) + table_name)
        result = ingest_pages(
            pages,
            source_file=f"{file.filename}:{table_name}",
            source_type="sqlite",
            source_hash=source_hash,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("SQLite ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to ingest SQLite source.")


@app.post("/ingest/sqlite/tables")
async def list_sqlite_tables(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".db", ".sqlite")):
        raise HTTPException(status_code=400, detail="Upload a .db or .sqlite file.")

    try:
        content = await file.read()
    except Exception as e:
        logger.exception("Failed reading SQLite upload for table discovery: %s", e)
        raise HTTPException(status_code=500, detail="Could not read SQLite upload.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOADS_DIR / file.filename
    temp_path.write_bytes(content)

    try:
        tables = get_sqlite_table_names(str(temp_path))
        return {"tables": tables}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("SQLite table discovery failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to inspect SQLite tables.")


def validate_query_text(req: QueryRequest) -> str:
    """Validate that the request contains a non-empty question."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return question


def get_cached_query_response(cache_key: str) -> QueryResponse | None:
    """Return a cached query response when available."""
    cached = get_cached_query(cache_key)
    if cached is None:
        return None

    logger.info(f"🔄 QUERY CACHE HIT: {cache_key[:70]}...")
    return QueryResponse(
        answer=cached["answer"],
        sources=cached.get("sources", []),
        evaluation=cached.get("evaluation"),
    )


def load_search_store() -> FaissVectorStore | None:
    """Load the vector store when persisted metadata and index exist."""
    index_path = STORE_DIR / "index.faiss"
    meta_path = STORE_DIR / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        return None
    return load_or_create_store(dim=EMBED_DIM)


def rewrite_and_embed_query(question: str) -> tuple[str, Any]:
    """Rewrite the user question and embed it for retrieval."""
    from src.query_rewriter import rewrite_query

    rewritten = rewrite_query(question)
    query_vector = get_embedder().embed_query(rewritten)
    return rewritten, query_vector


def search_documents(
    store: FaissVectorStore, rewritten_query: str, query_vector: Any, req: QueryRequest
) -> list:
    """Search the vector store with optional hybrid retrieval and reranking candidate count."""
    retrieve_k = RERANKER_TOP_K if req.use_reranker else req.top_k
    return store.search(
        query_vec=query_vector,
        query_text=rewritten_query,
        top_k=retrieve_k,
        use_hybrid=req.use_hybrid,
    )


def generate_answer_payload(
    question: str,
    retrieved: list,
    req: QueryRequest,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate the final answer payload from retrieval and reranking."""
    out = rag_answer(
        question=question,
        retrieved=retrieved,
        min_score=MIN_SCORE,
        use_reranker=req.use_reranker,
        final_top_k=req.top_k,
        history=history,
    )

    evaluation = compute_ragas_metrics(
        answer=out["answer"],
        sources=out.get("sources", []),
    )

    return {
        "answer": out["answer"],
        "sources": out.get("sources", []),
        "evaluation": evaluation,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Handle query requests by validating input, checking cache, searching, and generating answers."""
    question = validate_query_text(req)
    history = normalize_chat_history(req.history)
    cache_key = get_cache_key(
        question=question,
        use_hybrid=req.use_hybrid,
        use_reranker=req.use_reranker,
        top_k=req.top_k,
    )
    cached_response = get_cached_query_response(cache_key)
    if cached_response is not None:
        return cached_response

    try:
        store = load_search_store()
        if store is None:
            return QueryResponse(
                answer="No documents indexed yet. Upload at least one source first.",
                sources=[],
            )

        rewritten_query, query_vector = rewrite_and_embed_query(question)
        retrieved = search_documents(store, rewritten_query, query_vector, req)

        logger.info(
            f"Retrieved {len(retrieved)} chunks, user requested top_k={req.top_k}, "
            f"reranking={'enabled' if req.use_reranker else 'disabled'}"
        )

        result = generate_answer_payload(question, retrieved, req, history=history)
        set_cached_query(cache_key, result)

        logger.info(f"✅ Query cached: {question[:60]}...")
        return QueryResponse(**result)
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error")


class EvaluationRequest(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    reference: Optional[str] = None


@app.post("/evaluate", response_model=EvaluationMetrics)
def evaluate_answer(req: EvaluationRequest):
    """Evaluate a RAG answer with support and optional reference comparison."""
    evaluation = compute_ragas_metrics(
        answer=req.answer,
        sources=req.sources,
        reference=req.reference,
    )
    return EvaluationMetrics(**evaluation)


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
