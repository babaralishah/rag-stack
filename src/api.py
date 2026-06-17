"""FastAPI backend for Local RAG ingestion, retrieval, and document management.

This module exposes endpoints for uploading files, ingesting web/youtube/sqlite
sources, performing queries, retrieving document metadata, deleting sources,
and clearing caches.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
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
    PHASE_FLAGS,
    OLLAMA_MODEL,
)
from src.utils import file_sha256
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer
from src.evaluator import compute_ragas_metrics
from src.advanced_metrics import compute_comprehensive_metrics
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


# web page URLs, YouTube transcripts, and SQL tables.
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
    rewriting_strategy: Literal["none", "keyword_expansion", "hyde"] = "hyde"
    phase: Optional[str] = None
    history: Optional[List[ChatMessage]] = None


def resolve_phase_flags(phase: Optional[str]) -> Dict[str, bool]:
    if not phase:
        return {}

    phase_key = phase.strip().upper()
    if phase_key not in PHASE_FLAGS:
        logger.warning("Unknown phase '%s', falling back to runtime toggles.", phase_key)
        return {}

    return PHASE_FLAGS[phase_key].copy()


def get_query_settings(req: QueryRequest) -> Dict[str, bool]:
    phase_settings = resolve_phase_flags(req.phase)
    if phase_settings:
        return phase_settings

    return {
        "use_query_rewriter": True,
        "use_hybrid": req.use_hybrid,
        "use_reranker": req.use_reranker,
        "use_chat_history": True,
        "use_guardrails": False,
    }


class EvaluationMetrics(BaseModel):
    ragas_score: float
    source_confidence: float
    max_source_score: float
    source_support: float
    source_count: int
    label: str
    warnings: List[str]
    reference_scores: Optional[Dict[str, float]] = None


class StandardMetrics(BaseModel):
    """Standard retrieval and answer quality metrics"""
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    hit_rate: Optional[float] = None
    exact_match: Optional[float] = None
    f1_score: Optional[float] = None


class RAGMetrics(BaseModel):
    """RAG-specific quality metrics"""
    faithfulness: float
    answer_relevance: float
    context_precision: float


class SystemConfiguration(BaseModel):
    """System configuration and parameters used for the query"""
    hardware_info: Optional[str] = None
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_dimensions: int
    temperature: float
    timestamp: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    evaluation: Optional[EvaluationMetrics] = None
    standard_metrics: Optional[StandardMetrics] = None
    rag_metrics: Optional[RAGMetrics] = None
    system_config: Optional[SystemConfiguration] = None


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


# PDFs or text files
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


def rewrite_and_embed_query(
    question: str,
    use_query_rewriter: bool = True,
    history: list | None = None,
    rewriting_strategy: str = "hyde",
) -> tuple[str, Any]:
    """Rewrite the user question (optionally using recent history) and embed it for retrieval."""
    
    logger.info("============== PIPELINE ENTRY ==============")
    logger.info(f"📥 Input Question: '{question}'")
    logger.info(f"📥 Strategy Selected: '{rewriting_strategy}' (Rewriter Enabled: {use_query_rewriter})")
    logger.info(f"📥 History Chunks Received: {history}")
    
    if not use_query_rewriter or rewriting_strategy == "none":
        return question, get_embedder().embed_query(question)

    from src.query_rewriter import rewrite_query

    rewritten = rewrite_query(question, history=history, strategy=rewriting_strategy)
    logger.info(f"📤 Final Search String returning from pipeline: '{rewritten}'")
    logger.info("=============================================")
    query_vector = get_embedder().embed_query(rewritten)
    return rewritten, query_vector


def search_documents(
    store: FaissVectorStore,
    query_text: str,
    query_vector: Any,
    top_k: int,
    use_hybrid: bool,
    use_reranker: bool,
) -> list:
    """Search the vector store with optional hybrid retrieval and reranking candidate count."""
    retrieve_k = RERANKER_TOP_K if use_reranker else top_k
    return store.search(
        query_vec=query_vector,
        query_text=query_text,
        top_k=retrieve_k,
        use_hybrid=use_hybrid,
    )


def generate_answer_payload(
    question: str,
    retrieved: list,
    req: QueryRequest,
    flags: Dict[str, bool],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate the final answer payload from retrieval and reranking with comprehensive metrics."""
    out = rag_answer(
        question=question,
        retrieved=retrieved,
        min_score=MIN_SCORE,
        use_reranker=flags.get("use_reranker", True),
        final_top_k=req.top_k,
        use_guardrails=flags.get("use_guardrails", False),
        use_cot=flags.get("use_cot", False),
        use_few_shot=flags.get("use_few_shot", False),
        history=history,
    )

    answer = out["answer"]
    sources = out.get("sources", [])
    
    # === Compute traditional RAGAS metrics ===
    evaluation = compute_ragas_metrics(
        answer=answer,
        sources=sources,
    )
    
    # === Compute comprehensive standard and RAG metrics ===
    try:
        comprehensive_metrics = compute_comprehensive_metrics(
            question=question,
            answer=answer,
            retrieved=retrieved,
            sources=sources,
            reference=None,
            relevant_document_ids=None,
            top_k=req.top_k,
        )
        
        standard_metrics = StandardMetrics(
            recall_at_k=comprehensive_metrics.get("recall_at_k"),
            mrr=comprehensive_metrics.get("mrr"),
            ndcg_at_k=comprehensive_metrics.get("ndcg_at_k"),
            hit_rate=comprehensive_metrics.get("hit_rate"),
            exact_match=comprehensive_metrics.get("exact_match"),
            f1_score=comprehensive_metrics.get("f1_score"),
        )
        
        rag_metrics = RAGMetrics(
            faithfulness=comprehensive_metrics.get("faithfulness", 0.0),
            answer_relevance=comprehensive_metrics.get("answer_relevance", 0.0),
            context_precision=comprehensive_metrics.get("context_precision", 0.0),
        )
    except Exception as e:
        logger.warning(f"Failed to compute comprehensive metrics: {e}")
        standard_metrics = None
        rag_metrics = None
    
    # === Capture system configuration ===
    try:
        import platform
        import psutil
        
        # Get hardware info
        try:
            cpu_count = psutil.cpu_count(logical=True)
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            hardware_info = f"{cpu_count}x CPU, {ram_gb:.1f}GB RAM ({platform.system()})"
        except Exception:
            hardware_info = platform.system()
        
        system_config = SystemConfiguration(
            hardware_info=hardware_info,
            embedding_model=EMBED_MODEL,
            llm_model=OLLAMA_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            top_k=req.top_k,
            embedding_dimensions=EMBED_DIM,
            temperature=0.3,  # Default temperature used by LLM
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        logger.warning(f"Failed to capture system configuration: {e}")
        system_config = None

    return {
        "answer": answer,
        "sources": sources,
        "evaluation": evaluation,
        "standard_metrics": standard_metrics,
        "rag_metrics": rag_metrics,
        "system_config": system_config,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Handle query requests by validating input, checking cache, searching, and generating answers."""
    question = validate_query_text(req)

    # Resolve phase flags early so we can honor cache and retrieval behavior
    flags = get_query_settings(req)

    # Normalize history only if enabled by phase flags
    history = normalize_chat_history(req.history) if flags.get("use_chat_history") else []
    
      # --- ADD LOGGER HERE ---
    logger.info("📱 --- NEW API INBOUND REQUEST ---")
    logger.info(f"Raw req.question: {req.question}")
    logger.info(f"Normalized History Length: {len(history)} items")
    logger.info(f"Resolved Settings Flags: {flags}")
    logger.info("----------------------------------")

    # Only compute and consult cache when the active phase enables it
    cached_response = None
    rewriting_strategy = (
        req.rewriting_strategy if flags.get("use_query_rewriter", True) else "none"
    )

    if flags.get("use_cache"):
        cache_key = get_cache_key(
            question=question,
            use_hybrid=flags.get("use_hybrid", req.use_hybrid),
            use_reranker=flags.get("use_reranker", req.use_reranker),
            top_k=req.top_k,
            phase=req.phase,
            rewriting_strategy=rewriting_strategy,
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

        # flags already resolved above
        query_text, query_vector = rewrite_and_embed_query(
            question,
            use_query_rewriter=flags.get("use_query_rewriter", True),
            history=history if history else None,  # <-- FIXED: Use history if it exists, regardless of the flag!
            rewriting_strategy=rewriting_strategy,
        )
        retrieved = search_documents(
            store,
            query_text,
            query_vector,
            top_k=req.top_k,
            use_hybrid=flags["use_hybrid"],
            use_reranker=flags["use_reranker"],
        )

        logger.info(
            f"Retrieved {len(retrieved)} chunks, user requested top_k={req.top_k}, "
            f"reranking={'enabled' if flags['use_reranker'] else 'disabled'}"
        )

        result = generate_answer_payload(question, retrieved, req, flags, history=history)
        # Cache only when enabled
        if flags.get("use_cache"):
            try:
                set_cached_query(cache_key, result)
            except Exception:
                logger.warning("Failed to set cache entry; continuing without cache")

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
