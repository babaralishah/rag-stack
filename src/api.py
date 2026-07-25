"""FastAPI backend for Local RAG ingestion, retrieval, and document management.

This module exposes endpoints for uploading files, ingesting web/youtube/sqlite
sources, performing queries, retrieving document metadata, deleting sources,
and clearing caches.
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import hashlib
import json
import time
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, Field

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
from src.ragas_framework import compute_ragas_framework_metrics
from src.source_loader import (
    fetch_web_text,
    fetch_youtube_transcript,
    load_sqlite_table,
    get_sqlite_table_names,
)

from agent import (
    run_scanner_pipeline,
    run_deep_dive_pipeline,
    SCORING_FORMULA,
    SCAN_TOP_N,
    SCAN_SCOPE_ALL,
    SCAN_SCOPES,
    get_provider_health_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag.log", encoding="utf-8"),
    ],
)

# Load environment variables early so LLM clients can read API keys.
load_dotenv()

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

SUPPORTED_UPLOAD_EXTENSIONS = {".md", ".txt", ".py", ".js", ".ts", ".json", ".csv"}

app = FastAPI(title="Local RAG API", version="0.1")

SCAN_CACHE_TTL_SECONDS = int(os.getenv("SCAN_CACHE_TTL_SECONDS", "45"))
_scan_cache: dict[str, dict[str, Any]] = {}
DASHBOARD_FILE = Path(__file__).resolve().parents[1] / "web" / "crypto_dashboard.html"


def _fmt_currency(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(amount) >= 1:
        return f"${amount:,.2f}" if abs(amount) < 100 else f"${amount:,.0f}"
    if amount == 0:
        return "$0.00"
    return f"${amount:,.6f}".rstrip("0").rstrip(".")


def _fmt_price(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    digits = 2 if abs(amount) >= 1 else (4 if abs(amount) >= 0.01 else 6)
    return f"${amount:,.{digits}f}"


def _fmt_percent(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{amount:+.2f}%"


def _fmt_compact_currency(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    abs_amount = abs(amount)
    if abs_amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    if abs_amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if abs_amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return _fmt_currency(amount)


def _with_scan_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["display"] = {
            "last_price": _fmt_price(row.get("last_price")),
            "quote_volume": _fmt_compact_currency(row.get("quote_volume")),
            "quote_volume_full": _fmt_currency(row.get("quote_volume")),
            "today_quote_volume": _fmt_compact_currency(row.get("today_quote_volume")),
            "today_quote_volume_full": _fmt_currency(row.get("today_quote_volume")),
            "avg_daily_quote_volume_7d": _fmt_compact_currency(row.get("avg_daily_quote_volume_7d")),
            "avg_daily_quote_volume_7d_full": _fmt_currency(row.get("avg_daily_quote_volume_7d")),
            "spike_ratio": f"{float(row.get('spike_ratio')):.2f}x baseline" if row.get("spike_ratio") is not None else "N/A",
            "volatility_pct": _fmt_percent(row.get("volatility_pct")),
            "recent_15m_range_pct": _fmt_percent(row.get("recent_15m_range_pct")),
            "spread_pct": _fmt_percent(row.get("spread_pct")),
            "depth_notional_0_5pct": _fmt_compact_currency(row.get("depth_notional_0_5pct")),
            "depth_notional_0_5pct_full": _fmt_currency(row.get("depth_notional_0_5pct")),
            "score": f"{float(row.get('score', 0.0)):.2f}",
        }
        rendered.append(item)
    return rendered


def _with_fundamentals_display(fundamentals: dict[str, Any]) -> dict[str, Any]:
    item = dict(fundamentals)
    item["max_supply_label"] = "Uncapped" if item.get("max_supply") in (None, "") else item.get("max_supply")
    item["display"] = {
        "current_price": _fmt_price(item.get("current_price")),
        "market_cap": _fmt_compact_currency(item.get("market_cap")),
        "market_cap_full": _fmt_currency(item.get("market_cap")),
        "volume_24h": _fmt_compact_currency(item.get("volume_24h")),
        "volume_24h_full": _fmt_currency(item.get("volume_24h")),
        "circulating_supply": f"{float(item.get('circulating_supply')):,.0f}" if item.get("circulating_supply") is not None else "N/A",
        "total_supply": f"{float(item.get('total_supply')):,.0f}" if item.get("total_supply") is not None else "N/A",
        "max_supply": "Uncapped" if item.get("max_supply") in (None, "") else f"{float(item.get('max_supply')):,.0f}",
        "ath": _fmt_price(item.get("ath")),
        "atl": _fmt_price(item.get("atl")),
        "fdv_estimated": _fmt_compact_currency(item.get("fdv_estimated")),
        "fdv_estimated_full": _fmt_currency(item.get("fdv_estimated")),
        "pct_from_ath": _fmt_percent(item.get("pct_from_ath")),
        "market_cap_rank": str(int(float(item.get("market_cap_rank")))) if item.get("market_cap_rank") is not None else "N/A",
        "community_score": str(item.get("community_score")) if item.get("community_score") not in (None, 0) else None,
        "developer_score": str(item.get("developer_score")) if item.get("developer_score") not in (None, 0) else None,
    }
    return item


@app.get("/")
def home():
    return {"message": "RAG app is running 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/health")
def api_health():
    """Report provider health, last successful fetches, and circuit-breaker state."""

    return {
        "status": "ok",
        "providers": get_provider_health_snapshot(),
    }


@app.get("/crypto-dashboard")
def crypto_dashboard():
    """Serve the single-page crypto research dashboard."""

    if not DASHBOARD_FILE.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    return FileResponse(str(DASHBOARD_FILE), media_type="text/html")


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
        loaded_store = FaissVectorStore.load(str(STORE_DIR))
        if loaded_store.dim != dim:
            logger.warning(
                "Persisted FAISS dim (%s) does not match active embedding dim (%s). "
                "Starting a fresh in-memory index for the active model.",
                loaded_store.dim,
                dim,
            )
            vs = FaissVectorStore(dim=dim, store_dir=str(STORE_DIR))
            return vs
        vs = loaded_store
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


def _extract_content_signals(
    pages: List[Dict[str, Any]],
    source_type: str,
    source_file: str,
) -> Dict[str, Any]:
    """Build lightweight source signals for UI confirmation cards."""
    first_meta = (pages[0] or {}).get("metadata", {}) if pages else {}
    signals: Dict[str, Any] = {
        "source_type": source_type,
        "source_file": source_file,
    }

    if source_type == "sqlite":
        table_name = first_meta.get("table_name")
        columns = first_meta.get("columns") or []
        signals.update(
            {
                "table_name": table_name,
                "row_count": len(pages),
                "column_count": len(columns),
                "sample_columns": columns[:5],
            }
        )
    elif source_type == "web":
        title = first_meta.get("title")
        if title:
            signals["title"] = title
    elif source_type == "youtube":
        video_id = first_meta.get("video_id")
        if video_id:
            signals["video_id"] = video_id
    else:
        file_extension = first_meta.get("file_extension")
        if file_extension:
            signals["file_extension"] = file_extension

    return signals


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
        # Standardized document identifier used by metrics and downstream code
        page = m.get("page")
        if page is not None and str(page).strip() != "":
            m["doc_id"] = f"{source_hash}:p{page}"
        else:
            m["doc_id"] = f"{source_hash}"
        metas.append(m)

    emb = get_embedder().embed_texts(texts)
    store = load_or_create_store(dim=emb.shape[1])
    store.add(emb, texts, metas)
    store.save()
    clear_all_caches()

    return {
        "status": "ingested",
        "source": source_file,
        "chunks_added": len(texts),
        "content_signals": _extract_content_signals(
            pages=pages,
            source_type=source_type,
            source_file=source_file,
        ),
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_reranker: bool = True
    use_hybrid: bool = True
    use_ragas_framework: bool = False
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


class RAGASFrameworkMetrics(BaseModel):
    """Official RAGAS framework metrics computed in isolated mode."""
    enabled: bool
    status: str
    provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


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
    ragas_framework: Optional[RAGASFrameworkMetrics] = None
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
        # Standardized document identifier for uploaded chunks
        page = m.get("page")
        if page is not None and str(page).strip() != "":
            m["doc_id"] = f"{file_hash}:p{page}"
        else:
            m["doc_id"] = f"{file_hash}"
        metas.append(m)

    emb = get_embedder().embed_texts(texts)
    store = load_or_create_store(dim=emb.shape[1])
    store.add(emb, texts, metas)
    store.save()
    clear_all_caches()

    source_type = "pdf" if file_ext == ".pdf" else "upload"
    return {
        "status": "ingested",
        "file": filename,
        "chunks_added": len(texts),
        "content_signals": {
            "source_type": source_type,
            "source_file": filename,
            "file_extension": file_ext,
            "page_count": len(pages),
        },
    }


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


# def load_search_store() -> FaissVectorStore | None:
#     """Load the vector store when persisted metadata and index exist."""
#     index_path = STORE_DIR / "index.faiss"
#     meta_path = STORE_DIR / "meta.jsonl"
#     if not index_path.exists() or not meta_path.exists():
#         return None
#     return load_or_create_store(dim=EMBED_DIM)

def load_search_store() -> FaissVectorStore | None:
    """Load the vector store when persisted metadata and index exist, else create it."""
    index_path = STORE_DIR / "index.faiss"
    meta_path = STORE_DIR / "meta.jsonl"
    
    # 🛠️ THE FIX: If files are missing, force the system to build them!
    if not index_path.exists() or not meta_path.exists():
        print("⚠️ Database files missing! Building new FAISS index from data source.pdf...")
        # This triggers your app's built-in creation function automatically
        return load_or_create_store(dim=EMBED_DIM)
        
    return load_or_create_store(dim=EMBED_DIM)



def rewrite_and_embed_query(
    question: str,
    use_query_rewriter: bool = True,
    history: list | None = None,
    rewriting_strategy: str = "hyde",
) -> tuple[str, Any, bool]:
    """Rewrite the user question (optionally using recent history) and embed it for retrieval."""
    
    logger.info("============== PIPELINE ENTRY ==============")
    logger.info(f"📥 Input Question: '{question}'")
    logger.info(f"📥 Strategy Selected: '{rewriting_strategy}' (Rewriter Enabled: {use_query_rewriter})")
    logger.info(f"📥 History Chunks Received: {history}")
    
    if not use_query_rewriter or rewriting_strategy == "none":
        return question, get_embedder().embed_query(question), False

    from src.query_rewriter import rewrite_query

    rewritten = rewrite_query(question, history=history, strategy=rewriting_strategy)
    logger.info(f"📤 Final Search String returning from pipeline: '{rewritten}'")
    logger.info("=============================================")
    query_vector = get_embedder().embed_query(rewritten)
    query_was_rewritten = rewritten.strip().lower() != question.strip().lower()
    return rewritten, query_vector, query_was_rewritten


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
    # === Extract retrieved document IDs from the raw retrieved results ===
    retrieved_ids = []
    try:
        for doc in retrieved:
            meta = doc.get("metadata") or {}
            doc_id = meta.get("doc_id") or meta.get("source_file") or meta.get("file") or doc.get("id")
            if doc_id:
                retrieved_ids.append(str(doc_id))
    except Exception:
        retrieved_ids = []
    
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
            relevant_document_ids=retrieved_ids if retrieved_ids else None,
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

    # === Compute official RAGAS framework metrics in isolated mode (optional) ===
    if req.use_ragas_framework:
        try:
            ragas_framework = compute_ragas_framework_metrics(
                question=question,
                answer=answer,
                sources=sources,
            )
        except Exception as e:
            logger.warning(f"Failed to compute RAGAS framework metrics: {e}")
            ragas_framework = {
                "enabled": False,
                "status": "error",
                "provider": None,
                "llm_model": None,
                "embedding_model": None,
                "metrics": {},
                "warnings": ["ragas_framework_exception"],
                "error": str(e),
            }
    else:
        ragas_framework = {
            "enabled": False,
            "status": "unavailable",
            "provider": None,
            "llm_model": None,
            "embedding_model": None,
            "metrics": {},
            "warnings": ["ragas_disabled_by_user"],
            "error": "RAGAS framework is turned off in UI settings.",
        }
    
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
        "ragas_framework": ragas_framework,
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
        query_text, query_vector, _query_was_rewritten = rewrite_and_embed_query(
            question,
            use_query_rewriter=flags.get("use_query_rewriter", True),
            history=history if history else None,  # <-- FIXED: Use history if it exists, regardless of the flag!
            rewriting_strategy=rewriting_strategy,
        )
        if rewriting_strategy != "none" and query_text.strip().lower() == question.strip().lower():
            logger.error(
                "Rewrite collapse detected in /query. strategy=%s question='%s'",
                rewriting_strategy,
                question[:160],
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


# ============================================================================
# EVALUATION ENDPOINT - For automated benchmark/ablation testing
# ============================================================================
class EvalRequest(BaseModel):
    """Evaluation request for retrieval-only (no LLM generation)"""
    query: str
    chunks: int = TOP_K
    rerank: str = "true"
    hybrid: str = "true"
    ablation: Optional[str] = None
    rewriting_strategy: Literal["none", "keyword_expansion", "hyde"] = "none"
    use_query_rewriter: Optional[str] = None
    use_chat_history: Optional[str] = None
    history: Optional[List[ChatMessage]] = None


class EvalResponse(BaseModel):
    """Response containing only retrieved context keys"""
    status: str
    retrieved_context_keys: List[str] = []
    query_was_rewritten: bool = False
    message: Optional[str] = None


def _parse_bool_param(value: str) -> bool:
    """Parse string boolean parameter"""
    return str(value).strip().lower() == "true"


def _resolve_optional_bool(explicit_value: Optional[str], phase_value: Optional[bool], default: bool) -> bool:
    """Resolve booleans with explicit request values taking precedence over phase presets."""
    if explicit_value is not None:
        return _parse_bool_param(explicit_value)
    if phase_value is not None:
        return phase_value
    return default


@app.post("/eval", response_model=EvalResponse)
def eval_retrieval(req: EvalRequest):
    """
    Retrieval-only endpoint for automated evaluation/ablation testing.
    
    Accepts query parameters:
    - query: The search question
    - chunks: Number of results to retrieve (top_k)
    - rerank: "true" or "false" to enable cross-encoder reranking
    - hybrid: "true" or "false" to enable hybrid (BM25 + semantic) search
    - ablation: Optional phase/ablation identifier (e.g., "V1", "V2")
    - rewriting_strategy: "none", "keyword_expansion", or "hyde"
    
    Returns: {
        "status": "success",
        "retrieved_context_keys": ["extracted text block 1", "extracted text block 2", ...]
    }
    """
    try:
        # Validate and parse inputs
        query_text = req.query.strip()
        if not query_text:
            return EvalResponse(status="error", message="Query parameter is empty")
        
        top_k = max(1, min(req.chunks, 50))  # Clamp between 1 and 50
        rewriting_strategy = req.rewriting_strategy.strip().lower()

        # Optional phase/ablation controls for evaluation experiments.
        phase_flags = resolve_phase_flags(req.ablation)

        use_rerank = _resolve_optional_bool(req.rerank, phase_flags.get("use_reranker"), True)
        use_hybrid = _resolve_optional_bool(req.hybrid, phase_flags.get("use_hybrid"), True)
        use_query_rewriter = (
            _parse_bool_param(req.use_query_rewriter)
            if req.use_query_rewriter is not None
            else (rewriting_strategy != "none")
        )
        use_chat_history = _resolve_optional_bool(
            req.use_chat_history,
            phase_flags.get("use_chat_history"),
            False,
        )

        # Explicit request strategy wins over phase presets.
        if not use_query_rewriter:
            rewriting_strategy = "none"

        history = (
            normalize_chat_history(req.history)
            if use_chat_history and req.history
            else None
        )

        logger.info(
            "📊 EVAL ENDPOINT: query='%s...' | chunks=%s | rerank=%s | hybrid=%s | "
            "rewriter=%s | strategy=%s | chat_history=%s | phase=%s",
            query_text[:60],
            top_k,
            use_rerank,
            use_hybrid,
            use_query_rewriter,
            rewriting_strategy,
            use_chat_history,
            req.ablation,
        )
        
        # Load vector store
        store = load_search_store()
        if store is None:
            return EvalResponse(
                status="error",
                message="No indexed documents found. Upload sources first."
            )
        
        # Rewrite query (if strategy != "none") and embed
        query_final, query_vector, query_was_rewritten = rewrite_and_embed_query(
            query_text,
            use_query_rewriter=use_query_rewriter,
            history=history,
            rewriting_strategy=rewriting_strategy,
        )
        if rewriting_strategy != "none" and not query_was_rewritten:
            logger.error(
                "Rewrite collapse detected in /eval. strategy=%s query='%s'",
                rewriting_strategy,
                query_text[:160],
            )
        
        # Retrieve documents
        retrieved = search_documents(
            store,
            query_final,
            query_vector,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranker=use_rerank,
        )
        
        # 🛠️ FIXED: Extract raw text content instead of doc_id for evaluation string comparison
        retrieved_keys = []
                # 🛠️ NATIVE TEXT EXTRACTOR: Bulletproof extraction for both dictionaries and objects
        retrieved_keys = []
        for doc in retrieved:
            text_content = ""
            
            # 1. Try dictionary lookup
            if hasattr(doc, "get"):
                text_content = doc.get("text") or doc.get("page_content") or doc.get("content") or ""
            
            # 2. Try object attribute lookup fallback
            if not text_content:
                text_content = (
                    getattr(doc, "text", None) or 
                    getattr(doc, "page_content", None) or 
                    getattr(doc, "content", "")
                )
            
            # 3. Last resort fallback: Convert the whole thing to a string to catch any text
            if not text_content:
                text_content = str(doc)
                
            if text_content:
                retrieved_keys.append(str(text_content))
        
        logger.info(f"✅ EVAL successfully returned {len(retrieved_keys)} raw text chunks to client.")
        return EvalResponse(
            status="success",
            retrieved_context_keys=retrieved_keys,
            query_was_rewritten=bool(query_was_rewritten),
        )

        
    except Exception as e:
        logger.error(f"❌ EVAL endpoint failed: {e}", exc_info=True)
        return EvalResponse(status="error", message=str(e))


@app.get("/api/scan")
def api_scan(
    top_n: int = Query(default=SCAN_TOP_N, ge=1, le=30),
    scope: str = Query(default=SCAN_SCOPE_ALL),
):
    """Run scanner pipeline and return ranked rows plus LLM summary with short caching."""

    scope_value = scope if scope in SCAN_SCOPES else SCAN_SCOPE_ALL
    now = time.time()
    cache_key = f"{top_n}:{scope_value}"
    cache_bucket = _scan_cache.get(cache_key)
    if cache_bucket is not None and cache_bucket.get("expires_at", 0.0) > now:
        cached_payload = dict(cache_bucket.get("payload") or {})
        cached_payload["stale_badge"] = f"stale - last updated {int(now - cache_bucket.get('cached_at', now))}s ago"
        return cached_payload

    payload: dict[str, Any] = {
        "status": "ok",
        "error": None,
        "rows": [],
        "summary": "",
        "scoring_formula": SCORING_FORMULA,
        "scan_stats": {},
        "pool_audit": {},
        "scope": scope_value,
        "context_row_roundtrip_ok": False,
        "message": "",
        "stale_badge": None,
        "spread_source_hint": "spread_source identifies which market endpoint produced bid/ask spread.",
        "depth_source_hint": "depth_source identifies how near-mid depth notional was computed.",
    }

    try:
        scan = run_scanner_pipeline(top_n=top_n, scope=scope_value)
        payload["rows"] = _with_scan_display(scan.get("candidates", []))
        payload["summary"] = scan.get("summary", "")
        payload["scan_stats"] = scan.get("scan_stats", {})
        payload["pool_audit"] = scan.get("scan_stats", {})
        payload["scope"] = scan.get("scope", scope_value)
        payload["context_row_roundtrip_ok"] = bool(scan.get("context_row_roundtrip_ok", False))
        payload["message"] = scan.get("message", "")
        if scan.get("error"):
            payload["error"] = str(scan.get("error"))
            payload["status"] = "partial"
    except Exception as exc:
        payload["status"] = "partial"
        payload["error"] = f"scanner pipeline failed: {exc}"

    _scan_cache[cache_key] = {"payload": dict(payload), "expires_at": now + SCAN_CACHE_TTL_SECONDS, "cached_at": now}
    return payload


@app.get("/api/deepdive/{symbol_or_name}")
def api_deepdive(symbol_or_name: str):
    """Run deep-dive pipeline and return technical/fundamental/news plus LLM report."""

    payload: dict[str, Any] = {
        "status": "ok",
        "error": None,
        "symbol": symbol_or_name,
        "resolved": {},
        "technicals": {},
        "support_levels": [],
        "resistance_levels": [],
        "fundamentals": {},
        "depth": {},
        "news_status": "unavailable",
        "news_sources_used": [],
        "news_note": None,
        "news": [],
        "llm_report": "",
        "technical_gate": {},
        "fundamentals_note": None,
        "fundamentals_error": None,
        "fundamentals_degraded": False,
    }

    try:
        result = run_deep_dive_pipeline(symbol_or_name)
        research = result.get("research", {}) if isinstance(result, dict) else {}
        if isinstance(research, dict):
            payload["resolved"] = research.get("resolved", {}) or {}
            technicals = research.get("technicals", {}) or {}
            normalized_technicals: dict[str, Any] = {}
            if isinstance(technicals, dict):
                for timeframe, data in technicals.items():
                    snapshot = dict(data) if isinstance(data, dict) else {}
                    if snapshot.get("rsi14") is None:
                        snapshot["trend_label"] = "unknown"
                        snapshot["trend"] = "unknown"
                        snapshot["wick_risk_unusual"] = None
                    snapshot["display"] = {
                        "latest_close": _fmt_price(snapshot.get("latest_close")),
                        "support": _fmt_price(snapshot.get("support")),
                        "resistance": _fmt_price(snapshot.get("resistance")),
                        "rsi14": "N/A" if snapshot.get("rsi14") is None else f"{float(snapshot.get('rsi14')):.2f}",
                        "wick_flag_rate_pct": _fmt_percent(snapshot.get("wick_flag_rate_pct")),
                        "wick_risk": (
                            "High" if snapshot.get("wick_risk_unusual") is True else "Low" if snapshot.get("wick_risk_unusual") is False else "N/A"
                        ),
                    }
                    normalized_technicals[str(timeframe)] = snapshot
            payload["technicals"] = normalized_technicals
            payload["support_levels"] = research.get("support_levels", []) or []
            payload["resistance_levels"] = research.get("resistance_levels", []) or []
            payload["fundamentals"] = _with_fundamentals_display(research.get("fundamentals", {}) or {})
            payload["fundamentals_note"] = research.get("fundamentals", {}).get("fundamentals_note")
            payload["fundamentals_error"] = research.get("fundamentals_error")
            payload["fundamentals_degraded"] = bool(research.get("fundamentals", {}).get("fundamentals_degraded", False))
            payload["depth"] = research.get("depth", {}) or {}
            payload["news_status"] = research.get("news_status", "unavailable") or "unavailable"
            payload["news_sources_used"] = research.get("news_sources_used", []) or []
            payload["news_note"] = research.get("news_note")
            payload["news"] = research.get("news", []) or []
        payload["llm_report"] = result.get("summary", "") if isinstance(result, dict) else ""
        payload["technical_gate"] = result.get("technical_gate", {}) if isinstance(result, dict) else {}
        if isinstance(result, dict) and result.get("error"):
            payload["status"] = "partial"
            payload["error"] = str(result.get("error"))
    except Exception as exc:
        payload["status"] = "partial"
        payload["error"] = f"deep-dive failed: {exc}"

    if payload.get("news_status") not in {"ok", "unavailable"}:
        payload["news_status"] = "ok" if payload.get("news") else "unavailable"
    return payload
