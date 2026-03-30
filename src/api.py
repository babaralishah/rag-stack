import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import json

from src.config import UPLOADS_DIR, STORE_DIR, EMBED_MODEL, EMBED_DIM, TOP_K, MIN_SCORE
from src.utils import file_sha256
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api")

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

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

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
        chunks = chunk_text(page_dicts, chunk_size=800, chunk_overlap=150)

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

    emb = get_embedder()
    # Ensure store exists (if no index yet, return safe message)
    index_path = STORE_DIR / "index.faiss"
    meta_path = STORE_DIR / "meta.jsonl"
    if not index_path.exists() or not meta_path.exists():
        return QueryResponse(answer="No documents indexed yet. Upload a PDF first.", sources=[])

    store = load_or_create_store(dim=EMBED_DIM)
    qv = emb.embed_query(q)
    retrieved = store.search(qv, top_k=req.top_k)

    out = rag_answer(
        question=q,
        retrieved=retrieved,
        min_score=MIN_SCORE
    )

    return QueryResponse(answer=out["answer"], sources=out["sources"])