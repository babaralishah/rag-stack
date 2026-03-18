# Main pipeline for loading a PDF, chunking it, embedding the chunks, and storing them in a FAISS index. 
# Also includes a simple command-line loop for asking questions and retrieving relevant chunks based on cosine similarity search in the vector store.

import logging
from pathlib import Path

from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.config import TOP_K
from src.vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "sample.pdf"
STORE_DIR = BASE_DIR / "storage" / "faiss"


def main():
    store_path = Path(STORE_DIR)
    embedder = HFEmbedder()

    store_path.mkdir(parents=True, exist_ok=True)

    # Load existing index if present
    if (store_path / "index.faiss").exists() and (store_path / "meta.jsonl").exists():
        logging.info("Loading existing FAISS index from %s", store_path)
        vs = FaissVectorStore.load(STORE_DIR)
    else:
        logging.info("No existing index found. Building a new one...")

        pages = load_pdf(PDF_PATH)
        logging.info("Loaded %d pages from PDF", len(pages))

        # Ensure consistent metadata
        page_dicts = []
        for p in pages:
            meta = p.metadata or {}
            page_dicts.append({
                "text": p.text,
                "metadata": {
                    **meta,
                    # Make sure these keys exist for printing later
                    "source_file": meta.get("source_file") or meta.get("source") or Path(PDF_PATH).name,
                    "page": meta.get("page") or meta.get("page_number") or meta.get("page_index"),
                },
            })

        logging.info("Chunking pages...")
        chunks = chunk_text(page_dicts, chunk_size=800, chunk_overlap=150)
        logging.info("Chunking done. Total chunks: %d", len(chunks))

        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]

        logging.info("Embedding %d chunks (this may take a bit on first run)...", len(texts))
        embs = embedder.embed_texts(texts)
        logging.info("Embedding done. Shape: %s", getattr(embs, "shape", None))

        vs = FaissVectorStore(dim=embs.shape[1], store_dir=STORE_DIR)
        vs.add(embs, texts, metas)
        vs.save()

        print("✅ Saved FAISS index to:", STORE_DIR)

    # Retrieval test loop
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        qv = embedder.embed_query(q)
        results = vs.search(qv, top_k=TOP_K)

        print("\n=== TOP CHUNKS ===")
        if not results:
            print("No results.")
            continue

        for i, r in enumerate(results, 1):
            meta = r.get("metadata") or {}
            snippet = (r.get("text") or "")[:300].replace("\n", " ")

            print(f"\n[{i}] score={r['score']:.3f} | {meta.get('source_file')} p.{meta.get('page')}")
            print(snippet, "..." if len(r.get("text") or "") > 300 else "")


if __name__ == "__main__":
    main()