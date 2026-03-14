import logging
from pathlib import Path

from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embedder import HFEmbedder
from src.config import OLLAMA_MODEL, TOP_K
from src.vector_store import FaissVectorStore
from src.rag_pipeline import rag_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

STORE_DIR = "storage/faiss"
PDF_PATH = "data/sample.pdf"

def ensure_index(embedder: HFEmbedder) -> FaissVectorStore:
    store_path = Path(STORE_DIR)
    if (store_path / "index.faiss").exists() and (store_path / "meta.jsonl").exists():
        return FaissVectorStore.load(STORE_DIR)

    pages = load_pdf(PDF_PATH)
    page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages]
    chunks = chunk_text(page_dicts, chunk_size=600, chunk_overlap=120)

    texts = [c.text for c in chunks]
    metas = [c.metadata for c in chunks]
    embs = embedder.embed_texts(texts)

    vs = FaissVectorStore(dim=embs.shape[1], store_dir=STORE_DIR)
    vs.add(embs, texts, metas)
    vs.save()
    return vs

def main():
    embedder = HFEmbedder()
    vs = ensure_index(embedder)

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            break

        qv = embedder.embed_query(q)
        retrieved = vs.search(qv, top_k=TOP_K)

        out = rag_answer(
            question=q,
            retrieved=retrieved,
            min_score=0.35,
            model=OLLAMA_MODEL,
        )

        print("\n=== ANSWER ===")
        print(out["answer"])

        print("\n=== SOURCES ===")
        for i, s in enumerate(out["sources"], 1):
            print(f"[{i}] score={s['score']:.3f} | {s['file']} p.{s['page']}")
            print(s["snippet"], "\n")

if __name__ == "__main__":
    main()