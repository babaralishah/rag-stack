from pathlib import Path
import sys

import numpy as np

# Ensure project root is on sys.path so `src` imports work when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vector_store import FaissVectorStore, BM25Okapi


def test_faiss_cosine_normalization_on_add_and_query(tmp_path: Path):
    store = FaissVectorStore(dim=3, store_dir=str(tmp_path))

    # Intentionally use non-normalized float64 vectors to verify cast+normalization.
    embeddings = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    texts = ["alpha", "beta", "gamma"]
    metas = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]

    store.add(embeddings, texts, metas)

    # Reconstructed vectors should be unit-normalized in the FAISS index.
    for i in range(store.index.ntotal):
        vec = store.index.reconstruct(i)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    # Large-norm query should still behave like cosine after normalization.
    results = store._semantic_search(np.array([100.0, 0.0, 0.0], dtype=np.float64), k=3)
    assert results
    assert results[0]["text"] == "alpha"

    # Cosine/inner-product on normalized vectors is bounded in [-1, 1].
    assert all(-1.0001 <= r["semantic_score"] <= 1.0001 for r in results)


def test_hybrid_rrf_still_returns_fused_results(tmp_path: Path):
    if BM25Okapi is None:
        # BM25 dependency is optional in this project setup.
        return

    store = FaissVectorStore(dim=3, store_dir=str(tmp_path))
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    texts = [
        "aspirin reduces pain and fever",
        "ibuprofen is an anti inflammatory medicine",
        "biomedical evidence synthesis workflow",
    ]
    metas = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}]

    store.add(embeddings, texts, metas)

    results = store.search(
        query_vec=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        query_text="ibuprofen pain",
        top_k=2,
        use_hybrid=True,
    )

    assert len(results) <= 2
    assert len(results) > 0
    assert all("score" in item for item in results)


def test_save_load_1024_dim_index_roundtrip(tmp_path: Path):
    dim = 1024
    store = FaissVectorStore(dim=dim, store_dir=str(tmp_path))

    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(2, dim)).astype(np.float64)
    texts = ["doc one", "doc two"]
    metas = [{"doc_id": "x1"}, {"doc_id": "x2"}]

    store.add(embeddings, texts, metas)
    store.save()

    loaded = FaissVectorStore.load(str(tmp_path))
    assert loaded.dim == dim
    assert loaded.index.d == dim
    assert loaded.index.ntotal == 2

    query = rng.normal(size=(dim,)).astype(np.float64)
    results = loaded.search(query_vec=query, query_text="doc", top_k=1, use_hybrid=False)
    assert len(results) == 1


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        test_faiss_cosine_normalization_on_add_and_query(Path(d))
    with TemporaryDirectory() as d:
        test_hybrid_rrf_still_returns_fused_results(Path(d))
    with TemporaryDirectory() as d:
        test_save_load_1024_dim_index_roundtrip(Path(d))

    print("Vector store cosine + persistence tests passed.")
