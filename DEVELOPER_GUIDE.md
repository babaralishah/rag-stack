# Developer Guide

## Project Overview

This Local RAG app is a retrieval-augmented generation system built with:

- `FastAPI` backend (`src/api.py`) for ingestion, query routing, document management, and caching.
- `Streamlit` frontend (`ui.py`) for uploading sources, ingesting URLs/YouTube/SQLite, and asking questions.
- `FAISS` vector store (`src/vector_store.py`) for semantic retrieval.
- `sentence-transformers/all-MiniLM-L6-v2` embeddings via `src/embedder.py`.
- Optional BM25 hybrid retrieval and reciprocal rank fusion.
- LLM answer generation via `src/hosted_llm.py` using Groq and/or Google Gemini.

## Architectural Flow

1. **Ingestion**
   - User uploads a file or ingests a URL/YouTube link/SQLite table.
   - The backend normalizes the text into pages and chunks.
   - Embeddings are generated and stored in FAISS.
   - Metadata is persisted alongside the vectors.

2. **Query**
   - The user asks a question in the UI.
   - The backend rewrites the question for retrieval when appropriate.
   - The query is embedded and used for semantic retrieval.
   - If hybrid search is enabled, BM25 results are merged with semantic results.
   - If reranking is enabled, a cross-encoder reranker sorts candidates.
   - The top chunks are assembled into a context-only prompt.
   - The LLM returns a grounded answer with associated sources.

## Core Modules

### `src/api.py`

- Exposes ingestion endpoints: `/upload`, `/ingest/url`, `/ingest/youtube`, `/ingest/text`, `/ingest/sqlite`, `/ingest/sqlite/tables`.
- Supports query routing through `/query`.
- Handles document listing and deletion via `/documents`.
- Uses `src/cache.py` to avoid repeated work.

### `src/source_loader.py`

- Extracts text from web pages, YouTube transcripts, and SQLite tables.
- Normalizes text and attaches source metadata.
- Contains helper functions for validating URLs and table names.

### `src/chunker.py`

- Splits long text into overlapping chunks.
- Preserves page metadata for source tracking.
- Ensures forward progress while maintaining overlap between chunks.

### `src/embedder.py`

- Wraps `SentenceTransformer` and lazily loads the model.
- Provides `embed_query` and `embed_texts` helpers.

### `src/vector_store.py`

- Manages FAISS index operations and persistence.
- Builds optional BM25 indexes when `rank_bm25` is available.
- Supports semantic search alone or hybrid search with reciprocal rank fusion.
- Includes document deletion and rebuild logic.

### `src/rag_pipeline.py`

- Builds the answer prompt from retrieved chunks.
- Optionally reranks candidates before prompt construction.
- Generates grounded answers with strict instructions.

### `src/hosted_llm.py`

- Unified wrapper for external LLM providers.
- Defaults to Groq and optionally supports Gemini if `google-genai` is installed.
- Uses environment variables `GROQ_API_KEY` and `GEMINI_API_KEY`.

### `src/cache.py`

- Implements simple in-memory TTL caches for embeddings and query responses.
- Provides metrics such as hit rate and cache sizes.

### `src/config.py`

- Central source of constants and default settings.
- Includes model names, chunking settings, and store paths.

## Environment and Optional Dependencies

- Required:
  - `fastapi`, `streamlit`, `sentence-transformers`, `faiss`, `pydantic`, `requests`, `pypdf`, `cachetools`
- Optional:
  - `rank_bm25` for BM25 hybrid search
  - `beautifulsoup4` for web page ingestion
  - `youtube-transcript-api` for YouTube transcript ingestion
  - `google-genai` for Gemini integration

## Running the App

```bash
python -m streamlit run ui.py
```

The FastAPI backend is expected to run locally on port `8000` by default.

## Notes for Developers

- The backend caches query results by question, hybrid toggle, reranker toggle, and `top_k`.
- `top_k` is the final number of chunks used in the prompt, while reranking may fetch more candidates first.
- `rank_bm25` is optional. If it is missing, BM25-based retrieval is skipped gracefully.
- If the vector store is missing, the app will report "No documents indexed yet." until ingestion occurs.

## Formatting and Linting

- This project uses `ruff` for lint checks and formatting.
- The codebase was also formatted with `black` for consistent style.
