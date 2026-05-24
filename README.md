---
title: RAG LLM App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

## 🎬 Demo

![Local RAG Demo](docs/demo.gif)

# 🧠 Local RAG App (Fully Local, No Paid APIs)

A fully local Retrieval-Augmented Generation (RAG) system built with:

- 🦙 Ollama (local LLM)
- 🤗 HuggingFace embeddings
- ⚡ FAISS vector database
- 🚀 FastAPI backend
- 🎨 Streamlit frontend

No paid APIs required.

---

## 📌 Features

- Upload PDFs
- Automatic chunking with overlap
- Local embedding generation
- Persistent FAISS vector store
- Similarity search (top-k retrieval)
- Guardrail for low-confidence answers
- Evaluation metrics for answer quality (RAGAS-style score, source support, confidence)
- Context-grounded LLM responses
- Source citations (file + page + snippet)
- Clean modular architecture

---

# 🏗 Architecture

## Indexing Phase

1. Upload PDF
2. Extract text (pypdf)
3. Chunk text (overlapping windows)
4. Generate embeddings (sentence-transformers)
5. Store vectors in FAISS
6. Persist index to disk

## Query Phase

1. Embed user question
2. Similarity search (top-k)
3. Guardrail: if similarity too low → “I don’t know”
4. Build context-only prompt
5. Call Ollama
6. Return answer + sources

---

# 📂 Project Structure
local-rag/
│
├── src/
│ ├── api.py
│ ├── chunker.py
│ ├── config.py
│ ├── document_loader.py
│ ├── embedder.py
│ ├── rag_pipeline.py
│ ├── utils.py
│ └── vector_store.py
│
├── scripts/
│ ├── hello_ollama.py
│ ├── index_and_search.py
│ └── rag_chat.py
│
├── data/
│ ├── uploads/
│ └── sample.pdf
│
├── storage/
├── ui.py
├── requirements.txt
├── README.md
└── rag.log


---

# ⚙️ Installation

## 1️⃣ Install Ollama

Download from: https://ollama.com

Pull a model:

```bash
ollama pull llama3.2:3b

## 🐍 Python Version

Tested on Python 3.10+ (recommended 3.10–3.12)

## 💻 System Requirements

- 8GB RAM minimum (16GB recommended)
- Ollama installed and running
- Internet connection for first-time embedding model download

## 🔄 Reset Vector Store

To clear indexed documents:

Delete the storage folder:

```bash
rm -rf storage/    # Mac/Linux
rmdir /s storage   # Windows
## Maintenance Notes

- Optional dependencies: `rank_bm25` is used for BM25 keyword search. If it is not installed the app will still run; BM25-based features will be disabled and a warning is logged.
- To enable full hybrid search, install optional packages:

```bash
pip install rank_bm25 beautifulsoup4 youtube-transcript-api
```

- Quick smoke test (without starting the server):

```bash
# from project root
python -c "from fastapi.testclient import TestClient; import src.api as a; c=TestClient(a.app); print(c.get('/health').json())"
```

- UI notes: the `top_k` slider in `ui.py` controls how many final chunks are returned to the LLM. If re-ranking is enabled the backend fetches more candidates before re-ranking down to `top_k`.

