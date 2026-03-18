---
title: RAG LLM App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
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
