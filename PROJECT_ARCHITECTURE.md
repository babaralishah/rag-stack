---

# Project Architecture – Local RAG App

**Fully functional Retrieval-Augmented Generation (RAG) system** with FastAPI + Streamlit, using Groq LLM.

---

### 1. High-Level Overview

**What is this project?**  
A complete **Retrieval-Augmented Generation (RAG)** application that lets users:

1. Upload PDF documents
2. Automatically index them (chunk + embed + store)
3. Ask questions in natural language
4. Get accurate answers **grounded in their documents** + fallback to general knowledge when needed

**Core Idea of RAG**:  
Instead of asking an LLM directly (which can hallucinate), we first retrieve relevant pieces from the user’s documents, then pass only those pieces to the LLM.

---

### 2. Architecture Diagram (Text Version)

```
User → Streamlit UI (ui.py)
         ↓ (POST /upload or /query)
      FastAPI Backend (api.py)
         ↓
   ┌─────────────────────┐
   │   Indexing Flow     │ ← Upload PDF
   └─────────────────────┘
         ↓
   document_loader.py → chunker.py → embedder.py → vector_store.py (Faiss)

         ↓
   ┌─────────────────────┐
   │   Query Flow        │ ← User question
   └─────────────────────┘
         ↓
   embedder.py → vector_store.py (search) → rag_pipeline.py → hosted_llm.py (Groq)
         ↓
      Return answer + sources → Streamlit UI
```

---

### 3. What Each File Does

| File                    | Purpose                                                                 | Key Functions / Classes                  |
|-------------------------|-------------------------------------------------------------------------|------------------------------------------|
| `ui.py`                 | Frontend (Streamlit) – User interface                                  | Upload PDF, Ask question, Show chat     |
| `api.py`                | Backend API (FastAPI) – Handles upload & query                         | `/upload`, `/query`, `/health`          |
| `config.py`             | All configuration and paths                                            | Paths, constants, environment vars      |
| `document_loader.py`    | Reads PDF and extracts text per page                                   | `load_pdf()`                            |
| `chunker.py`            | Splits long text into smaller overlapping chunks                       | `chunk_text()`                          |
| `embedder.py`           | Converts text → vectors (embeddings)                                   | `HFEmbedder` class                      |
| `vector_store.py`       | Stores and searches vectors using FAISS                                | `FaissVectorStore` class                |
| `rag_pipeline.py`       | Main RAG logic – decides when to use documents vs general knowledge   | `rag_answer()`                          |
| `hosted_llm.py`         | Calls Groq LLM (llama-3.3-70b-versatile)                              | `generate_answer()`                     |
| `utils.py`              | Helper functions                                                       | `file_sha256()`                         |
| `Dockerfile`            | Production image for Hugging Face (single container)                   | Uses `start.sh`                         |
| `Dockerfile.ui`         | Only for local development (Streamlit container)                       | —                                       |
| `docker-compose.yml`    | Local development setup (2 containers)                                 | API + UI                                |
| `start.sh`              | Starts both FastAPI + Streamlit in one container (for HF)              | Background uvicorn + foreground streamlit |

---

### 4. Detailed Data Flow

#### **Upload Flow** (Indexing)
1. User uploads PDF in Streamlit
2. `api.py` → `upload_pdf()`
3. `document_loader.py` → extracts text page by page
4. `chunker.py` → splits into overlapping chunks
5. `embedder.py` → converts chunks to vectors
6. `vector_store.py` → saves vectors + metadata into FAISS + `meta.jsonl`
7. Index is persisted to disk (`storage/faiss/`)

#### **Query Flow**
1. User asks a question
2. `api.py` → `query()`
3. Embed question → search FAISS (`vector_store.search()`)
4. `rag_pipeline.py` → `rag_answer()`
   - If similarity score too low → fallback to general knowledge
   - Else → build context + send prompt to Groq
5. `hosted_llm.py` → calls Groq and gets answer
6. Return answer + sources to UI

---

### 5. Key Concepts Explained

- **Embeddings**: Text is converted into numbers (vectors) so we can measure similarity mathematically.
- **FAISS**: Extremely fast vector database for similarity search.
- **RAG Guardrail**: We check the top similarity score. If too low (`MIN_SCORE = 0.35`), we don’t trust the documents.
- **Hybrid Response**: The model is instructed to first admit when documents are insufficient, then optionally answer from general knowledge.
- **Persistence**: Once you upload PDFs, the vector index is saved to disk so you don’t lose data when restarting.

---

### 6. Docker Setup Explained

- **Local Development** (`docker-compose.yml`):
  - Two separate containers (`api` + `ui`)
  - Live code reloading (`--reload`)
  - Easy debugging

- **Production (Hugging Face)** (`Dockerfile` + `start.sh`):
  - Single container only (HF requirement)
  - `start.sh` runs FastAPI in background + Streamlit in foreground on port 7860

---

### 7. How GitHub + Hugging Face Works Together

1. You push code to GitHub
2. Hugging Face Space is linked to your GitHub repo
3. HF detects `Dockerfile` in root
4. HF automatically builds and deploys the image
5. Every new `git push` → HF rebuilds automatically

That’s why you only needed to add the special header in `README.md`:
```yaml
---
sdk: docker
app_port: 7860
---
```

---
