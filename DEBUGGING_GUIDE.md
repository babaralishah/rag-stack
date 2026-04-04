```markdown
# Debugging Guide - Local RAG App (Dry Run)

This guide helps you manually test each component to understand and debug the code.

### 1. Setup for Dry Run

```bash
# Activate virtual environment
.venv\Scripts\activate     # Windows
source .venv/bin/activate   # Linux/Mac

# Make sure GROQ_API_KEY is set
echo $GROQ_API_KEY          # Should show your key
```

### 2. Test Individual Components (Dry Run)

#### Step A: Document Loader
```python
from src.document_loader import load_pdf

pages = load_pdf("data/uploads/your_file.pdf")
print(len(pages))
print(pages[0].text[:500])        # First 500 chars of page 1
print(pages[0].metadata)
```

#### Step B: Chunker
```python
from src.chunker import chunk_text

page_dicts = [{"text": p.text, "metadata": p.metadata} for p in pages]
chunks = chunk_text(page_dicts, chunk_size=800, chunk_overlap=150)

print(f"Created {len(chunks)} chunks")
print(chunks[0].text[:300])
```

#### Step C: Embedder
```python
from src.embedder import HFEmbedder

embedder = HFEmbedder()
texts = [c.text for c in chunks[:5]]
embeddings = embedder.embed_texts(texts)
print(embeddings.shape)        # Should be (5, 384)
```

#### Step D: Vector Store
```python
from src.vector_store import FaissVectorStore

store = FaissVectorStore(dim=384, store_dir="storage/faiss")
store.add(embeddings, texts, [c.metadata for c in chunks[:5]])
store.save()

# Test search
query_vec = embedder.embed_query("What is RAG?")
results = store.search(query_vec, top_k=3)
print(results[0]["score"])
```

#### Step E: RAG Pipeline (Most Important)
```python
from src.rag_pipeline import rag_answer

out = rag_answer(
    question="What is RAG?",
    retrieved=results,
    min_score=0.35
)
print(out["answer"])
print("Sources:", len(out["sources"]))
```

#### Step F: LLM Call (hosted_llm)
```python
from src.hosted_llm import generate_answer

answer = generate_answer("Explain RAG in one sentence.")
print(answer)
```

---

### 2. Flowcharts (Text Version)

#### Upload / Indexing Flow

```
User uploads PDF in Streamlit
          ↓
     api.py → upload_pdf()
          ↓
   document_loader.py → load_pdf() 
          ↓
      chunker.py → chunk_text()
          ↓
     embedder.py → embed_texts()
          ↓
   vector_store.py → store.add() + store.save()
          ↓
   FAISS index + meta.jsonl saved to disk
```

#### Query Flow

```
User types question + clicks Ask
          ↓
     api.py → query()
          ↓
   embedder.py → embed_query()
          ↓
   vector_store.py → store.search()
          ↓
     rag_pipeline.py → rag_answer()
          ↓
   Decision:
   ├── High similarity → Build context + call LLM
   └── Low similarity → "I don't have sufficient information..." + general knowledge
          ↓
   hosted_llm.py → generate_answer() (Groq)
          ↓
   Return answer + sources → Streamlit UI
```