```markdown
# Interview Summary – Local RAG Project

**Project Name:** Local RAG App (Retrieval-Augmented Generation)

**Tech Stack:**
- Backend: FastAPI
- Frontend: Streamlit
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: FAISS
- LLM: Groq (llama-3.3-70b-versatile)
- Containerization: Docker + Docker Compose
- Deployment: Hugging Face Spaces (auto-deploy from GitHub)

**Core Architecture:**
- Two-phase system: Indexing + Querying
- Modular design with clear separation of concerns
- Hybrid response strategy (grounded + general knowledge fallback)

**Key Features Implemented:**
- PDF upload and automatic indexing
- Overlapping chunking strategy
- Cosine similarity search using normalized embeddings
- Relevance guardrail (`MIN_SCORE = 0.35`)
- Transparent fallback when documents are insufficient
- Source citation (file name + page + snippet)
- Persistent vector store

**My Role & Learnings:**
- Built complete end-to-end RAG pipeline from scratch
- Implemented proper error handling and logging
- Designed hybrid prompt strategy for better user experience
- Managed dual deployment (local Docker + HF Spaces)
- Understood trade-offs between local Ollama vs cloud LLM (Groq)

**Why This Project Matters:**
Shows practical understanding of:
- Vector embeddings & semantic search
- RAG architecture & hallucination reduction
- Production considerations (Docker, environment variables, secrets)
- Clean code practices and modular design

**Future Improvements:**
- Add user authentication
- Support multiple file types (docx, txt)
- Implement caching layer
- Add evaluation metrics ( faithfulness, relevance)
```
