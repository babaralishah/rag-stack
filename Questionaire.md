### **RAG Project Interview Q&A – 30 Questions with Model Answers**

**1. Tell me about your RAG project.**  
I built a complete end-to-end Retrieval-Augmented Generation (RAG) application. Users can upload PDF documents, the system automatically indexes them, and then answers user questions based on the uploaded content. It uses FastAPI as backend, Streamlit as frontend, FAISS as vector store, sentence-transformers for embeddings, and Groq (Llama-3.3-70B) as the LLM.

**2. What is RAG and why did you build this project?**  
RAG stands for Retrieval-Augmented Generation. It combines document retrieval with LLM generation to reduce hallucinations and provide answers grounded in user-provided documents. I built this project to gain hands-on experience with the full RAG pipeline from ingestion to generation and to understand production deployment challenges.

**3. What are the main components of your RAG system?**  
The main components are: document loader, chunker, embedder, vector store (FAISS), RAG pipeline with guardrail, and the LLM caller (Groq). Frontend is Streamlit and backend is FastAPI.

**4. Walk me through the end-to-end flow when a user uploads a PDF and asks a question.**  
Upload flow: PDF → `document_loader` extracts text → `chunker` splits into overlapping chunks → `embedder` creates vectors → `vector_store` saves vectors and metadata in FAISS.  
Query flow: Question → embedded → searched in FAISS → `rag_pipeline` decides relevance → relevant chunks sent to Groq LLM → final answer + sources returned to UI.

**5. How does the document upload and indexing flow work?**  
User uploads PDF via Streamlit → FastAPI receives it → `load_pdf()` extracts text page by page → `chunk_text()` creates overlapping chunks → `embed_texts()` generates vectors → `FaissVectorStore.add()` stores them with metadata.

**6. Explain the chunking strategy you used.**  
I used a simple character-based chunker with `chunk_size=800` and `chunk_overlap=150`. This ensures context is preserved between chunks and prevents information loss at boundaries.

**7. What embedding model did you use and why?**  
I used `sentence-transformers/all-MiniLM-L6-v2`. It is lightweight, fast on CPU, produces 384-dimensional embeddings, and gives good semantic quality for general English text.

**8. How does FAISS work in your project?**  
FAISS stores only the normalized embedding vectors. I keep the actual text chunks and metadata in a parallel Python list (`self.records`). When searching, FAISS returns indices, which I use to fetch the corresponding text and metadata.

**9. What is cosine similarity and why do we normalize embeddings?**  
Cosine similarity measures how similar two vectors are in direction. We normalize embeddings (make their length = 1) so that cosine similarity becomes equivalent to inner product, which FAISS can compute very efficiently.

**10. Explain the role of `min_score` in your `rag_pipeline.py`.**  
`min_score` (set to 0.35) acts as a guardrail. If the highest similarity score of retrieved chunks is below this threshold, the system does not trust the documents and falls back to general knowledge answer while clearly stating that it is not from the uploaded PDFs.

**11. How do you handle the case when relevant documents are not found?**  
If no chunks are retrieved or the top score is below `min_score`, the system returns: “I don’t have sufficient information in the uploaded documents…” and then provides a general knowledge answer from the LLM.

**12. Why did you choose FastAPI for backend and Streamlit for frontend?**  
FastAPI is modern, fast, and excellent for building REST APIs. Streamlit allows rapid development of interactive UI with very little code, which was perfect for the frontend of this RAG demo.

**13. Why do you use two containers locally but one on Hugging Face?**  
Locally I use two containers (API + UI) for better development experience, faster reload, and easier debugging. On Hugging Face, only one container is allowed, so I combined both using `start.sh`.

**14. How does communication happen between UI and API locally?**  
The UI container calls the API using the Docker service name `http://api:8000`. The environment variable `API_BASE_URL` is set to this value in `docker-compose.yml`.

**15. What is the difference between your local and production setup?**  
Locally: 2 containers, live reload, volume mounts.  
Production (HF): 1 container, no reload, no volume mounts, runs on port 7860, uses `start.sh` to run both services.

**16. How do you manage secrets like `GROQ_API_KEY`?**  
Locally I store it in a `.env` file (ignored by Git). On Hugging Face I add it as a Secret in Space Settings. The code reads it using `os.getenv()`.

**17. What would you do if the retrieved chunks are not relevant but the LLM still tries to answer?**  
I already prevent this using the `min_score` guardrail. If the score is low, the system explicitly tells the user that it doesn’t have enough information from the documents before giving a general knowledge answer.

**18. How would you improve the retrieval quality?**  
I would add reranking, hybrid search (keyword + semantic), query rewriting, and metadata filtering (e.g., by file or page).

**19. What are some limitations of your current RAG implementation?**  
Limited to English, basic chunking strategy, no advanced retrieval, no proper evaluation metrics, and no user authentication.

**20. How would you add support for multiple languages?**  
I would use a multilingual embedding model (like `paraphrase-multilingual-mpnet-base-v2`) and improve the prompt to handle Arabic/Urdu responses.

**21. If the app gets slow with many PDFs, what optimizations would you make?**  
I would implement caching, use a faster embedding model, add indexing optimizations in FAISS, or switch to a managed vector database like Pinecone.

**22. How do you prevent hallucination in your system?**  
Through the guardrail (`min_score`), clear system prompt instructions, and hybrid fallback logic that forces the model to be transparent when documents are insufficient.

**23. What guardrails have you implemented?**  
I implemented a similarity-based guardrail using `min_score = 0.35`. Other possible guardrails could include toxicity filters, answer faithfulness checks, and length control.

**24. How would you implement this using LangChain?**  
I would use `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `FAISS` wrapper, and `RetrievalQA` chain from LangChain to reduce boilerplate code.

**25. Explain your Docker setup.**  
Locally I use `docker-compose.yml` with two services (API and UI). For production on Hugging Face, I use a single `Dockerfile` that runs both services via `start.sh`.

**26. How does your CI/CD pipeline work?**  
I have a GitHub Actions workflow that runs on push to `ci-cd` branch. It builds both Docker images and validates files (CI). CD part is currently a placeholder and can be extended to deploy to HF.

**27. How is your project deployed on Hugging Face?**  
The Space is linked to my GitHub repository. It automatically builds using the root `Dockerfile` whenever I push to the connected branch.

**28. What challenges did you face while deploying on Hugging Face?**  
Hugging Face only supports one container, so I had to combine FastAPI and Streamlit using `start.sh`. Also, managing secrets and ensuring everything runs on port 7860 were key challenges.

**29. What was the biggest challenge you faced?**  
Making the system honest — i.e., not forcing the LLM to answer from weak context. I solved it by implementing a proper guardrail and hybrid response logic.

**30. What did you learn from building this project?**  
I learned the full RAG pipeline, importance of guardrails, Docker deployment differences between local and production, and how to manage secrets securely.