# RAG — Upcoming Work

1. **Caching** — Identify and implement caching at appropriate layers to reduce latency and cost
2. **Document Management** — Support listing, viewing, and deleting uploaded documents
3. **Cloud Deployment** — Deploy to a real cloud provider (AWS, GCP, or Azure)
4. **Cost & Effectiveness Analysis** — Evaluate the trade-offs of each feature (e.g. query rewriting, reranking) in terms of performance gains vs. cost
5. **Architecture Approach** — Document whether the current implementation follows a modular or component-based design, and justify the choice
6. **RAG Scope** — Clarify whether the system is built for general-purpose or domain-specific retrieval, and align the design accordingly
7. **LLM Integration** — Integrate Anthropic Claude and OpenAI GPT APIs to gain hands-on experience with both
8. **Conversation Memory** — Maintain chat history with contextual awareness across turns
9. **Evaluation Framework** — Define and implement metrics to assess retrieval and generation quality
10. **Advanced RAG** — Explore agentic patterns, query routing, and other advanced techniques
11. **Custoimization** — Add toggle buttons in the UI for almost every feature, query-rewriting, reranking, hybrid-search, multi-llm 
12. **Latency Optimization** — Do improve the code functions in order to reduce the latecny, try running a few things parallel where possible [Hypothetical Document Embeddings (HyDE)]
13. **Multiple Data sources** — Read data from multile sources, wikipedia, database and others

### RAG focus summary:

   1. Preparation: Slice data smartly (Recursive Chunking) so you search small but read big (Parent-Child Indexing).
   2. Retrieval: Use Hybrid Search (Keywords + Meaning) and a Reranker to find the needle in the haystack.
   3. Verification: Always measure the RAG Triad (Faithfulness, Relevance, and Precision) to prove the answer is a fact, not a hallucination.

The Golden Rule: Search for Context, solve for Intent, and evaluate for Truth.
