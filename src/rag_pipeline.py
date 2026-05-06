import logging
from typing import List, Dict, Any
from src.config import MAX_CHARS, USE_RERANKER, RERANKER_KEEP_TOP_K, RERANKER_FUSION_ALPHA
from src.hosted_llm import generate_answer
from src.reranker import get_reranker

logger = logging.getLogger(__name__)

def build_context(results: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
    parts = []
    total = 0
    for r in results:
        meta = r["metadata"]
        header = f"[source: {meta.get('source_file', 'unknown')} p.{meta.get('page', '?')}]"
        text = r["text"].strip().replace("\n", " ")
        block = f"{header}\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def rag_answer(
    question: str,
    retrieved: List[Dict[str, Any]],
    min_score: float = 0.35,
    use_reranker: bool = True,
    final_top_k: int = 5,
) -> Dict[str, Any]:
    
    # === Reranking Step ===
    if use_reranker and retrieved:
        try:
            reranker = get_reranker()
            logger.info(f"✅ Re-ranking ENABLED - processing {len(retrieved)} chunks")
            
            retrieved = reranker.rerank(
                query=question,
                candidates=retrieved,
                top_k=final_top_k,
                fusion_alpha=RERANKER_FUSION_ALPHA
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to original scores.")
            for r in retrieved:
                r["final_score"] = r.get("score", 0.0)
    else:
        logger.info("⛔ Re-ranking DISABLED by user")
        for r in retrieved:
            r["final_score"] = r.get("score", 0.0)
            r["rerank_score"] = 0.0

    # Limit to requested number
    retrieved = retrieved[:final_top_k]

    # Case 1: No results
    if not retrieved:
        general_answer = generate_answer(f"Answer this question concisely using your general knowledge: {question}")
        return {
            "answer": f"I don't have any documents about this topic.\n\n"
                      f"However, based on my general knowledge:\n{general_answer}",
            "sources": [],
        }

    # Build context
    context = build_context(retrieved)

    # === Improved Prompt with Few-Shot ===
    prompt = f"""You are a careful, honest, and helpful assistant. 
Answer questions using **only** the provided context.

**Instructions:**
- If the context has enough information, answer directly from it.
- If the context does NOT have enough information, say: "I don't have sufficient information in the uploaded documents to answer this accurately."
- Do not hallucinate or make up facts.
- Be clear, concise, and professional.

**Examples:**

Question: What is the main topic of this document?
Context: [Document talks about building RAG systems using FastAPI and FAISS]
Answer: The document is about building a Local Retrieval-Augmented Generation (RAG) application using FastAPI, Streamlit, and FAISS.

Question: What is the person's work experience?
Context: [Resume context mentioning Associate Software Engineer role]
Answer: According to the document, the person worked as Associate Software Engineer from July 2020 to July 2022.

Now answer the real question:

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    answer = generate_answer(prompt)

    # Prepare sources for UI
    sources = []
    for r in retrieved[:3]:
        file = r["metadata"].get("source_file", "unknown")
        page = r["metadata"].get("page", "?")
        sources.append({
            "score": round(r.get("score", 0), 3),
            "rerank_score": round(r.get("rerank_score", 0), 3),
            "final_score": round(r.get("final_score", 0), 4),
            "file": file,
            "page": page,
            "snippet": r["text"][:280].replace("\n", " "),
        })

    return {"answer": answer, "sources": sources}