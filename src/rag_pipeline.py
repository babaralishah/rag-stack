import logging
from typing import List, Dict, Any
from src.config import MAX_CHARS, USE_RERANKER, RERANKER_KEEP_TOP_K
from src.hosted_llm import generate_answer
from src.reranker import get_reranker   # ← New import

logger = logging.getLogger(__name__)

def build_context(results: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
    parts = []
    total = 0
    for r in results:
        meta = r["metadata"]
        header = f"[source: {meta.get('source_file')} p.{meta.get('page')}]"
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
    use_reranker: bool = USE_RERANKER,      # ← New parameter with default
) -> Dict[str, Any]:
    
    # === Re-ranking Step ===
    if use_reranker and retrieved:
        try:
            reranker = get_reranker()
            logger.info(f"Re-ranking {len(retrieved)} retrieved chunks")
            retrieved = reranker.rerank(
                query=question, 
                candidates=retrieved, 
                top_k=RERANKER_KEEP_TOP_K
            )
            logger.info(f"After reranking, keeping top {len(retrieved)} chunks")
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Falling back to original retrieval.")
            # Continue with original retrieved chunks if reranker fails

    # Case 1: No results retrieved at all
    if not retrieved:
        general_answer = generate_answer(f"Answer this question concisely using your general knowledge: {question}")
        return {
            "answer": f"I don't have any documents about this topic.\n\n"
                      f"However, based on my general knowledge:\n{general_answer}",
            "sources": [],
        }

    # Case 2: Context exists
    context = build_context(retrieved)
    top_score = retrieved[0].get("score", 0.0)          # original embedding score
    top_rerank_score = retrieved[0].get("rerank_score", None)

    prompt = f"""You are a careful, honest, and helpful assistant.

Use the provided context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- If the context contains enough relevant information, answer **directly** using the context.
- If the context does **NOT** contain enough information to answer accurately:
  1. First say: "I don't have sufficient information in the uploaded documents to answer this question accurately."
  2. Then, you **may** add: "However, based on my general knowledge, ..." and provide your best answer.

Be transparent. Clearly separate information coming from the documents vs your general knowledge.
Do not hallucinate facts from the documents.

ANSWER:"""

    answer = generate_answer(prompt)

    # Prepare sources (show top 3) - Improved for reranking
    seen = set()
    sources = []
    for r in retrieved[:3]:                    # Already reranked, so top ones first
        file = r["metadata"].get("source_file", "unknown")
        page = r["metadata"].get("page", "?")
        key = (file, page)
        if key in seen:
            continue
        seen.add(key)
        
        sources.append({
            "score": round(r.get("score", 0.0), 3),           # original embedding score
            "rerank_score": round(r.get("rerank_score", 0.0), 3) if r.get("rerank_score") is not None else None,
            "file": file,
            "page": page,
            "snippet": r["text"][:280].replace("\n", " "),
        })

    return {"answer": answer, "sources": sources}