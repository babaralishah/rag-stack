import logging
from typing import List, Dict, Any, Optional
from src.config import MAX_CHARS, MIN_SCORE, RERANKER_FUSION_ALPHA, CHAT_HISTORY_TURNS
from src.hosted_llm import generate_answer
from src.reranker import get_reranker

logger = logging.getLogger("rag")


def build_context(results: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
    """
    Build a concatenated context string from retrieved chunks.

    - Preserves source headers and simple metadata per chunk.
    - Stops when `max_chars` is reached to avoid overly long prompts.
    """
    parts = []
    total = 0
    for r in results:
        meta = r["metadata"]
        header = (
            f"[source: {meta.get('source_file', 'unknown')} p.{meta.get('page', '?')}]"
        )
        text = r["text"].strip().replace("\n", " ")
        block = f"{header}\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def build_history_section(
    history: Optional[List[Dict[str, str]]], max_turns: int = CHAT_HISTORY_TURNS
) -> str:
    """Build a simple conversation history section from the last chat turns."""
    if not history:
        return ""

    normalized = [
        {
            "role": str(item.get("role", "")).strip().lower(),
            "content": str(item.get("content", "")).strip(),
        }
        for item in history
        if item.get("content", "").strip()
    ]

    if not normalized:
        return ""

    normalized = normalized[-max_turns:]
    lines = []
    for msg in normalized:
        role = "Assistant" if msg["role"] == "assistant" else "User"
        lines.append(f"{role}: {msg['content']}")

    return "Conversation history:\n" + "\n".join(lines)


def rag_answer(
    question: str,
    retrieved: List[Dict[str, Any]],
    min_score = MIN_SCORE,
    use_reranker: bool = True,
    final_top_k: int = 5,
    use_guardrails: bool = False,
    use_cot: bool = True,
    use_few_shot: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate an answer for `question` using `retrieved` document chunks.

    Steps:
    1. Optionally rerank candidate chunks using a cross-encoder reranker.
    2. Trim to `final_top_k` and build a prompt context.
    3. Call the LLM with strict instructions to use only the provided context.

    Returns a dict with keys: `answer` and `sources`.
    """

    # === Reranking Step ===
    if use_reranker and retrieved:
        try:
            reranker = get_reranker()
            logger.info(f"✅ Re-ranking ENABLED - processing {len(retrieved)} chunks")

            retrieved = reranker.rerank(
                query=question,
                candidates=retrieved,
                top_k=final_top_k,
                fusion_alpha=RERANKER_FUSION_ALPHA,
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

    # Safety limit
    retrieved = retrieved[:final_top_k]

    # Guardrail: abstain if retrieved documents are low-confidence
    if use_guardrails and retrieved:
        max_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
        if max_score < min_score:
            logger.warning(
                "Guardrail triggered: max retrieved score %.3f is below threshold %.3f",
                max_score,
                min_score,
            )
            return {
                "answer": (
                    "I don't have enough high-confidence information from the documents "
                    "to answer that accurately."
                ),
                "sources": [],
            }

    # Case 1: No results
    if not retrieved:
        general_answer = generate_answer(
            f"Answer this question concisely using your general knowledge: {question}"
        )
        return {
            "answer": f"I don't have any documents about this topic.\n\n"
            f"However, based on my general knowledge:\n{general_answer}",
            "sources": [],
        }

    # Build context
    context = build_context(retrieved)
    history_block = build_history_section(history)
    if history_block:
        history_block = history_block + "\n\n"

    # Build the prompt dynamically based on flags
    instructions = [
        "You are a careful, honest, and helpful assistant. Answer using only the provided context.",
        "Use the context to answer the question whenever possible.",
        "Do not hallucinate or invent facts beyond the context.",
    ]

    # If chain-of-thought is requested, add reasoning guidance
    if use_cot:
        instructions.append(
            "Think through the relevant information step by step before giving your final answer."
        )

    # Always optionally mention history usage (history_block is empty if not enabled)
    if history_block:
        instructions.append("Use the conversation history to resolve follow-up questions and references.")

    # Compose the prompt
    prompt_parts = ["\n".join(instructions)]

    # Add few-shot examples only when enabled
    if use_few_shot:
        prompt_parts.append(
            "Examples:\n\nQuestion: What is the main topic of this document?\nContext: [Document about RAG system]\nAnswer: The document is about building a Local RAG application using FastAPI, Streamlit, and FAISS.\n"
        )

    prompt_parts.append("Now answer the real question:")
    prompt_parts.append(f"{history_block}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:")

    prompt = "\n\n".join(prompt_parts)

    answer = generate_answer(prompt, temperature=0.1)  # Low temperature = more grounded

    # Prepare sources
    sources = []
    for r in retrieved:
        file = r["metadata"].get("source_file", "unknown")
        page = r["metadata"].get("page", "?")
        sources.append(
            {
                "score": round(r.get("score", 0), 3),
                "rerank_score": round(r.get("rerank_score", 0), 3),
                "final_score": round(r.get("final_score", 0), 4),
                "file": file,
                "page": page,
                "snippet": r["text"][:280].replace("\n", " "),
            }
        )

    return {"answer": answer, "sources": sources}

