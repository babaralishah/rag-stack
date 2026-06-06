import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger("rag")


def rewrite_query(original_question: str, history: list | None = None) -> str:
    """
    Advanced Query Rewriter - Makes the question much better for retrieval.
    """
    normalized = original_question.strip()
    if len(normalized.split()) < 8 and (
        "db" in normalized.lower() or "database" in normalized.lower()
    ):
        logger.info(
            "Short SQLite/database query detected; skipping rewrite to preserve intent."
        )
        return original_question

    # Include recent history when available to disambiguate pronouns or references
    history_block = ""
    if history and isinstance(history, list):
        # Keep only the last 3 turns for context
        last_turns = history[-4:]
        history_lines = []
        for m in last_turns:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if content:
                history_lines.append(f"{role.title()}: {content}")
        if history_lines:
            history_block = "Conversation history:\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""You are an expert RAG Query Optimizer. Your job is to rewrite the user's question for retrieval,
while preserving the original meaning exactly and without adding any new topics, entities, or assumptions.

If the user's question is already unambiguous and sufficient for retrieval, simply return it unchanged.

When available, use the recent conversation history to resolve pronouns or references. Use only the last few turns:
{history_block}
Original Question: {original_question}

Rewrite the question with these goals:
- Keep the same meaning and focus as the original question
- Do not invent or add any details, topics, or domain-specific assumptions
- Keep it as ONE single, natural question
- Do not answer the question, only rewrite it

Return only the rewritten question (or the original if no rewrite is needed):
Rewritten Question:"""

    try:
        rewritten = generate_answer(
            prompt=prompt,
            # model="llama-3.1-8b-instant",   # Fast model for rewriting
            model="gemini-2.5-flash-lite",  # gemini-2.5-flash-preview-05-20; gemini-2.5-flash
            temperature=0.3,
            max_tokens=250,
        )

        rewritten = rewritten.strip()

        # Fallback if rewrite is too short or bad
        if len(rewritten) < 15 or rewritten.lower() == original_question.lower():
            logger.info(f"Query rewrite too weak, using original: {original_question}")
            return original_question

        logger.info(f"🔄 QUERY REWRITTEN: '{original_question}' → '{rewritten}'")
        return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return original_question
