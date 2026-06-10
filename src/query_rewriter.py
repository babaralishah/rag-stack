import logging
import re
from src.hosted_llm import generate_answer

logger = logging.getLogger("rag")


def _clean_rewritten_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(
        r'^(Expanded Query:|Rewritten Query:|Hypothetical document excerpt:|Hypothetical answer:|Keywords?:|Output:)?\s*',
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def rewrite_query(original_question: str, history: list | None = None, strategy: str = "hyde") -> str:
    """Rewrite the user's question for retrieval using the chosen strategy."""
    normalized = original_question.strip()
    if not normalized:
        return original_question

    strategy = str(strategy or "hyde").strip().lower()
    if strategy == "none":
        return original_question

    if strategy not in {"keyword_expansion", "hyde"}:
        logger.warning(
            "Unknown rewrite strategy '%s'; falling back to 'hyde'.", strategy
        )
        strategy = "hyde"

    if len(normalized.split()) < 8 and (
        "db" in normalized.lower() or "database" in normalized.lower()
    ):
        logger.info(
            "Short SQLite/database query detected; skipping rewrite to preserve intent."
        )
        return original_question

    history_block = ""
    if history and isinstance(history, list):
        last_turns = history[-4:]
        history_lines = []
        for m in last_turns:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if content:
                history_lines.append(f"{role.title()}: {content}")
        if history_lines:
            history_block = "Conversation history:\n" + "\n".join(history_lines) + "\n\n"

    if strategy == "keyword_expansion":
        prompt = f"""You are an expert retrieval query optimizer.

Given the user's question, return ONLY a space-separated list of technical keywords, synonyms, acronyms, and relevant domain concepts that will help a vector search engine retrieve documents that answer the question.
Do not answer the question.
Do not add any explanation, headers, or labels.
{history_block}Original Question: {original_question}

Return only the keyword list:"""
        max_tokens = 80
    else:
        prompt = f"""You are an expert technical writer and retrieval assistant.

Given the user's question, write ONLY a 2-3 sentence authoritative technical paragraph that reads like a hypothetical answer document excerpt relevant to the question.
Do not answer the question directly as a chat response.
Do not include any labels, headings, or explanations.
{history_block}Original Question: {original_question}

Return only the hypothetical document excerpt:"""
        max_tokens = 220

    try:
        rewritten = generate_answer(
            prompt=prompt,
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            max_tokens=max_tokens,
        )

        rewritten = _clean_rewritten_text(rewritten)
        if strategy == "keyword_expansion":
            rewritten = re.sub(r"[\n,]+", " ", rewritten)
            rewritten = " ".join(rewritten.split())
        else:
            rewritten = rewritten.strip()

        if not rewritten:
            logger.info("Query rewrite returned empty text; using original question.")
            return original_question

        if rewritten.lower() == original_question.lower():
            logger.info(
                "Query rewrite returned the original question unchanged; using original."
            )
            return original_question

        if strategy == "keyword_expansion" and len(rewritten.split()) < 2:
            logger.info(
                "Keyword expansion output too weak, using original question."
            )
            return original_question

        if strategy == "hyde" and len(rewritten) < 40:
            logger.info("HyDE output too short, using original question.")
            return original_question

        logger.info(f"🔄 QUERY REWRITE STRATEGY: {strategy}")
        logger.info(f"🔄 QUERY REWRITTEN: '{original_question}' → '{rewritten}'")
        logger.info(f"🔄 History of chat conversation:  '{history_block}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return original_question
