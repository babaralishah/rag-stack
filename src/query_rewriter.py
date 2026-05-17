import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger("rag")

def rewrite_query(original_question: str) -> str:
    """
    Advanced Query Rewriter - Makes the question much better for retrieval.
    """
    normalized = original_question.strip()
    if len(normalized.split()) < 8 and ("db" in normalized.lower() or "database" in normalized.lower()):
        logger.info("Short SQLite/database query detected; skipping rewrite to preserve intent.")
        return original_question

    prompt = f"""You are an expert RAG Query Optimizer. Your job is to rewrite the user's question for retrieval,
while preserving the original meaning exactly and without adding any new topics, entities, or assumptions.

Original Question: {original_question}

Rewrite the question with these goals:
- Keep the same meaning and focus as the original question
- Do not invent or add any details, topics, or domain-specific assumptions
- Do not add information about customers, sales, demographics, product preferences, or other entities
  unless the original question explicitly mentioned them
- Keep it as ONE single, natural question
- Do not answer the question, only rewrite it

Examples:
Original: "what is this"
Rewritten: "What is the main topic and purpose of this document?"

Original: "tell me about project"
Rewritten: "What is the Local RAG project? Summarize its architecture, key components, and main features."

Original: "experience"
Rewritten: "What is the professional experience and work history mentioned in this resume?"

Original: "what does this db says"
Rewritten: "What information does this database contain?"

Now rewrite the following question:

Rewritten Question:"""

    try:
        rewritten = generate_answer(
            prompt=prompt,
            # model="llama-3.1-8b-instant",   # Fast model for rewriting
            model="gemini-2.5-flash-lite",   # gemini-2.5-flash-preview-05-20; gemini-2.5-flash
            temperature=0.3,
            max_tokens=250
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