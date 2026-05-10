import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger(__name__)

def rewrite_query(original_question: str) -> str:
    """
    Rewrites the user's question to be more effective for retrieval.
    """
    prompt = f"""You are an expert at rewriting questions for document retrieval systems.

Original Question: {original_question}

Task: Rewrite the question to be more specific, detailed, and effective for semantic search.
- Make it clear and professional.
- Expand vague questions.
- Keep it as ONE single question.
- Do not answer it, just rewrite.

Rewritten Question:"""

    try:
        rewritten = generate_answer(prompt, temperature=0.3, max_tokens=150)
        rewritten = rewritten.strip()
        
        if len(rewritten) < 10:  # Safety check
            return original_question
            
        logger.info(f"Query Rewritten: '{original_question}' → '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}. Using original question.")
        return original_question