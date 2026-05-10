import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger(__name__)

def rewrite_query(original_question: str) -> str:
    prompt = f"""You are an expert at rewriting questions for better document retrieval.

Original Question: {original_question}

Rewrite the question to be more specific, detailed, and effective for semantic search.
Keep it as ONE clear question. Do not answer it.

Rewritten Question:"""

    try:
        # Use cheaper & faster model for rewriting
        rewritten = generate_answer(
            prompt=prompt, 
            model="llama-3.1-8b-instant",   # Fast and cheap
            temperature=0.3,
            max_tokens=200
        )
        
        rewritten = rewritten.strip()
        
        if len(rewritten) < 5:   # fallback
            return original_question
            
        logger.info(f"🔄 Query Rewritten: '{original_question}' → '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}")
        return original_question