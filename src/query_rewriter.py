import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger("rewriter")

def rewrite_query(original_question: str) -> str:
    prompt = f"""You are an expert at rewriting questions for better document retrieval.

Original Question: {original_question}

Rewrite the question to be more specific, detailed, and effective for semantic search.
Keep it as ONE clear question.

Rewritten Question:"""

    try:
        rewritten = generate_answer(
            prompt=prompt, 
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=200
        )
        
        rewritten = rewritten.strip()
        
        if len(rewritten) < 5:
            logger.info(f"Query rewrite too short, using original: {original_question}")
            return original_question
            
        logger.info(f"🔄 QUERY REWRITTEN: '{original_question}' → '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return original_question