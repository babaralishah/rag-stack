import logging
from src.hosted_llm import generate_answer

logger = logging.getLogger("rag")

def rewrite_query(original_question: str) -> str:
    """
    Advanced Query Rewriter - Makes the question much better for retrieval.
    """
    prompt = f"""You are an expert RAG Query Optimizer. Your job is to rewrite the user's question 
to maximize the chance of retrieving the most relevant chunks from documents.

Original Question: {original_question}

Rewrite the question with these goals:
- Make it more specific and detailed
- Include key terms likely to appear in the document
- Expand vague questions into clear retrieval-friendly questions
- Keep it as ONE single, natural question
- Do not answer the question, only rewrite it

Examples:
Original: "what is this"
Rewritten: "What is the main topic and purpose of this document? What technologies and architecture does it describe?"

Original: "tell me about project"
Rewritten: "What is the Local RAG project? Summarize its architecture, key components, and main features."

Original: "experience"
Rewritten: "What is the professional experience and work history mentioned in this resume?"

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