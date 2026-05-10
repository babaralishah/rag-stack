import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    return Groq(api_key=api_key)


def generate_answer(
    prompt: str, 
    model: str = "llama-3.3-70b-versatile", 
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    try:
        client = get_groq_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated answer using model: {model} | Tokens: ~{len(answer.split())}")
        return answer
        
    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return f"LLM Error: {str(e)}"