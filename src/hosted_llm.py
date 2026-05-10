import os
import logging
from groq import Groq

logger = logging.getLogger("rag")

# Optional: Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai is not installed. Gemini will not be available.")

# Initialize Gemini if key and package exist
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Google Gemini API initialized")


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    return Groq(api_key=api_key)


def generate_answer(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",   # Default model
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """
    Unified function to call different LLMs.
    """
    try:
        # === GEMINI ROUTE ===
        if model.startswith("gemini") and GEMINI_AVAILABLE and GEMINI_API_KEY:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            logger.info(f"Generated using Gemini model: {model}")
            return response.text.strip()

        # === GROQ ROUTE (Default) ===
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated using Groq model: {model}")
        return answer

    except Exception as e:
        logger.error(f"LLM Error with model {model}: {e}")
        return f"LLM Error: {str(e)}"