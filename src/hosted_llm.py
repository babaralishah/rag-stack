import os
import logging
from groq import Groq

logger = logging.getLogger("rag")

# ====================== GEMINI SETUP (New Package) ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_AVAILABLE = False
client_gemini = None

try:
    from google import genai
    GEMINI_AVAILABLE = True
    if GEMINI_API_KEY:
        client_gemini = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("✅ Google Gemini (new genai package) initialized successfully")
    else:
        logger.info("GEMINI_API_KEY not found. Gemini disabled.")
except ImportError:
    logger.warning("google-genai package is not installed. Gemini will be unavailable.")


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
    """
    Unified LLM caller.
    - Gemini is tried only if explicitly requested.
    - Groq is used as default and fallback.
    """
    try:
        # === GEMINI ROUTE (Only if explicitly requested) ===
        if model.startswith("gemini"):
            if GEMINI_AVAILABLE and client_gemini:
                try:
                    response = client_gemini.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    logger.info(f"Generated using Gemini: {model}")
                    return response.text.strip()
                except Exception as gemini_error:
                    logger.warning(f"Gemini failed: {gemini_error}. Falling back to Groq.")
            else:
                logger.warning(f"Gemini requested but not available. Falling back to Groq.")

        # === GROQ ROUTE (Default + Fallback) ===
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated using Groq: {model}")
        return answer

    except Exception as e:
        logger.error(f"LLM Error with model {model}: {e}")
        return f"LLM Error: {str(e)}"
    
    