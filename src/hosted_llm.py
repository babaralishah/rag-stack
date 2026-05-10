import os
import logging
from functools import lru_cache
from groq import Groq

logger = logging.getLogger("rag")

# Optional: Google Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini will not be available.")

# Initialize Gemini if key and package exist
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Google Gemini API initialized")


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Return a cached Groq client (created once per process)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    return Groq(api_key=api_key)


def generate_answer(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Unified function to call different LLMs.
    Routes to Gemini when the model name starts with "gemini" and the API
    key is present; otherwise uses Groq.
    """
    try:
        # === GEMINI ROUTE ===
        if model.startswith("gemini"):
            if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
                # Explicit error instead of silently passing a Gemini model name to Groq
                raise ValueError(
                    f"Gemini model '{model}' requested but Gemini is not available. "
                    "Check GEMINI_API_KEY and google-generativeai installation."
                )
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            logger.info("Generated using Gemini model: %s", model)
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
        logger.info("Generated using Groq model: %s", model)
        return answer

    except Exception as e:
        logger.error("LLM error with model '%s': %s", model, e, exc_info=True)
        return f"LLM Error: {str(e)}"