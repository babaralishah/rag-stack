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
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """
    Unified LLM caller.
    - Gemini is used by default if a Gemini API key exists.
    - Groq is used as fallback when Gemini is unavailable or explicitly requested.
    """
    chosen_model = model
    if chosen_model is None:
        if GEMINI_AVAILABLE and client_gemini:
            chosen_model = "gemini-2.5-flash-lite"
        else:
            chosen_model = "llama-3.3-70b-versatile"

    try:
        if chosen_model.startswith("gemini"):
            if GEMINI_AVAILABLE and client_gemini:
                response = client_gemini.models.generate_content(
                    model=chosen_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                logger.info(f"Generated using Gemini: {chosen_model}")
                return response.text.strip()
            else:
                logger.warning(f"Gemini requested but not available. Falling back to Groq.")
                chosen_model = "llama-3.3-70b-versatile"

        client = get_groq_client()
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated using Groq: {chosen_model}")
        return answer

    except Exception as first_error:
        logger.warning(f"Primary model {chosen_model} failed: {first_error}")

        if GEMINI_AVAILABLE and client_gemini and not chosen_model.startswith("gemini"):
            try:
                fallback_model = "gemini-2.5-flash-lite"
                response = client_gemini.models.generate_content(
                    model=fallback_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                logger.info(f"Fell back to Gemini: {fallback_model}")
                return response.text.strip()
            except Exception as gemini_error:
                logger.error(f"Gemini fallback failed: {gemini_error}")
                return f"LLM Error: Groq failed with {first_error} and Gemini fallback failed with {gemini_error}"

        return f"LLM Error: {str(first_error)}"
    
    