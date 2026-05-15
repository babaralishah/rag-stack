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
    Unified LLM caller supporting Groq and Google Gemini.

    If GROQ_API_KEY exists, Groq will be used by default with a smaller model.
    If Groq fails, it falls back to Gemini when available.
    """
    chosen_model = model
    if chosen_model is None:
        if os.getenv("GROQ_API_KEY"):
            chosen_model = os.getenv("GROQ_MODEL", "llama-3.3-mini")
        elif GEMINI_AVAILABLE and client_gemini:
            chosen_model = "gemini-2.5-flash-lite"
        else:
            return "LLM Error: No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY."

    is_gemini_model = chosen_model.startswith("gemini")

    def call_gemini(model_name: str) -> str:
        if not GEMINI_AVAILABLE or not client_gemini:
            raise ValueError("Gemini is not available.")
        response = client_gemini.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        logger.info(f"Generated using Gemini: {model_name}")
        return response.text.strip()

    def call_groq(model_name: str) -> str:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated using Groq: {model_name}")
        return answer

    try:
        if is_gemini_model:
            return call_gemini(chosen_model)
        return call_groq(chosen_model)

    except Exception as first_error:
        logger.warning(f"Primary model {chosen_model} failed: {first_error}")

        if not is_gemini_model and GEMINI_AVAILABLE and client_gemini:
            try:
                return call_gemini("gemini-2.5-flash-lite")
            except Exception as gemini_error:
                logger.error(f"Gemini fallback failed: {gemini_error}")
                return f"LLM Error: Groq failed with {first_error} and Gemini fallback failed with {gemini_error}"

        if not is_gemini_model and os.getenv("GROQ_API_KEY"):
            for fallback_model in ["llama-3.3-mini", "llama-3.3-fast", "llama-3.3-70b-versatile"]:
                if fallback_model == chosen_model:
                    continue
                try:
                    return call_groq(fallback_model)
                except Exception as fallback_error:
                    logger.warning(f"Fallback Groq model {fallback_model} also failed: {fallback_error}")

        return f"LLM Error: {first_error}"
