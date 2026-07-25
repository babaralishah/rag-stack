"""Unified LLM interface for Groq and Google Gemini.

This module abstracts provider selection and centralizes environment key
handling for the app's answer generation pipeline.
"""

import os
import logging
from groq import Groq

logger = logging.getLogger("rag")

# State tracking variables
GEMINI_AVAILABLE = None
client_gemini = None


def _initialize_gemini_on_demand():
    """Lazily initializes the Gemini client to guarantee environment variables are loaded first."""
    global GEMINI_AVAILABLE, client_gemini
    
    if GEMINI_AVAILABLE is not None:
        return  # Already attempted initialization
        
    try:
        from google import genai
        
        # Pull the key right now, after dotenv has completed loading
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            client_gemini = genai.Client(api_key=api_key)
            GEMINI_AVAILABLE = True
            logger.info("✅ Google Gemini (new genai package) initialized successfully on-demand")
        else:
            GEMINI_AVAILABLE = False
            logger.warning("⚠️ GEMINI_API_KEY not found during on-demand activation. Gemini disabled.")
            
    except ImportError:
        GEMINI_AVAILABLE = False
        logger.warning("❌ google-genai package is not installed. Gemini will be unavailable.")


def get_groq_client():
    """Create and return a Groq API client from environment configuration."""
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
    Unified LLM caller supporting Groq and Google Gemini.
    """
    try:
        # === GEMINI ROUTE ===
        if model.startswith("gemini"):
            try:
                # Force lazy-loading to capture newly loaded environment keys
                _initialize_gemini_on_demand()

                if not GEMINI_AVAILABLE or not client_gemini:
                    raise ValueError(
                        f"Gemini model '{model}' requested but Gemini is not available. Check your GEMINI_API_KEY."
                    )

                # Import types locally to ensure google package exists
                from google import genai

                response = client_gemini.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                logger.info(f"Generated using Gemini: {model}")
                return response.text.strip()
            except Exception as gemini_error:
                # Hard fallback to Groq to avoid rewrite/query pipeline collapse.
                fallback_model = "llama-3.3-70b-versatile"
                logger.warning(
                    "Gemini call failed for model %s, falling back to Groq model %s. Error: %s",
                    model,
                    fallback_model,
                    gemini_error,
                )
                client = get_groq_client()
                response = client.chat.completions.create(
                    model=fallback_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = response.choices[0].message.content.strip()
                logger.info(
                    "Generated using Groq fallback: requested=%s actual=%s",
                    model,
                    fallback_model,
                )
                return answer

        # === GROQ ROUTE (Default) ===
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