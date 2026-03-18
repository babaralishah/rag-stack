import logging
import requests
from typing import List, Dict, Any, Tuple
from src.config import MAX_CHARS, MIN_SIMILARITY, OLLAMA_BASE_URL, API_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

OLLAMA_URL = OLLAMA_BASE_URL + "/api/generate"
DEFAULT_MODEL = OLLAMA_MODEL

def build_context(results: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
    """
    Turn retrieved chunks into a compact context block.
    """
    parts = []
    total = 0
    for r in results:
        meta = r["metadata"]
        header = f"[source: {meta.get('source_file')} p.{meta.get('page')}]"
        text = r["text"].strip().replace("\n", " ")
        block = f"{header}\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()

def call_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "temperature": temperature}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    except Exception as e:
        logger.exception("Ollama call failed: %s", e)
        raise

def rag_answer(
    question: str,
    retrieved: List[Dict[str, Any]],
    min_score: float = 0.35,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Guardrail:
      - If top similarity score is too low => "I don't know based on the documents."
    Returns:
      - answer
      - sources: file + page + snippet + score
    """
    if not retrieved:
        return {
            "answer": "I don’t know based on the documents.",
            "sources": [],
        }

    top_score = retrieved[0]["score"]
    if top_score < min_score:
        return {
            "answer": "I don’t know based on the documents.",
            "sources": [
                {
                    "score": r["score"],
                    "file": r["metadata"].get("source_file"),
                    "page": r["metadata"].get("page"),
                    "snippet": r["text"][:250].replace("\n", " "),
                }
                for r in retrieved
            ],
        }

    context = build_context(retrieved)

    prompt = f"""You are a careful assistant.

    Use ONLY the provided context to answer the question.
    If the answer is clearly supported by the context, answer directly and briefly.
    If the answer is not present in the context, say:
    "I don't know based on the documents."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    answer = call_ollama(prompt, model=model, temperature=0.1)

    seen = set()
    sources = []

    for r in retrieved:
        file = r["metadata"].get("source_file")
        page = r["metadata"].get("page")
        key = (file, page)

        if key in seen:
            continue
        seen.add(key)

        sources.append({
            "score": r["score"],
            "file": file,
            "page": page,
            "snippet": r["text"][:250].replace("\n", " "),
        })

    return {"answer": answer, "sources": sources}