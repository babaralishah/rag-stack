import logging
from typing import List, Dict, Any
from src.config import MAX_CHARS
from src.hosted_llm import generate_answer

logger = logging.getLogger(__name__)

def build_context(results: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> str:
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

def rag_answer(
    question: str,
    retrieved: List[Dict[str, Any]],
    min_score: float = 0.35,
) -> Dict[str, Any]:
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

Use the provided context to answer the question.
If the answer is supported by the context, answer directly and a little explained.
I don't have sufficient information in the uploaded documents to answer this question accurately."

  After that, you can optionally add: 
  "However, based on my general knowledge, ..." and give your best general answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    answer = generate_answer(prompt)

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