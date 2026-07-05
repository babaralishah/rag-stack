"""Official RAGAS framework integration for runtime RAG evaluation.

This module is intentionally isolated from existing custom metrics so the
application can keep running even when RAGAS dependencies or provider keys are
not available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from typing import Any, Dict, List


logger = logging.getLogger("rag")


def _ensure_vertexai_compat_module() -> None:
    """Provide a compatibility module expected by some RAGAS versions.

    Certain RAGAS/langchain combinations import:
    `langchain_community.chat_models.vertexai`.
    Newer langchain stacks moved VertexAI integrations into
    `langchain_google_vertexai`.
    """
    try:
        __import__("langchain_community.chat_models.vertexai")
        return
    except Exception:
        pass

    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore

        vertexai_module = types.ModuleType(
            "langchain_community.chat_models.vertexai"
        )
        vertexai_module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module
        logger.info("Applied VertexAI compatibility shim for RAGAS imports")
    except Exception as exc:
        # Keep running; compute path will return a structured error later if needed.
        logger.warning("Could not prepare VertexAI compatibility shim: %s", exc)


def _build_contexts(sources: List[Dict[str, Any]]) -> List[str]:
    contexts: List[str] = []
    for src in sources:
        text = str(src.get("snippet") or src.get("text") or "").strip()
        if text:
            contexts.append(text)
    return contexts


async def _compute_ragas_async(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    _ensure_vertexai_compat_module()

    # Import inside the function so missing optional dependencies never break
    # backend startup.
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    llm_model = os.getenv("RAGAS_LLM_MODEL", "gemini-1.5-flash")
    embedding_model = os.getenv("RAGAS_EMBED_MODEL", "models/text-embedding-004")

    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.0)
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    llm_wrapper = LangchainLLMWrapper(llm)
    embedding_wrapper = LangchainEmbeddingsWrapper(embeddings)

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    faithfulness_metric = Faithfulness(llm=llm_wrapper)
    answer_relevancy_metric = ResponseRelevancy(
        llm=llm_wrapper,
        embeddings=embedding_wrapper,
    )
    context_precision_metric = ContextPrecision(llm=llm_wrapper)

    faithfulness = await faithfulness_metric.single_turn_ascore(sample)
    answer_relevancy = await answer_relevancy_metric.single_turn_ascore(sample)
    context_precision = await context_precision_metric.single_turn_ascore(sample)

    numeric_metrics = {
        "faithfulness": float(faithfulness),
        "answer_relevancy": float(answer_relevancy),
        "context_precision": float(context_precision),
    }
    numeric_metrics["average_score"] = (
        numeric_metrics["faithfulness"]
        + numeric_metrics["answer_relevancy"]
        + numeric_metrics["context_precision"]
    ) / 3.0

    return {
        "enabled": True,
        "status": "ok",
        "provider": "gemini",
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "metrics": {k: round(v, 4) for k, v in numeric_metrics.items()},
        "warnings": [],
        "error": None,
    }


def compute_ragas_framework_metrics(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute official RAGAS metrics with graceful fallback behavior.

    This function never raises. If RAGAS cannot run, it returns a structured
    unavailable/error payload so existing backend behavior remains unchanged.
    """
    if not question or not answer:
        return {
            "enabled": False,
            "status": "unavailable",
            "provider": None,
            "llm_model": None,
            "embedding_model": None,
            "metrics": {},
            "warnings": ["missing_question_or_answer"],
            "error": "Question and answer are required for RAGAS evaluation.",
        }

    contexts = _build_contexts(sources)
    if not contexts:
        return {
            "enabled": False,
            "status": "unavailable",
            "provider": None,
            "llm_model": None,
            "embedding_model": None,
            "metrics": {},
            "warnings": ["missing_contexts"],
            "error": "No retrieved contexts available for RAGAS evaluation.",
        }

    if not os.getenv("GEMINI_API_KEY"):
        return {
            "enabled": False,
            "status": "unavailable",
            "provider": "gemini",
            "llm_model": None,
            "embedding_model": None,
            "metrics": {},
            "warnings": ["missing_gemini_api_key"],
            "error": "Set GEMINI_API_KEY to enable official RAGAS metrics.",
        }

    try:
        return asyncio.run(_compute_ragas_async(question, answer, contexts))
    except RuntimeError:
        # Fallback path for environments that already have an event loop.
        try:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    _compute_ragas_async(question, answer, contexts)
                )
            finally:
                loop.close()
        except Exception as err:
            logger.warning("RAGAS loop fallback failed: %s", err)
            return {
                "enabled": False,
                "status": "error",
                "provider": "gemini",
                "llm_model": None,
                "embedding_model": None,
                "metrics": {},
                "warnings": ["ragas_runtime_error"],
                "error": str(err),
            }
    except Exception as err:
        logger.warning("RAGAS framework evaluation failed: %s", err)
        return {
            "enabled": False,
            "status": "error",
            "provider": "gemini",
            "llm_model": None,
            "embedding_model": None,
            "metrics": {},
            "warnings": ["ragas_dependency_or_provider_error"],
            "error": str(err),
        }