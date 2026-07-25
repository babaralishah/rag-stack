from unittest.mock import patch
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when running tests directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.api import resolve_phase_flags, get_query_settings
from src.cache import get_cache_key
from src.rag_pipeline import rag_answer
import src.rag_pipeline as rag_module


def test_phase_keys_present():
    expected = {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert set(config.PHASE_FLAGS.keys()) >= expected


def test_resolve_phase_flags_and_get_query_settings():
    for phase_key, expected_flags in config.PHASE_FLAGS.items():
        # resolve_phase_flags should return the same mapping
        resolved = resolve_phase_flags(phase_key)
        assert resolved == expected_flags

        # get_query_settings should return phase flags when phase is provided
        Q = type("Q", (), {})
        req = Q()
        req.phase = phase_key
        req.use_hybrid = False
        req.use_reranker = False
        settings = get_query_settings(req)
        assert settings == expected_flags


def test_cache_key_includes_phase():
    k_none = get_cache_key("hello", use_hybrid=False, use_reranker=False, top_k=5, phase=None)
    k_v1 = get_cache_key("hello", use_hybrid=False, use_reranker=False, top_k=5, phase="V1")
    assert k_none != k_v1


def test_guardrail_abstain_and_allow():
    # Patch generate_answer to avoid external LLM calls
    with patch.object(rag_module, "generate_answer", return_value="STUB_ANSWER"):
        # Low-confidence retrieval should trigger guardrail abstention
        retrieved_low = [{"text": "x", "metadata": {}, "score": 0.2, "rerank_score": 0.0}]
        out_low = rag_answer(
            "what is x?",
            retrieved_low,
            use_reranker=False,
            final_top_k=1,
            use_guardrails=True,
        )
        assert isinstance(out_low, dict)
        # Should either return no sources or a refusal message
        assert out_low.get("sources", []) == [] or "don't have" in out_low.get("answer", "").lower()

        # High-confidence retrieval should allow normal LLM call (patched)
        retrieved_high = [
            {
                "text": "useful",
                "metadata": {"source_file": "f", "page": 1},
                "score": 0.95,
                "rerank_score": 0.0,
                "final_score": 0.95,
            }
        ]
        out_high = rag_answer(
            "what is x?",
            retrieved_high,
            use_reranker=False,
            final_top_k=1,
            use_guardrails=True,
        )
        assert isinstance(out_high, dict)
        assert "STUB_ANSWER" in out_high.get("answer", "")
        assert out_high.get("sources")


if __name__ == "__main__":
    # Simple runner for environments without pytest installed
    test_phase_keys_present()
    test_resolve_phase_flags_and_get_query_settings()
    test_cache_key_includes_phase()
    test_guardrail_abstain_and_allow()
    print("All phase-flag smoke tests passed.")
