# Query Rewriter Strategies

This document describes the new query rewriting feature and how it works in the Local RAG app.

## New UI Controls

The Streamlit sidebar now includes a `Query Rewriting Strategy` control with three options:

- `none`: Do not rewrite the user question. Use the raw user query directly for retrieval.
- `keyword_expansion`: Ask the LLM to return a space-separated list of keywords, synonyms, acronyms, and domain concepts. The keyword list is used only for vector search.
- `hyde`: Ask the LLM to generate a 2-3 sentence authoritative technical paragraph that acts as a hypothetical document excerpt. That paragraph is used only for retrieval.

This strategy is sent to the backend as `rewriting_strategy` and is used only for vector DB search. The final answer generation step continues to receive the original user question.

## Backend Flow

1. The front-end sends `rewriting_strategy` with the query payload.
2. `src/api.py` resolves the query settings and calls `rewrite_and_embed_query(...)`.
3. When `rewriting_strategy` is `keyword_expansion` or `hyde`, `src/query_rewriter.py` generates the retrieval text.
4. The rewritten text is embedded and used for vector retrieval only.
5. The original question is passed to `rag_answer(...)` so the answer LLM is never confused by the rewritten retrieval prompt.

## UI Improvements

- The chunk retrieval slider now has a minimum value of `5`.
- The slider value is sent as `top_k` in the query request.
- The results display now reports the actual number of retrieved source chunks used in the answer.

## Evaluation Improvements

The answer quality block now shows:

- RAGAS Score
- Support Coverage
- Source Confidence
- Retrieved Sources
- Reference Precision / Recall / F1 when reference scores are available
- Label and warnings

## Notes

- If an experimental phase disables query rewriting via flags (`V1`), the `rewriting_strategy` selection is still accepted but ignored by the backend.
- The `keyword_expansion` and `hyde` strategies are designed to improve retrieval without changing the final answer prompt.
