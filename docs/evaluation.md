# Evaluation System for Local RAG

This project now includes a lightweight answer-quality evaluation system for RAG responses.

## What was added

- `src/evaluator.py`
  - Computes a simple RAGAS-style quality score for each answer.
  - Uses retrieval signal and source support to estimate answer fidelity.
  - Optionally computes reference comparison scores (precision/recall/F1) when a reference answer is provided.

- `src/api.py`
  - Imports the evaluation module.
  - Extends `QueryResponse` with an optional `evaluation` field.
  - Adds `/evaluate` endpoint for manual answer scoring with optional reference.
  - Computes evaluation metrics for every `/query` result.

- `src/ui.py`
  - Displays answer quality metrics in the Streamlit UI.
  - Shows RAGAS score, support coverage, source confidence, label, and warnings.

## How it works

1. The user asks a question and the backend retrieves candidate document chunks.
2. `src/rag_pipeline.py` generates the answer and returns the final source citations.
3. `src/api.py` passes the generated answer and returned sources to `src/evaluator.py`.
4. `src/evaluator.py` computes:
   - `source_confidence`: average final score of the cited sources.
   - `source_support`: token overlap between the answer and the retrieved snippets.
   - `max_source_score`: highest source relevance score.
   - `ragas_score`: a weighted combination of retrieval confidence and source support.
   - `label`: high / medium / low quality.
   - `warnings`: signals when the answer may be weak.
   - `reference_scores` (optional): precision, recall, and F1 against a reference answer.

## Files changed

- `src/evaluator.py`
- `src/api.py`
- `src/ui.py`
- `README.md`
- `RAG upcomings.md`

## Benefits

- Provides a real-time quality score for each RAG answer.
- Helps detect weak or unsupported responses.
- Enables future extensions to true reference evaluation and more advanced metrics.

## Limitations

- This is a heuristic metric; it does not prove factual correctness.
- It uses token overlap, so it can miss semantic support in paraphrased answers.
- It does not replace human validation for critical applications.

## Using the new feature

- Ask a question through the Streamlit UI as before.
- The answer card now includes an "Answer Quality" section.
- Use the API `/evaluate` endpoint to score any answer and sources manually.
