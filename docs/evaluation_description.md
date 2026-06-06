# RAGAS Evaluation Overview

This document describes how the current custom evaluation system works in theory.

## What is being evaluated

The system computes a small set of quality metrics for a RAG answer using:
- the final generated answer text
- the retrieved source snippets returned by the retrieval pipeline
- the retrieved source relevance scores
- an optional reference answer, if provided
- an optional query text, when available

## Core metrics

### 1. `source_confidence`
- Defined as the average of the selected sources' relevance scores.
- The code uses the field `final_score` from each source entry, or `score` if `final_score` is missing.
- This is a heuristic for retrieval confidence: higher average source scores mean the retrieved chunks were judged more relevant.

### 2. `source_support`
- Measures lexical overlap between text and the retrieved source snippets.
- The evaluator builds a single support text by joining all source `snippet` values.
- It computes a token overlap ratio between:
  - the rewritten/original query text, if available, otherwise
  - the answer text.
- This is used as the “support coverage” signal.
- Because it uses token overlap, it is a surface-level support metric rather than a deep semantic entailment score.

### 3. `max_source_score`
- The highest relevance score among all selected source chunks.
- This captures whether at least one very strong source was retrieved.

### 4. `ragas_score`
- A weighted combination of the metrics above:
  - `0.45 * mean_source_score`
  - `0.45 * source_support`
  - `0.10 * max_source_score`
- The combined score is clamped between 0.0 and 1.0.
- This gives a single answer quality estimate by blending retrieval confidence and support coverage.

## Output labels and warnings

The system also produces:
- `label`: a quality tier based on `ragas_score`
  - `high` if score >= 0.75
  - `medium` if score >= 0.45
  - `low` otherwise
- `warnings`: a small list of caution signals
  - `low_retrieval_confidence` if mean source score is below 0.35
  - `low_source_support` if source support is below 0.35
  - `review_answer_quality` if the score is low but no other warning was added

## Reference evaluation

If a reference answer is supplied, the evaluator also computes:
- `precision` = overlap of answer tokens inside reference tokens
- `recall` = overlap of answer tokens from reference tokens
- `f1` = harmonic mean of precision and recall

This is only added when a `reference` string is provided.

## How the evaluation is used

- The API evaluates answers during normal `/query` responses.
- It also supports a manual `/evaluate` endpoint that accepts an answer, sources, optional reference, and optional query.
- The evaluation is therefore a heuristic quality layer on top of the retrieval + answer generation pipeline.

## Important caveats

- The system is heuristic and does not prove factual correctness.
- `source_support` is based on token overlap, so paraphrased support may be undercounted.
- `source_confidence` depends entirely on the retrieval scoring model, not on the factual accuracy of the final answer.
- The combined `ragas_score` is a weighted indicator, not a true evaluation of answer correctness.
