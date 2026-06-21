---
title: RAG LLM App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Empirical Ablation Study of Retrieval-Augmented Generation Architectures for Biomedical Question Answering

**An Open-Source R&D Framework for Controlled, Multi-Phase Evaluation of Query Augmentation, Hybrid Retrieval, and Neural Re-Ranking Strategies in Dense Vector Information Retrieval**

---

## Abstract

This repository constitutes the primary computational artifact of an ongoing doctoral research programme investigating the empirical performance boundaries of Retrieval-Augmented Generation (RAG) systems under controlled ablation conditions in the biomedical domain. The central research problem addressed is the well-documented **vocabulary mismatch problem** inherent to dense vector retrieval, wherein short, underspecified natural language queries produce embeddings that fail to adequately span the semantic neighbourhood of relevant biomedical passages. This problem is compounded by **semantic drift in high-dimensional embedding spaces**, where query rewriting strategies can simultaneously improve specificity for some query types while introducing corruptive token inflation for others, and by **zero-shot hallucination boundaries** in generative models operating on specialised scientific corpora with limited pre-training coverage.

This framework operationalises a **six-way controlled ablation matrix** designed to isolate the individual and compound contributions of five architectural retrieval augmentations over a fixed baseline configuration, evaluated using a locked retrieval depth of exactly five candidate passages per query across a 100-query biomedical benchmark. All experimental conditions maintain strict isolation: no advanced feature activates silently through phase inheritance or default parameter leakage. Results are captured at row granularity, persisted as structured Excel workbooks with three analytical sheets, and accompanied by a row-level delta analysis engine that quantifies per-query performance divergence relative to the unaided baseline.

**Core Research Questions:**
1. Does Hypothetical Document Embedding (HyDE) query expansion yield statistically significant improvements in biomedical retrieval precision over unaugmented semantic search?
2. Does keyword-based query expansion act as a positive retrieval signal or as a negative control through token inflation in dense vector spaces?
3. What is the marginal contribution of cross-encoder reranking, BM25-FAISS sparse-dense fusion, and conversational memory enrichment when applied in isolation to the baseline retrieval configuration?

---

## Repository Structure

```
local-rag/
├── src/
│   ├── api.py               # FastAPI backend: /query, /eval, /upload endpoints
│   ├── query_rewriter.py    # LLM-based query augmentation (HyDE, keyword expansion)
│   ├── hosted_llm.py        # Unified LLM interface (Groq, Gemini) with fallback chain
│   ├── vector_store.py      # FAISS store with BM25 hybrid fusion and RRF scoring
│   ├── embedder.py          # HuggingFace sentence-transformer embedding wrapper
│   ├── reranker.py          # Cross-encoder re-scoring layer (BGE-Reranker-Base)
│   ├── hybrid_retriever.py  # Reciprocal Rank Fusion orchestration
│   ├── rag_pipeline.py      # End-to-end generation pipeline with guardrail abstention
│   ├── evaluator.py         # RAGAS-style evaluation metrics
│   ├── advanced_metrics.py  # Precision@k, Recall@k, nDCG, MRR computation
│   ├── cache.py             # TTL-based embedding and query response caching
│   ├── config.py            # Centralised constants and PHASE_FLAGS ablation registry
│   └── chunker.py           # Sliding-window chunk tokenisation
├── docs/
│   └── 3.0/
│       ├── script.py                          # Automated evaluation driver (6-profile matrix)
│       ├── generate_charts.py                 # Seaborn/matplotlib publication chart generator
│       ├── Thesis_Metrics_none.xlsx           # Baseline run output
│       ├── Thesis_Metrics_hyde.xlsx           # HyDE experimental run output
│       ├── Thesis_Metrics_keyword_expansion.xlsx
│       ├── Thesis_Metrics_rerank_only.xlsx
│       ├── Thesis_Metrics_hybrid_only.xlsx
│       ├── Thesis_Metrics_memory_only.xlsx
│       └── TODO_NEXT_STEPS.md                 # Execution playbook
├── storage/faiss/           # Persisted FAISS index and JSONL metadata
├── data/uploads/            # Ingested document store
├── ui.py                    # Streamlit frontend interface
└── requirements.txt
```

---

## Methodology and 6-Way Architectural Ablation Matrix

Each profile isolates exactly one architectural variable over the fixed baseline configuration. The evaluation script enforces strict mutual exclusion: when a profile is selected, all other advanced features are explicitly set to false in the API payload, preventing silent activation via phase flag inheritance.

| Profile | Query Rewriter | Rewrite Strategy | Hybrid Search | Cross-Encoder Rerank | Conversational Memory | LLM Temp | Research Role |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `none` | No | — | No | No | No | 0.0 | Baseline reference floor |
| `keyword_expansion` | Yes | keyword_expansion | No | No | No | 0.0 | Negative control: sparse token inflation |
| `hyde` | Yes | HyDE | No | No | No | 0.2 | Experimental: dense semantic expansion |
| `rerank_only` | No | — | No | Yes | No | 0.0 | Isolated: cross-encoder re-scoring |
| `hybrid_only` | No | — | Yes | No | No | 0.0 | Isolated: sparse-dense BM25+FAISS fusion |
| `memory_only` | No | — | No | No | Yes | 0.0 | Isolated: conversational context enrichment |

**Fixed System Parameters (invariant across all profiles):**

| Parameter | Value |
|---|---|
| Embedding Model | `BAAI/bge-large-en-v1.5` |
| Embedding Dimensionality | 1024 |
| Chunk Size | 600 characters |
| Chunk Overlap | 50 characters |
| Retrieval Depth (top-k) | **5 (locked; no overrides permitted)** |
| Inference Model | `llama-3.3-70b-versatile` via Groq API |
| Vector Store | FAISS `IndexFlatIP` (inner-product cosine similarity) |
| Benchmark Size | 100 queries (CovidQA biomedical corpus) |

---

## System Architecture

### Indexing Pipeline

```
Raw Document
  → pypdf Text Extraction
  → Sliding-Window Chunker (600 chars / 50 overlap)
  → BAAI/bge-large-en-v1.5 Embedding (1024d, normalised)
  → FAISS IndexFlatIP Vector Store
  → JSONL Metadata Persistence
  → BM25Okapi Inverted Index (parallel construction)
```

### Query Execution Pipeline

```
User Query
  ↓
[Query Augmentation Layer]  ← HyDE / keyword_expansion / passthrough
  ↓
Embed Augmented Query (bge-large-en-v1.5, 1024d)
  ↓
[Retrieval Layer]
  ├── Semantic: FAISS Inner-Product Search (top k×4 candidates)
  └── (if hybrid_only) BM25 Keyword Search → Reciprocal Rank Fusion
  ↓
[Reranking Layer]  ← (if rerank_only) BGE-Reranker-Base cross-encoder scoring
  ↓
Top-5 Passages Selected
  ↓
[Guardrail Abstention Check]  ← similarity floor filter
  ↓
Context Prompt Construction → LLM Generation (Groq llama-3.3-70b-versatile)
  ↓
Answer + Source Citations
```

---

## Evaluation Pipeline and Automated Verification Layer

### The /eval Endpoint

The `/eval` endpoint (`POST http://localhost:8001/eval`) is a **retrieval-only** interface that bypasses answer generation entirely. It accepts structured JSON payloads specifying all feature toggles explicitly, executes the retrieval stack as configured, and returns the raw top-5 text passages along with a rewrite verification flag.

**Request Schema:**

```json
{
  "query": "string",
  "chunks": 5,
  "rewriting_strategy": "none | keyword_expansion | hyde",
  "use_query_rewriter": "true | false",
  "hybrid": "true | false",
  "rerank": "true | false",
  "use_chat_history": "true | false",
  "ablation": "V1 | V2 | ... | null"
}
```

**Response Schema:**

```json
{
  "status": "success",
  "retrieved_context_keys": ["passage_1_text", "..."],
  "query_was_rewritten": true
}
```

**Explicit Control Precedence Rule:** Payload-level boolean flags override any server-side `PHASE_FLAGS` preset. This guarantees the evaluation script's profile toggles are always authoritative and cannot be silently overridden.

---

### Rewrite Verification Tracking Layer

A critical design concern in automated ablation studies is **silent rewrite collapse**: the condition whereby the LLM augmentation call fails silently (network error, API authentication failure, empty return, or error string output), and the system transparently falls back to the original query text without logged indication. This renders the experimental condition scientifically indistinguishable from the baseline while falsely reporting it as the augmented condition.

This framework implements a dedicated **Rewrite Verification Tracking Layer**:

- `rewrite_and_embed_query()` returns a three-tuple: `(query_final, query_vector, query_was_rewritten: bool)`.
- `query_was_rewritten` is `True` if and only if the LLM returned a non-empty string that differs lexicographically from the original query after normalisation.
- The boolean is serialised in every `/eval` response as `"query_was_rewritten": true | false`.
- Per-row, the evaluation script populates `rewrite_status`: `"Rewritten Query"` or `"Original Query"`.
- The `Macro_Averages` sheet includes `rewrite_success_rate` for audit.
- Any row where `EXPERIMENT_PROFILE in {keyword_expansion, hyde}` but `rewrite_status = "Original Query"` is flagged at ERROR level in the backend log as a **verified silent collapse event**.

---

### Row-Level Delta Analysis Engine (Delta_vs_None)

Each non-baseline workbook contains a third sheet `Delta_vs_None`, computed as:

$$\Delta_m^{(i)} = m_{\text{experimental}}^{(i)} - m_{\text{baseline}}^{(i)}$$

for each of the eight tracked metrics at every query row $i$. Per-metric impact labels (`Helped`, `Hurt`, `No Change`) and row-level aggregates (`helped_metric_count`, `hurt_metric_count`, `net_delta_sum`, `overall_impact`) provide direct row-granularity evidence for results chapter claims.

---

## Fuzzy Ground-Truth Matching Engine

Retrieved passages are evaluated against nested ground-truth sentence mappings using a calibrated **three-tier fuzzy matching** strategy to account for chunk boundary truncation:

| Tier | Method | Condition |
|---|---|---|
| 1 | Strict substring containment | `norm_gt ⊆ norm_chunk` or `norm_chunk ⊆ norm_gt` |
| 2 | Sliding 7-word phrase window | Any 7-consecutive-word phrase from GT present in chunk |
| 3 | Jaccard token overlap | `|T_chunk ∩ T_GT| / |T_GT| >= 0.55` |

---

## Evaluation Metrics

All metrics are bounded to [0.0, 1.0] by construction.

| Metric | Formula | Level |
|---|---|---|
| Precision@k | `|relevant chunks retrieved| / k` | Chunk/rank |
| Recall@k | `|wanted keys found| / |wanted keys|` | Sentence |
| F1-Score | `2 · P · R / (P + R)` | Derived |
| Exact Match | `1 if wanted ⊆ caught else 0` | Binary |
| Hit Rate | `1 if caught ≠ ∅ else 0` | Binary |
| MRR | `1 / rank of first hit` | Rank |
| nDCG@5 | `DCG / IDCG`, one gain per unique key, first occurrence | Rank |
| Support Coverage | Mirrors Recall@k — clinical factual coverage proxy | Sentence |

---

## Early Empirical Findings

Results from the initial 98-query validated run (preliminary; pending bge-large re-indexing):

| Metric | none (Baseline) | keyword_expansion | hyde |
|---|:---:|:---:|:---:|
| Precision@5 | **0.2327** | 0.1816 | 0.2327 |
| Recall@k | **0.4279** | 0.3144 | 0.3747 |
| F1-Score | **0.2721** | 0.2055 | 0.2609 |
| Exact Match | **0.2551** | 0.1837 | **0.2551** |
| Hit Rate | **0.6020** | 0.5102 | 0.5612 |
| MRR | 0.4459 | 0.3736 | **0.4500** |
| nDCG@5 | **0.4507** | 0.3524 | 0.4202 |

**Interpretive Summary:**

- **Baseline dominates Recall@k and nDCG@5**, consistent with the hypothesis that unaugmented queries, though semantically narrow, do not introduce off-distribution noise into the retrieval neighbourhood.
- **HyDE marginally improves MRR (+0.0041)** relative to baseline, suggesting that synthetic paragraph-form query expansion successfully surfaces the most relevant passage at higher rank even when set recall is reduced. This is consistent with HyDE's theoretical mechanism: improving rank precision at the cost of set recall.
- **Keyword Expansion degrades all metrics**, confirming the negative control hypothesis. Term proliferation without semantic coherence disperses the query embedding away from tightly clustered biomedical passage centroids, consistent with the curse of dimensionality in high-dimensional inner-product spaces.
- The rewrite success rate reached 100% (98/98 rows) for both augmentation profiles following resolution of the Gemini-to-Groq fallback chain, confirming zero silent collapse events in the logged runs.

> **Important:** These findings are preliminary and based on the `all-MiniLM-L6-v2` embedding configuration. Final thesis results require full re-execution under the thesis-locked `BAAI/bge-large-en-v1.5` (1024d) embedding model with a freshly constructed FAISS index.

---

## Installation

### Prerequisites

- Python 3.10–3.13
- Groq API key (LLM inference and query rewriting)
- 16 GB RAM recommended (bge-large-en-v1.5 model)

### Environment Setup

```bash
git clone <repository-url>
cd local-rag
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key   # optional
```

### Running the Backend

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

### Running the Evaluation Driver

```bash
cd docs/3.0
# Set EXPERIMENT_PROFILE in script.py to the desired profile, then:
python script.py
```

### Generating Publication Charts

```bash
cd docs/3.0
python generate_charts.py
# Output: thesis_ablation_results.png (300 DPI)
```

### Resetting the Vector Index

```bash
# Windows
rmdir /s /q storage\faiss

# Linux/macOS
rm -rf storage/faiss
```

---

## Citation

If this framework, evaluation methodology, or evaluation driver code is used in academic work, please cite appropriately once the associated paper is submitted for peer review.

---

## Acknowledgements

This research makes use of the CovidQA benchmark dataset. Embedding infrastructure is provided by BAAI (`bge-large-en-v1.5`, `bge-reranker-base`). LLM inference is provided via the Groq API (`llama-3.3-70b-versatile`).