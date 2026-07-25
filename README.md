---
title: AI Trading Bot
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
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

### Quick Results Snapshot

- **Biomedical (primary benchmark):** `rerank_only` is strongest overall; `hybrid_only` is best for early-rank quality (MRR/nDCG).
- **ScienceQA (external validation):** `hyde` gives best end-task QA accuracy (0.35), while `keyword_expansion` raises support-hit frequency but lowers final answer accuracy.
- **Cross-dataset lesson:** retrieval-support gains and answer-accuracy gains can diverge; strategy selection should be objective-specific.

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
├── agent.py                 # Crypto scalping research CLI (scanner + deep-dive)
├── crypto_ui.py             # Standalone Streamlit UI for crypto research workflow
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

Results from the latest **98-query validated benchmark** under the thesis-locked `BAAI/bge-large-en-v1.5` embedding space:

| Metric | none (Baseline) | keyword_expansion | hyde | rerank_only | hybrid_only |
|---|:---:|:---:|:---:|:---:|:---:|
| Precision@5 | 0.2327 | 0.1816 | 0.2327 | **0.3653** | 0.2265 |
| Recall@k | 0.4279 | 0.3144 | 0.3747 | **0.5206** | 0.4351 |
| F1-Score | 0.2721 | 0.2055 | 0.2609 | **0.3896** | 0.2757 |
| Exact Match | 0.2551 | 0.1837 | 0.2551 | **0.3571** | 0.2653 |
| Hit Rate | 0.6020 | 0.5102 | 0.5612 | **0.6939** | 0.6429 |
| MRR | 0.4459 | 0.3736 | 0.4500 | 0.4580 | **0.5337** |
| nDCG@5 | 0.4507 | 0.3524 | 0.4202 | 0.4876 | **0.4878** |
| Support Coverage | 0.4279 | 0.3144 | 0.3747 | **0.5206** | 0.4351 |

### Comparative Outcome Summary

**Overall strongest performer (balanced retrieval quality):** `rerank_only`
- **Success profile:** highest Precision, Recall, F1, Exact Match, Hit Rate, and Support Coverage.
- **Interpretation:** cross-encoder re-scoring effectively corrects FAISS rank-ordering errors, especially for semantically close biomedical candidates.
- **Limitation:** MRR gain is positive but smaller than hybrid, suggesting top-rank sharpness is not maximised in every query family.

**Top rank optimiser:** `hybrid_only`
- **Success profile:** best MRR and best nDCG@5.
- **Interpretation:** sparse-dense fusion (BM25 + FAISS + RRF) improves early-rank placement for terminology-sensitive biomedical queries.
- **Limitation:** Precision and F1 do not surpass rerank_only, indicating improved ordering does not always increase total relevant set capture.

**Reference floor:** `none` (baseline)
- **Role:** canonical control condition for all delta calculations.
- **Interpretation:** provides stable mid-range behaviour without augmentation artifacts.
- **Limitation:** underperforms rerank_only and hybrid_only on most ranking and coverage criteria.

**Mixed profile:** `hyde`
- **Success profile:** slight MRR improvement over baseline.
- **Struggle profile:** lower Recall, F1, nDCG, and Hit Rate relative to baseline.
- **Interpretation:** synthetic semantic expansion can improve first-hit localisation for some queries, but may induce context drift for broader relevance capture.

**Negative control (expected degradation):** `keyword_expansion`
- **Outcome:** lowest values across all tracked macro metrics.
- **Interpretation:** token inflation without semantic coherence disrupts dense-vector neighbourhood alignment in biomedical space.
- **Research value:** validates the ablation matrix by demonstrating that not all augmentation is beneficial.

### Verification Layer Status

- `none`: rewrite_status remained Original Query as expected.
- `keyword_expansion` and `hyde`: rewrite_status audited through `query_was_rewritten` and `rewrite_status` columns.
- `Delta_vs_None`: row-level discrepancy analysis available for all non-baseline runs to identify where each strategy helped, hurt, or produced no change.

> **Current interpretation boundary:** memory_only results are to be integrated after completion of the final isolated conversational-memory run and significance testing pass.

### ScienceQA External Validation (100 rows, no GT evidence columns)

To stress-test transferability beyond the biomedical benchmark, the same thesis matrix settings were executed on `scienceqa_eval_dataset.xlsx` (100 processed rows) through `/eval` with locked `top_k=5`, chunking `600/50`, and embedding model `BAAI/bge-large-en-v1.5`.

Because this dataset export does not include ground-truth evidence columns in the expected schema, retrieval metrics such as Precision@k/Recall@nDCG are not computable for this run. The comparison therefore uses the available macro outputs:

- `QA Support Hit`: fraction of rows where at least one retrieved chunk supports the QA item.
- `QA Support MRR`: early-rank quality of the first supporting chunk.
- `QA Choice Acc`: final multiple-choice answer accuracy.

| Strategy | QA Support Hit | QA Support MRR | QA Choice Acc | Processed Rows |
|---|:---:|:---:|:---:|:---:|
| `none` | 0.0000 | 0.0000 | 0.3000 | 100 |
| `rerank_only` | 0.0000 | 0.0000 | 0.3000 | 100 |
| `keyword_expansion` | **0.1000** | 0.0250 | 0.2000 | 100 |
| `hyde` | 0.0500 | **0.0500** | **0.3500** | 100 |

#### Delta vs Baseline (`none`)

| Strategy | $\Delta$ QA Support Hit | $\Delta$ QA Support MRR | $\Delta$ QA Choice Acc |
|---|:---:|:---:|:---:|
| `rerank_only` | +0.0000 | +0.0000 | +0.0000 |
| `keyword_expansion` | +0.1000 | +0.0250 | -0.1000 |
| `hyde` | +0.0500 | +0.0500 | +0.0500 |

Interpretation shortcut:
- `hyde` is the only strategy that improved both support quality and final answer accuracy relative to baseline.
- `keyword_expansion` improved support detection but reduced end-task accuracy, indicating a support-to-answer conversion gap.
- `rerank_only` produced no measurable movement on this slice.

### ScienceQA Strategy Effects (Interpretation)

**Best end-task accuracy:** `hyde`
- `QA Choice Acc` increased from 0.3000 (baseline) to **0.3500**.
- Despite fewer support hits than keyword expansion, HyDE achieved the best `QA Support MRR`, indicating that when support is found, it is found earlier in ranking.
- Practical implication: in this dataset, semantically richer rewrites improved answer selection quality more than raw support-hit frequency.

**Highest support-hit frequency but weakest answering:** `keyword_expansion`
- Highest `QA Support Hit` (0.1000), but low `QA Support MRR` (0.0250) and the lowest `QA Choice Acc` (0.2000).
- This pattern suggests expanded keywords may retrieve loosely related context but not consistently rank high-value evidence early enough to improve answer decisions.
- Practical implication: more retrieved support signals did not translate into better final QA outcomes.

**No measurable gain over baseline:** `rerank_only`
- Identical to baseline on all available metrics in this run (`QA Choice Acc` 0.3000, support metrics 0.0000).
- Likely interpretation: reranking cannot add value when the candidate pool itself lacks supporting chunks for this schema/configuration slice.
- Practical implication: reranking remains dependent on upstream retrieval recall.

**Reference behavior:** `none`
- Serves as control with moderate answer accuracy (0.3000) but no measured support hits.
- Confirms that this dataset/configuration pairing is challenging for evidence retrieval under current prompt+index settings.

### ScienceQA Conclusion (Current Evidence)

For this new ScienceQA dataset slice, `hyde` is the most effective strategy for end-task QA accuracy, `keyword_expansion` improves support-hit frequency without downstream answer gains, and `rerank_only` shows no standalone benefit under the present retrieval candidate quality. Since GT evidence columns were missing, these findings should be treated as **external validation signals** rather than full retrieval-grounded ablation evidence; adding GT evidence annotations would enable full thesis-metric parity in a future pass.

### Recommended Next Optimization Pass (ScienceQA)

1. **Promote `hyde` as the ScienceQA default profile** for answer-generation runs, since it currently maximizes `QA Choice Acc`.
2. **Tune HyDE generation temperature in a narrow band (0.1-0.3)** and compare variance across 3 repeated seeds/runs to test stability.
3. **Run a retrieval-depth sweep (`top_k` = 5, 8, 10)** for `hyde` only, to verify whether support-hit growth converts into additional answer-accuracy gains.
4. **Add GT evidence columns to ScienceQA export schema** so Precision/Recall/MRR/nDCG can be computed and compared at full thesis-metric parity.
5. **Evaluate `hyde + rerank` as a controlled compound profile** after isolation studies, to test whether stronger candidate generation plus cross-encoder ordering improves both support and final QA accuracy.

### Threats to Validity and Reproducibility Notes

- **Missing evidence labels:** GT evidence columns were absent, so standard retrieval metrics are unavailable for this run.
- **Single-run sensitivity:** each profile currently reflects a single macro pass; variance across repeated runs is not yet reported.
- **Prompt/model dependence:** results are tied to the current model stack and generation settings, especially rewrite temperature.
- **Dataset transfer caution:** cross-domain transfer (biomedical -> ScienceQA) may change ranking behavior and answer calibration.

To improve reproducibility for thesis reporting:
- record script commit hash, dataset hash, and exact endpoint flags per run;
- run each profile at least 3 times and report mean ± std;
- keep fixed ingestion config (chunk size/overlap) constant within each comparison block.

### Final Thesis Claims (Current Evidence Base)

**Claim 1 (Primary Benchmark Superiority):**
On the biomedical benchmark under locked retrieval depth and fixed embedding settings, isolated cross-encoder reranking (`rerank_only`) provides the strongest overall retrieval-quality gains across most core metrics, supporting the claim that post-retrieval semantic ordering is a high-impact intervention in dense RAG pipelines.

**Claim 2 (Early-Rank Optimization Effect):**
Hybrid sparse-dense fusion (`hybrid_only`) delivers the strongest early-rank quality (MRR/nDCG pattern), indicating that lexical-semantic complementarity is particularly effective for first-hit prioritization in terminology-sensitive scientific queries.

**Claim 3 (Dataset-Dependent Query Rewriting Utility):**
In ScienceQA external validation, HyDE is the only tested strategy that improves both support-quality indicators and end-task answer accuracy relative to baseline, while keyword expansion increases support hits but degrades final answer accuracy. This supports a dataset-dependent view of rewrite effectiveness rather than a universal benefit assumption.

**Claim 4 (Dependency Constraint for Reranking):**
Where upstream retrieval recall is weak (ScienceQA slice in current schema), reranking alone does not improve outcomes, reinforcing that rerankers optimize candidate order but cannot recover absent evidence.

**Claim 5 (Methodological Contribution):**
The strict ablation-control design (explicit toggle isolation, locked top-k, rewrite verification auditing, and row-level delta analysis) constitutes a reproducible evaluation protocol for separating true strategy effects from configuration leakage.

### One-Page Dissertation Results Summary

| Dimension | Biomedical Benchmark (Primary) | ScienceQA External Validation (Current Slice) | Thesis-Level Interpretation |
|---|---|---|---|
| Best overall profile | `rerank_only` | `hyde` (by QA Choice Acc) | Best strategy is objective- and dataset-dependent |
| Best early-rank profile | `hybrid_only` (MRR/nDCG leader) | `hyde` (best QA Support MRR) | First-hit optimization can arise from different mechanisms across datasets |
| Baseline (`none`) role | Stable control for delta computation | Moderate QA accuracy, zero support hits | Necessary anchor for causal comparison |
| `keyword_expansion` effect | Negative-control degradation pattern | Higher support hits, lower answer accuracy | More retrieval signals do not guarantee better decisions |
| `rerank_only` effect | Strong positive on primary benchmark | No change vs baseline | Reranking is bounded by candidate recall quality |
| Evidence completeness | Full retrieval metric set available | Retrieval metrics unavailable (missing GT evidence schema) | External findings are indicative, not fully retrieval-grounded |
| Reproducibility status | Structured and auditable | Structured but schema-limited | Protocol is strong; ScienceQA labeling is the main gap |

### Limitations and Future Work

**Current Limitations**
- ScienceQA run lacks GT evidence columns, preventing full retrieval-grounded metric parity.
- Current ScienceQA comparisons are single-pass macros without repeated-run variance estimates.
- Cross-domain transfer from biomedical indexing assumptions may underfit science education question styles.
- Compound profiles (for example, `hyde + rerank`) are not yet fully mapped in the same controlled grid.

**Planned Future Work**
1. Introduce GT evidence annotations for ScienceQA to enable Precision/Recall/F1/MRR/nDCG parity with the primary benchmark.
2. Run repeated trials per profile and report mean ± std with confidence intervals.
3. Extend the ablation matrix with controlled compound settings (`hyde + rerank`, `hyde + hybrid`, `hyde + rerank + hybrid`) after isolated baselines are locked.
4. Add statistical significance testing for pairwise profile deltas (for example, bootstrap CIs and non-parametric paired tests).
5. Evaluate robustness under ingestion perturbations (chunk size/overlap sweeps) while holding runtime toggles fixed.
6. Add calibration analysis linking retrieval support metrics to final answer correctness to quantify support-to-decision conversion efficiency.

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

# Optional but recommended news enrichment layers
# CryptoPanic signup: https://cryptopanic.com/developers/api/
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key

# CoinDesk Data signup: https://developers.coindesk.com/
COINDESK_DATA_API_KEY=your_coindesk_data_api_key

# Optional scanner/runtime tuning
SCAN_TOP_N=20
SCALPING_MIN_VOLATILITY_PCT=1.5
TECHNICAL_MISSING_FIELD_THRESHOLD=0.4
```

Notes:
- `GROQ_API_KEY` is required for LLM summaries/reports.
- `CRYPTOPANIC_API_KEY` and `COINDESK_DATA_API_KEY` are optional but recommended; the news pipeline works keylessly via RSS first, then enriches when keys exist.

### Running the Backend

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload
```

### Running the Crypto Research CLI

```bash
python agent.py
```

### Running the Crypto Research Streamlit Dashboard

```bash
streamlit run crypto_ui.py
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