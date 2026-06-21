# Doctoral Research Execution Playbook
## RAG Ablation Study — Biomedical Question Answering

**Purpose:** Sequential, dependency-ordered execution protocol for completing the 6-way ablation study, statistical validation, and publication asset generation.

**Scope:** Covers all remaining R&D milestones from environment lock through final reproducibility packaging.

**Convention:** Mark `[ ]` as `[x]` upon verified completion. Never skip a task that is a declared prerequisite of the next.

---

## Phase 1 — Vector Space Reproducibility Controls

### Task 1.1 — Freeze Runtime Environment
- **Prerequisite:** None
- **Scientific rationale:** Embedding model behaviour, tokenisation, and numerical precision are all package-version-dependent. Thesis reviewers require a locked dependency state to reproduce results.
- **Execution:**
  1. Activate the evaluation virtual environment.
  2. Export the full dependency snapshot: `pip freeze > requirements-lock-final.txt`
  3. Record Python interpreter version, OS version, and CPU/RAM specs in a dedicated `ENVIRONMENT.md` note file.
- **Done when:** `requirements-lock-final.txt` exists with pinned versions and `ENVIRONMENT.md` records interpreter and hardware context.

- [ ] **1.1 Complete**

---

### Task 1.2 — Align Embedding Model Configuration
- **Prerequisite:** None
- **Scientific rationale:** All preliminary runs were executed under `all-MiniLM-L6-v2` (384d). The thesis architectural claim requires `BAAI/bge-large-en-v1.5` (1024d). Any mismatch between the claimed and actual embedding model invalidates the thesis experiment as a whole.
- **Execution:**
  1. Open `src/config.py`.
  2. Set `EMBED_MODEL = "BAAI/bge-large-en-v1.5"` (uncomment the correct line).
  3. Confirm `EMBED_DIM` resolves to `1024` via the conditional expression.
  4. Save file.
- **Verification:** Run `py -c "from src.config import EMBED_MODEL, EMBED_DIM; print(EMBED_MODEL, EMBED_DIM)"` and confirm output.
- **Done when:** Config output shows `BAAI/bge-large-en-v1.5 1024`.

- [ ] **1.2 Complete**

---

### Task 1.3 — Purge Stale FAISS Index and Re-Ingest Corpus
- **Prerequisite:** Task 1.2 must be complete.
- **Scientific rationale:** The FAISS index encodes the embedding geometry of the corpus. An index built under MiniLM is geometrically incompatible with queries embedded under bge-large. Mixing embeddings from different models produces undefined inner-product similarity scores and will corrupt all retrieval results.
- **Execution:**
  1. Delete the stale index: `rmdir /s /q storage\faiss` (Windows) or `rm -rf storage/faiss` (Linux/macOS).
  2. Start the FastAPI backend: `uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload`
  3. Re-ingest the full corpus via the `/upload` endpoint or the ingestion pipeline with the corpus PDF.
  4. Verify in backend logs that bge-large-en-v1.5 is loaded and the new index size is reported.
- **Done when:** Backend startup log shows `Loading embedding model: BAAI/bge-large-en-v1.5` and retrieval on a test query returns coherent passages.

- [ ] **1.3 Complete**

---

## Phase 2 — Controlled 6-Way Experiment Execution

> All runs in this phase share the same fixed parameters:
> top_k = 5 (locked), chunk_size = 600, chunk_overlap = 50, embedding = bge-large-en-v1.5 (1024d).
> Restart the backend between profile groups if any backend code was modified.

### Task 2.1 — Restart Backend and Validate Payload Controls
- **Prerequisite:** Phase 1 complete.
- **Execution:**
  1. Restart the FastAPI service.
  2. Send a test request and inspect the log line beginning with `EVAL ENDPOINT:`.
  3. Confirm fields: `rewriter=`, `strategy=`, `hybrid=`, `rerank=`, `chat_history=`.
- **Done when:** Log output exactly reflects the configured profile toggles with no unexpected overrides.

- [ ] **2.1 Complete**

---

### Task 2.2 — Run Baseline Profile (none)
- **Prerequisite:** Tasks 1.3 and 2.1 complete. This run must exist before any delta analysis.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "none"` in `docs/3.0/script.py`.
  2. Confirm feature toggles auto-derive: rewriter=False, strategy=none, hybrid=False, rerank=False, memory=False.
  3. Execute: `cd docs/3.0 && py script.py`
  4. Confirm `Thesis_Metrics_none.xlsx` is created with all three sheets: Row_Metrics, Macro_Averages, Delta_vs_None.
  5. Verify `rewrite_status` column in Row_Metrics is entirely "Original Query".
- **Done when:** Workbook exists, all 98+ rows are valid, and Macro_Averages records the run metadata.

- [ ] **2.2 Complete**

---

### Task 2.3 — Run Keyword Expansion Profile (Negative Control)
- **Prerequisite:** Task 2.2 complete.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "keyword_expansion"`.
  2. Execute: `py script.py`
  3. Inspect `Thesis_Metrics_keyword_expansion.xlsx` → Row_Metrics → `rewrite_status` column.
  4. Verify the `rewrite_success_rate` in Macro_Averages sheet is > 0.90.
  5. Verify `Delta_vs_None` sheet rows have alignment with the none baseline by row index or question key.
- **Done when:** Workbook exists with no ERROR-level rewrite collapse events in the backend log.

- [ ] **2.3 Complete**

---

### Task 2.4 — Run HyDE Experimental Profile
- **Prerequisite:** Task 2.2 complete.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "hyde"`.
  2. Execute: `py script.py`
  3. Confirm rewrite success rate in Macro_Averages is > 0.90.
  4. Inspect Delta_vs_None for rows where `overall_impact = "Helped"` — these are your strongest thesis evidence rows.
  5. Note MRR vs Recall@k trade-off direction in Macro_Averages as this is the key finding to report.
- **Done when:** Workbook exists, rewrite tracking confirms active rewrites, and Delta_vs_None is populated.

- [ ] **2.4 Complete**

---

### Task 2.5 — Run Isolated Component: Reranker Only
- **Prerequisite:** Task 2.2 complete.
- **Critical isolation check:** Payload must show `rerank=true`, all other flags false, strategy=none. Inspect first log line of the run to verify.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "rerank_only"`.
  2. Execute: `py script.py`
  3. Confirm rewrite_status column is entirely "Original Query" (no LLM query augmentation).
  4. Compare Macro_Averages Precision@5 and MRR with baseline to assess cross-encoder contribution.
- **Done when:** Workbook exists with confirmed Original Query status across all rows.

- [ ] **2.5 Complete**

---

### Task 2.6 — Run Isolated Component: Hybrid Search Only
- **Prerequisite:** Task 2.2 complete.
- **Critical isolation check:** Payload must show `hybrid=true`, all other flags false, strategy=none.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "hybrid_only"`.
  2. Execute: `py script.py`
  3. Note Recall@k change relative to baseline — sparse-dense fusion is expected to improve recall for short exact-term queries.
  4. Inspect Delta_vs_None for which query types benefit most from BM25 fusion.
- **Done when:** Workbook exists with confirmed isolation.

- [ ] **2.6 Complete**

---

### Task 2.7 — Run Isolated Component: Conversational Memory Only
- **Prerequisite:** Task 2.2 complete.
- **Note:** memory_only passes context history into the query rewriter stage. Since the benchmark is stateless (each query is independent), the expected result is minimal or no improvement — this profile acts as a **null treatment control** for conversational enhancement.
- **Execution:**
  1. Set `EXPERIMENT_PROFILE = "memory_only"`.
  2. Execute: `py script.py`
  3. Confirm rewrite_status is "Original Query" throughout (history enrichment does not alter embedding without the rewriter).
- **Done when:** Workbook exists and confirms null-treatment hypothesis.

- [ ] **2.7 Complete**

---

## Phase 3 — Verification and Statistical Validation

### Task 3.1 — Audit Rewrite Verification Tracking Layer
- **Prerequisite:** Tasks 2.3 and 2.4 complete.
- **Execution:**
  1. For each of the six workbooks, open Row_Metrics and check `rewrite_status` and `query_was_rewritten` columns.
  2. Expected mapping:
     - none, rerank_only, hybrid_only, memory_only → "Original Query" for all rows.
     - keyword_expansion, hyde → "Rewritten Query" for the majority of rows.
  3. Any row in an augmentation profile with "Original Query" is a verified silent collapse event. Count and document these as data quality notes.
- **Done when:** Summary table produced noting rewrite success rate per profile.

- [ ] **3.1 Complete**

---

### Task 3.2 — Validate Delta_vs_None Sheet Row Alignment
- **Prerequisite:** All Phase 2 runs complete.
- **Execution:**
  1. For each experimental workbook, open Delta_vs_None.
  2. Spot-check 10 random rows: manually compute `f1_score - f1_score_none` and confirm it matches `delta_f1_score`.
  3. Confirm `impact_f1_score` label matches sign of `delta_f1_score`.
  4. Verify `overall_impact = "Helped"` rows have positive `net_delta_sum`.
- **Done when:** No arithmetic or sign errors found in random sample.

- [ ] **3.2 Complete**

---

### Task 3.3 — Perform Paired Statistical Significance Tests
- **Prerequisite:** Tasks 3.1 and 3.2 complete.
- **Scientific rationale:** Macro averages alone are insufficient for rigorous peer-reviewed claims. Paired tests operating on the row-level delta arrays provide the p-values required to state whether differences are statistically distinguishable from noise.
- **Execution:**
  1. Load `delta_f1_score`, `delta_recall_at_k`, `delta_ndcg_at_k`, `delta_mrr_score` arrays from each Delta_vs_None sheet.
  2. Run two-sided Wilcoxon signed-rank test (or sign test for binary metrics): `scipy.stats.wilcoxon(delta_array)`.
  3. Record: test statistic, p-value, effect direction (positive or negative mean delta).
  4. Apply Bonferroni correction if reporting multiple metrics simultaneously (alpha / num_metrics).
  5. Summarise in a table: Profile | Metric | Mean Delta | p-value | Significant (yes/no at alpha=0.05).
- **Significance interpretation:** p < 0.05 supports a claim of statistically distinguishable difference from baseline. p >= 0.05 requires the phrase "directional trend without statistical significance at the 0.05 threshold."
- **Done when:** Significance table is complete with p-values for at minimum F1, Recall@k, MRR, and nDCG for all five experimental profiles.

- [ ] **3.3 Complete**

---

## Phase 4 — Publication Asset Generation

### Task 4.1 — Extend Chart Generator to Include All Six Profiles
- **Prerequisite:** All Phase 2 runs complete.
- **Execution:**
  1. Open `docs/3.0/generate_charts.py`.
  2. Add the three new workbooks to `STRATEGY_FILES`:
     - `"rerank_only"`: `Thesis_Metrics_rerank_only.xlsx`
     - `"hybrid_only"`: `Thesis_Metrics_hybrid_only.xlsx`
     - `"memory_only"`: `Thesis_Metrics_memory_only.xlsx`
  3. Update the panel count from 2x3 to 3x3 or use a single grouped multi-bar chart layout.
  4. Ensure strategy labels on the x-axis are readable and ordered: none, keyword_expansion, hyde, rerank_only, hybrid_only, memory_only.
- **Done when:** Chart code has no import or key errors.

- [ ] **4.1 Complete**

---

### Task 4.2 — Generate Final Publication-Ready Charts
- **Prerequisite:** Task 4.1 complete.
- **Execution:**
  1. `cd docs/3.0 && py generate_charts.py`
  2. Open `thesis_ablation_results.png` and verify:
     - All six strategies are visible.
     - Numerical value annotations appear on each bar to 3 decimal places.
     - Axis labels, legend, and title are correctly rendered.
     - Output at 300 DPI is legible at print scale (A4/letter figure size).
- **Done when:** PNG file is confirmed readable and all bars are correctly labelled.

- [ ] **4.2 Complete**

---

### Task 4.3 — Generate Significance Results Table as Publication Asset
- **Prerequisite:** Task 3.3 complete.
- **Execution:**
  1. Format the significance table from Task 3.3 as a LaTeX or Markdown table suitable for direct paste into the thesis.
  2. Include columns: Profile, Metric, Macro_Baseline, Macro_Experimental, Mean_Delta, p-value, Significant.
- **Done when:** Table is formatted and saved to `docs/3.0/significance_results.md`.

- [ ] **4.3 Complete**

---

## Phase 5 — Thesis Write-Up

### Task 5.1 — Write Results Section Per Profile
- **Prerequisite:** Phases 3 and 4 complete.
- **Content required per profile:**
  - Macro metric table (copy from Macro_Averages sheet).
  - Delta interpretation: which metrics improved, which degraded, net direction.
  - Significance statement from Task 3.3.
  - Rewrite success rate and any collapse events noted.
  - Mechanistic interpretation: why did this architecture behave as observed?
- **Profiles requiring specific narrative constructs:**
  - **none (Baseline):** Establish the reference floor. Do not interpret as good or bad; it is the measurement origin.
  - **keyword_expansion:** Frame as negative control confirming that unstructured term proliferation is corruptive in dense vector spaces.
  - **hyde:** Frame around the MRR improvement vs Recall trade-off. Argue that HyDE improves retrieval rank precision, not set coverage.
  - **rerank_only:** Assess whether cross-encoder post-scoring corrects FAISS ordering errors.
  - **hybrid_only:** Assess whether sparse signal (BM25) recovers queries that dense search misses.
  - **memory_only:** Confirm null-treatment result for stateless benchmark; note applicability in conversational deployments.
- **Done when:** Draft text covers all six profiles with metric tables, significance outcomes, and mechanistic narratives.

- [ ] **5.1 Complete**

---

### Task 5.2 — Write Methodology Section with Anti-Leakage Controls
- **Prerequisite:** None (can be written in parallel with Phase 2 execution).
- **Required content:**
  - Fixed parameter table (top_k=5, chunk_size=600, chunk_overlap=50, bge-large-en-v1.5, 1024d).
  - Profile isolation mechanism: how PROFILE_PRESETS enforce mutual exclusion.
  - Rewrite Verification Tracking Layer: explain the three-tuple return, the lexicographic difference check, and the silent collapse detection protocol.
  - Delta_vs_None computation engine: row alignment strategy, metric subtraction formula, impact label logic.
  - Fuzzy matching methodology: three-tier strategy with Jaccard threshold 0.55 and 7-word phrase window.
  - Request pacing: timeout, retry/backoff, row delay to prevent rate-limiting contamination.
- **Done when:** Methodology section is complete and references all relevant source files by name.

- [ ] **5.2 Complete**

---

### Task 5.3 — Assemble Final Reproducibility Package
- **Prerequisite:** All prior tasks complete.
- **Contents to bundle:**
  - All six `Thesis_Metrics_*.xlsx` workbooks.
  - `thesis_ablation_results.png` (300 DPI).
  - `significance_results.md` table.
  - Frozen `requirements-lock-final.txt`.
  - `ENVIRONMENT.md` hardware and interpreter note.
  - `docs/3.0/script.py` and `docs/3.0/generate_charts.py` at their final committed versions.
  - A `RUNBOOK.md` with the exact command sequence to reproduce any single run from scratch.
- **Done when:** Package is zipped or archived and a second person could execute the full study by following RUNBOOK.md alone.

- [ ] **5.3 Complete**

---

## Quick Daily Execution Checklist

Before each run session:
1. Confirm embedding model and dimension are correct in `src/config.py`.
2. Confirm FastAPI backend is running and responsive at `http://localhost:8001/health`.
3. Set `EXPERIMENT_PROFILE` in `docs/3.0/script.py`.
4. Inspect the first log line of the run for payload toggles.
5. After the run, verify the workbook has three sheets and the correct metadata in Macro_Averages.

---

## Priority Schedule (If Time-Constrained)

Execute in this minimum viable order:

| Priority | Task | Justification |
|:---:|---|---|
| 1 | 1.2 + 1.3 | Without correct embedding, all results are invalid |
| 2 | 2.2 (none) | Required baseline for all delta analysis |
| 3 | 2.4 (hyde) | Core experimental hypothesis |
| 4 | 2.5 (rerank_only) | Second-strongest architectural variable |
| 5 | 3.3 | Statistical significance for p-values in paper |
| 6 | 4.2 | Publication figure |
| 7 | 5.1 + 5.2 | Thesis text |
| 8 | 2.3, 2.6, 2.7, 5.3 | Complete picture and packaging |
