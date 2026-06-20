# Thesis RAG Next Steps Todo

This file is your execution playbook for the next phase.
Follow tasks in order from top to bottom.

## How To Use This File
- Mark each checkbox as complete when done.
- Do not skip dependency tasks.
- After each major block, commit or backup outputs.

## Phase 1: Environment And Reproducibility Lock

- [ ] 1. Freeze runtime environment
  - Why: Keeps results reproducible for thesis defense.
  - How:
    - Activate your evaluation environment.
    - Export package list to a frozen requirements file.
    - Record Python version and OS details in your notes.
  - Done when:
    - You have an updated requirements lock file and a short environment note.

- [ ] 2. Confirm embedding model alignment with thesis claim
  - Why: Current config can invalidate thesis claims if mismatch exists.
  - How:
    - Open [src/config.py](src/config.py).
    - Ensure EMBED_MODEL is set to BAAI/bge-large-en-v1.5.
    - Ensure embedding dimension expectation matches 1024.
  - Done when:
    - Config is aligned and saved.

- [ ] 3. Rebuild vector index after embedding changes
  - Why: New embedding model requires full re-embedding for fair evaluation.
  - How:
    - Clear existing FAISS artifacts.
    - Re-run ingestion pipeline on full corpus.
    - Verify index and metadata were regenerated.
  - Done when:
    - Retrieval works and index corresponds to new embedding model.

## Phase 2: Controlled Experiment Execution

- [ ] 4. Restart backend and verify eval payload controls are active
  - Why: Recent endpoint changes require restart.
  - How:
    - Restart FastAPI service.
    - Run one test query and verify log line prints active strategy and toggles.
  - Done when:
    - Logs show expected values for rewriter, strategy, hybrid, rerank, chat memory.

- [ ] 5. Run baseline profile
  - Why: Baseline is reference for all deltas.
  - How:
    - Set EXPERIMENT_PROFILE to none in [docs/3.0/script.py](docs/3.0/script.py).
    - Run evaluation script.
    - Confirm output file exists: Thesis_Metrics_none.xlsx.
  - Done when:
    - Workbook contains Row_Metrics, Macro_Averages, Delta_vs_None sheets.

- [ ] 6. Run rewriting profiles
  - Why: Completes 3-way rewriting comparison.
  - How:
    - Set EXPERIMENT_PROFILE to keyword_expansion and run.
    - Set EXPERIMENT_PROFILE to hyde and run.
  - Done when:
    - Both workbooks exist and include rewrite_status and query_was_rewritten columns.

- [ ] 7. Run isolated feature profiles
  - Why: Measures individual impact without cross-contamination.
  - How:
    - Run rerank_only profile.
    - Run hybrid_only profile.
    - Run memory_only profile.
  - Isolation checks:
    - Query rewriting must be false.
    - Rewriting strategy must be none.
    - Only the chosen feature is true.
  - Done when:
    - All three workbooks are generated with expected toggles in Macro_Averages.

## Phase 3: Quality Assurance And Statistical Validation

- [ ] 8. Validate rewrite tracking integrity
  - Why: Confirms LLM rewrite actually happened when expected.
  - How:
    - In Row_Metrics, verify rewrite_status values by profile.
    - none, rerank_only, hybrid_only, memory_only should be Original Query.
    - keyword_expansion and hyde should mostly be Rewritten Query.
  - Done when:
    - Behavior matches profile design.

- [ ] 9. Validate Delta_vs_None sheet correctness
  - Why: Delta sheet is key for thesis interpretation.
  - How:
    - Open Delta_vs_None for each experimental workbook.
    - Spot-check random rows against baseline row metrics.
    - Confirm helped and hurt indicators match sign of delta.
  - Done when:
    - No row alignment or sign errors found.

- [ ] 10. Perform paired significance tests
  - Why: Needed for rigorous claims.
  - How:
    - Run paired tests per metric versus baseline.
    - Use sign test or Wilcoxon signed-rank for row-level deltas.
    - Save p-values and effect directions.
  - Done when:
    - You have a table with p-values and interpretation per profile.

## Phase 4: Visualization And Reporting Assets

- [ ] 11. Generate thesis chart assets
  - Why: Publication-ready visuals for results chapter.
  - How:
    - Run [docs/3.0/generate_charts.py](docs/3.0/generate_charts.py).
    - Confirm output image exists: thesis_ablation_results.png.
  - Done when:
    - Chart is readable at print scale and labels are correct.

- [ ] 12. Add extended charts for isolated profiles
  - Why: Current chart script covers none, keyword_expansion, hyde only.
  - How:
    - Extend strategy file list in chart script for rerank_only, hybrid_only, memory_only.
    - Regenerate charts.
  - Done when:
    - Final figure compares all required strategies.

## Phase 5: Thesis Write-Up And Packaging

- [ ] 13. Write results narrative per profile
  - Why: Converts raw metrics into defendable conclusions.
  - How:
    - Summarize macro metrics, delta trends, and significance outcomes.
    - Explicitly state where each profile helps or hurts.
    - Include rewrite success rate interpretation.
  - Done when:
    - Results text and table references are complete.

- [ ] 14. Add methodology safeguards section
  - Why: Shows scientific rigor and fairness controls.
  - How:
    - Document fixed top_k, chunking, timeout policy, retry policy, and profile isolation logic.
    - Mention how Delta_vs_None is computed and aligned.
  - Done when:
    - Methodology section clearly explains anti-leakage controls.

- [ ] 15. Final reproducibility package
  - Why: Essential for supervisor review and future publication.
  - How:
    - Bundle scripts, config snapshot, run logs, and all output workbooks.
    - Add a short runbook with exact run order and expected files.
  - Done when:
    - Another person can reproduce your outputs from scratch.

## Quick Daily Execution Loop
- Choose profile.
- Run script.
- Validate workbook sheets.
- Append findings to thesis notes.
- Commit artifacts.

## Priority Recommendation
If time is limited, prioritize in this order:
1. Embedding model alignment and re-index.
2. Re-run baseline plus hyde plus rerank_only.
3. Statistical tests and delta interpretation.
4. Final chart generation and write-up.
