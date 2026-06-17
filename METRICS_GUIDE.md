# Comprehensive RAG Evaluation Metrics - User Guide

## Overview

Your RAG system now includes a complete evaluation metrics framework that automatically computes and displays performance metrics for every query. The metrics are organized into three comprehensive sections in the UI.

## Three Metrics Sections

### 📈 Section 1: Standard Evaluation Metrics

Displays traditional Information Retrieval (IR) and answer quality metrics:

| Metric | Range | Description | Interpretation |
|--------|-------|-------------|-----------------|
| **Recall@k** | 0-1 | Proportion of relevant docs in top-k results | Higher = better coverage of relevant docs |
| **MRR** | 0-1 | Reciprocal rank of first relevant document | Higher = relevant doc appears earlier |
| **nDCG@k** | 0-1 | Ranking quality with logarithmic discounting | Higher = better ranking of relevant docs |
| **Hit Rate** | 0/1 | Whether any relevant doc in top-k | 1 = at least one relevant doc found |
| **Exact Match** | 0/1 | Whether answer exactly matches reference (if available) | 1 = perfect match |
| **F1 Score** | 0-1 | Token-level F1 between answer and reference | Higher = better token overlap |

**When to Check:**
- Monitor Recall@k to ensure retrieval is comprehensive
- Use MRR to optimize for early relevance
- Track nDCG for ranking quality
- Watch Hit Rate for basic success rate

---

### 🎯 Section 2: RAG-Specific Quality Metrics

Metrics specifically designed for RAG system evaluation:

| Metric | Range | Description | What it Measures |
|--------|-------|-------------|------------------|
| **Faithfulness** | 0-1 | Degree answer is supported by sources | Token overlap between answer & sources. Higher = more grounded |
| **Answer Relevance** | 0-1 | Degree answer addresses the question | Key terms from Q appearing in A. Higher = more focused |
| **Context Precision** | 0-1 | Proportion of sources supporting answer | % of retrieved docs with significant overlap. Higher = better context selection |

**When to Check:**
- **Faithfulness** increases → Answers are more grounded in sources ✅
- **Answer Relevance** increases → Answers are more focused on questions ✅
- **Context Precision** increases → Retrieved contexts are more selective and useful ✅

**Target Benchmarks:**
- Faithfulness > 0.7 = Good source grounding
- Answer Relevance > 0.6 = Adequate question focus
- Context Precision > 0.5 = Reasonable context selection

---

### ⚙️ Section 3: System Configuration

Captured configuration and environment for each query:

**Hardware & Models:**
- CPU count, RAM, OS
- Embedding model used (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- LLM model used (e.g., `llama3.2:3b`)

**Retrieval Settings:**
- Chunk size (tokens per chunk)
- Chunk overlap (token overlap between chunks)
- Top-k (number of documents retrieved)

**Embedding Configuration:**
- Embedding dimensions (typically 384 or 1024)
- LLM temperature (controls randomness, usually 0.3)

**Timestamp:**
- Query execution time for performance tracking

**Why This Matters:**
- Track how configuration changes affect metrics
- Identify performance bottlenecks (hardware limitations)
- Compare results across different models
- Understand context window constraints

---

## How Metrics Are Calculated

### Retrieval Metrics

**Recall@k:** How many relevant documents are in your top-k results?
```
Recall@k = (# relevant docs found in top-k) / (# total relevant docs)
```
- Range: 0-1
- Example: If you need 5 docs and find 3 in top-3 → Recall@3 = 0.6

**MRR (Mean Reciprocal Rank):** How early is the first relevant document?
```
MRR = 1 / (position of first relevant doc)
```
- Rewards finding relevant docs earlier
- Example: First relevant at position 1 → MRR = 1.0
- Example: First relevant at position 3 → MRR = 0.333

**nDCG (Normalized Discounted Cumulative Gain):** How well-ranked are results?
```
DCG@k = Σ(relevance_i / log₂(i+1))
nDCG@k = DCG@k / IDCG@k
```
- Accounts for ranking position (better docs should rank higher)
- Range: 0-1
- Normalized so perfect ranking = 1.0

**Hit Rate:** Did you find ANY relevant document?
```
Hit Rate = 1 if any relevant doc in top-k, 0 otherwise
```
- Binary metric
- Basic success rate check

### Answer Quality Metrics

**Exact Match:** Does the answer exactly match the reference?
```
EM = 1 if normalize(answer) == normalize(reference), 0 otherwise
```
- Strict metric (0 or 1)
- Normalized: lowercase, whitespace, punctuation removed

**F1 Score:** Token-level similarity between answer and reference
```
Precision = (tokens in common) / (tokens in answer)
Recall = (tokens in common) / (tokens in reference)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Range: 0-1
- Rewards overlap without requiring exact match

### RAG-Specific Metrics

**Faithfulness:** How much of the answer comes from the sources?
```
Faithfulness = (answer tokens found in sources) / (total answer tokens)
```
- Range: 0-1
- Ensures generated answers don't hallucinate

**Answer Relevance:** How well does the answer address the question?
```
Answer_Relevance = (question key terms in answer) / (total question key terms)
```
- Range: 0-1
- Ignores common words (the, is, a, etc.)
- Ensures answer stays on topic

**Context Precision:** What proportion of sources support the answer?
```
Supporting_sources = # sources with ≥20% token overlap with answer
Context_Precision = Supporting_sources / total_sources
```
- Range: 0-1
- Identifies if retrieved context is relevant

---

## Performance Analysis Workflow

### 1. **Daily Monitoring**
After each query, check:
- ✅ **Faithfulness** > 0.7 (answer is grounded)
- ✅ **Context Precision** > 0.5 (context is selective)
- ✅ **Hit Rate** = 1.0 (found some relevant doc)

### 2. **Optimization Target**
Work on improving:
- If Recall@k < 0.5 → Increase chunk count or reranking
- If Faithfulness < 0.5 → Improve document retrieval
- If Answer Relevance < 0.4 → Query rewriting needs improvement

### 3. **Configuration Testing**
Track metric changes when you modify:
- Chunk size (affects context window)
- Embedding model (affects retrieval quality)
- Top-k value (affects comprehensiveness vs. noise)
- Reranker (affects final ranking quality)

### 4. **System Configuration Comparison**
Use the system config section to:
- Compare results across different models
- Track hardware capabilities
- Correlate settings with metric performance

---

## Interpreting Results

### Good Signs ✅
- Faithfulness: 0.7-0.9 (grounded answers)
- Answer Relevance: 0.6-0.8 (focused answers)
- Context Precision: 0.5-0.7 (good selectivity)
- Recall@k: 0.5+ (comprehensive retrieval)
- Hit Rate: 1.0 (always found something)

### Areas to Improve ⚠️
- Faithfulness < 0.5 (hallucinations likely)
- Answer Relevance < 0.4 (off-topic answers)
- Context Precision < 0.3 (poor context quality)
- Recall@k < 0.3 (missing relevant docs)

### Debugging Steps 🔧

**If Faithfulness is low:**
1. Check if retrieved sources are relevant
2. Verify embedding model quality
3. Increase TOP_K to get more sources

**If Answer Relevance is low:**
1. Check query rewriting strategy
2. Verify LLM is following instructions
3. Try different query rewriting method

**If Context Precision is low:**
1. Enable reranking to filter irrelevant docs
2. Adjust chunk size to be more focused
3. Improve embedding model

---

## API Integration

When querying the API, you'll receive:

```json
{
  "answer": "...",
  "sources": [...],
  "evaluation": {...},
  "standard_metrics": {
    "recall_at_k": 0.667,
    "mrr": 1.0,
    "ndcg_at_k": 0.923,
    "hit_rate": 1.0,
    "exact_match": null,
    "f1_score": null
  },
  "rag_metrics": {
    "faithfulness": 0.85,
    "answer_relevance": 0.72,
    "context_precision": 0.6
  },
  "system_config": {
    "hardware_info": "8x CPU, 32.0GB RAM (Windows)",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "llama3.2:3b",
    "chunk_size": 600,
    "chunk_overlap": 120,
    "top_k": 5,
    "embedding_dimensions": 384,
    "temperature": 0.3,
    "timestamp": "2024-12-19T10:30:45.123456Z"
  }
}
```

---

## Testing & Validation

Run the test script to verify metrics computation:

```bash
python test_metrics.py
```

This will test:
- Standard retrieval metrics calculation
- Answer quality metrics
- RAG-specific quality metrics
- Comprehensive metrics computation

Expected output shows all metrics computed correctly with explanation of values.

---

## Summary

Your RAG system now provides:

✅ **3 comprehensive metric categories** computed for every query
✅ **9 individual metrics** tracking different quality dimensions
✅ **System configuration capture** for reproducibility
✅ **Visual UI display** with expandable sections and tooltips
✅ **Automatic evaluation** with no manual intervention needed

Monitor these metrics to continuously improve your RAG system's performance and ensure high-quality, grounded, relevant answers.
