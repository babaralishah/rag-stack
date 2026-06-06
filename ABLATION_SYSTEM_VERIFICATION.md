# Ablation Study System - Complete Wiring Verification ✅

## Executive Summary
The ablation study system is **fully wired and operational**. All feature flags flow correctly from the Streamlit UI through the FastAPI backend to the RAG pipeline with proper cache isolation per phase.

---

## 1. UI Layer (ui.py)

### Phase Selector Creation
✅ **Line 278-285**: Phase selectbox created in sidebar
```python
phase = st.selectbox(
    "Select experimental phase",
    options=["V1", "V2", "V3", "V4", "V5", "V6"],
    index=5,
    help="V1: Basic RAG | V2: +Query Rewriter | V3: +History | V4: +Hybrid | V5: +Reranker | V6: +Guardrails",
)
```

### Query Payload
✅ **Line 345**: Phase included in every query request
```python
payload = {
    "question": question,
    "top_k": top_k,
    "use_reranker": use_reranker,
    "use_hybrid": use_hybrid,
    "phase": phase,  # ← SENT WITH EVERY QUERY
    "history": [...]
}
```

---

## 2. Backend Request Model (api.py)

### QueryRequest Definition
✅ **Line 175-180**: Phase field properly defined
```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    use_reranker: bool = True
    use_hybrid: bool = True
    phase: Optional[str] = None        # ← ACCEPTS PHASE FROM UI
    history: Optional[List[ChatMessage]] = None
```

---

## 3. Phase Resolution Logic (api.py)

### resolve_phase_flags()
✅ **Line 184-194**: Converts phase string to feature flags
```python
def resolve_phase_flags(phase: Optional[str]) -> Dict[str, bool]:
    if not phase:
        return {}
    
    phase_key = phase.strip().upper()  # "v6" → "V6"
    if phase_key not in PHASE_FLAGS:
        logger.warning("Unknown phase '%s', falling back to runtime toggles.", phase_key)
        return {}
    
    return PHASE_FLAGS[phase_key].copy()  # ← RETURNS DICT OF FLAGS
```

### get_query_settings()
✅ **Line 196-209**: Routes to correct phase or fallback
```python
def get_query_settings(req: QueryRequest) -> Dict[str, bool]:
    phase_settings = resolve_phase_flags(req.phase)
    if phase_settings:
        return phase_settings  # ← USE PHASE FLAGS IF PROVIDED
    
    # Fallback to individual toggles if no phase specified
    return {
        "use_query_rewriter": True,
        "use_hybrid": req.use_hybrid,
        "use_reranker": req.use_reranker,
        "use_chat_history": True,
        "use_guardrails": False,
    }
```

---

## 4. Phase Definitions (config.py)

### PHASE_FLAGS Dictionary
✅ **config.py lines 27-57**: All 6 phases defined
```python
PHASE_FLAGS = {
    "V1": {  # Baseline: No features
        "use_query_rewriter": False,
        "use_hybrid": False,
        "use_chat_history": False,
        "use_reranker": False,
        "use_guardrails": False,
    },
    "V2": {  # +Query Rewriter
        "use_query_rewriter": True,
        "use_hybrid": False,
        "use_chat_history": False,
        "use_reranker": False,
        "use_guardrails": False,
    },
    "V3": {  # +Chat History
        "use_query_rewriter": True,
        "use_hybrid": False,
        "use_chat_history": True,
        "use_reranker": False,
        "use_guardrails": False,
    },
    "V4": {  # +Hybrid Search
        "use_query_rewriter": True,
        "use_hybrid": True,
        "use_chat_history": True,
        "use_reranker": False,
        "use_guardrails": False,
    },
    "V5": {  # +Reranker
        "use_query_rewriter": True,
        "use_hybrid": True,
        "use_chat_history": True,
        "use_reranker": True,
        "use_guardrails": False,
    },
    "V6": {  # +Guardrails
        "use_query_rewriter": True,
        "use_hybrid": True,
        "use_chat_history": True,
        "use_reranker": True,
        "use_guardrails": True,
    },
}
```

---

## 5. Query Execution Pipeline (/query endpoint, api.py)

### Step 1: Extract Phase Flags
✅ **Line 607**: Get flags from selected phase
```python
flags = get_query_settings(req)
# flags = {"use_query_rewriter": True, "use_hybrid": True, ...}
```

### Step 2: Chat History Based on Flags
✅ **Line 608**: Include history only if phase enables it
```python
history = normalize_chat_history(req.history) if flags["use_chat_history"] else []
```

### Step 3: Query Rewriting Based on Flags
✅ **Line 609-611**: Use query rewriter only if phase enables it
```python
query_text, query_vector = rewrite_and_embed_query(
    question, use_query_rewriter=flags["use_query_rewriter"]  # ← FLAG PASSED
)
```
- If `use_query_rewriter=False` → question used as-is
- If `use_query_rewriter=True` → question rewritten via Gemini

### Step 4: Document Search with Flags
✅ **Line 612-618**: Pass hybrid & reranker flags to search
```python
retrieved = search_documents(
    store,
    query_text,
    query_vector,
    top_k=req.top_k,
    use_hybrid=flags["use_hybrid"],          # ← Semantic + BM25
    use_reranker=flags["use_reranker"],      # ← Sets retrieve_k
)
```
- If `use_reranker=True` → retrieve 12 chunks, keep top 6 after reranking
- If `use_reranker=False` → retrieve exactly `top_k` chunks

### Step 5: Answer Generation with Flags
✅ **Line 629-636**: Pass reranker & guardrail flags to LLM pipeline
```python
result = generate_answer_payload(question, retrieved, req, flags, history=history)
```

Where in `generate_answer_payload`:
```python
def generate_answer_payload(question: str, retrieved: list, req: QueryRequest, flags: Dict[str, bool], ...):
    out = rag_answer(
        question=question,
        retrieved=retrieved,
        min_score=MIN_SCORE,
        use_reranker=flags["use_reranker"],         # ← RERANKING FLAG
        final_top_k=req.top_k,
        use_guardrails=flags["use_guardrails"],     # ← GUARDRAIL FLAG
        history=history,
    )
    # ...
```

---

## 6. RAG Pipeline Logic (src/rag_pipeline.py)

### Reranking Based on Flag
✅ **Lines 75-98**: Reranking logic respects flag
```python
if use_reranker and retrieved:
    # Cross-encoder reranks the candidates
    reranker = get_reranker()
    retrieved = reranker.rerank(...)
else:
    logger.info("⛔ Re-ranking DISABLED by user")
    for r in retrieved:
        r["final_score"] = r.get("score", 0.0)
```

### Guardrails Based on Flag
✅ **Lines 107-121**: Guardrail abstention respects flag
```python
if use_guardrails and retrieved:
    max_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
    if max_score < min_score:  # min_score = 0.35
        logger.warning("Guardrail triggered: max retrieved score %.3f...", max_score, min_score)
        return {
            "answer": "I don't have enough high-confidence information from the documents...",
            "sources": [],
        }
```

---

## 7. Cache Isolation by Phase (src/cache.py)

### Cache Key Includes Phase
✅ **Line 47**: Function signature accepts phase
```python
def get_cache_key(question: str, use_hybrid=True, use_reranker=True, top_k=6, phase: str | None = None):
```

### Phase Baked into Hash
✅ **Lines 60-65**: Phase included in cache key computation
```python
data = {
    "q": question.strip().lower(),
    "hybrid": bool(use_hybrid),
    "rerank": bool(use_reranker),
    "k": int(top_k),
    "phase": phase.strip().upper() if phase else None,  # ← PHASE IN KEY
}
return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
```

### Cache Key Called with Phase
✅ **api.py line 590-596**: Cache key generation includes phase
```python
cache_key = get_cache_key(
    question=question,
    use_hybrid=req.use_hybrid,
    use_reranker=req.use_reranker,
    top_k=req.top_k,
    phase=req.phase,  # ← PHASE PASSED
)
```

**Result**: Same question returns different cache keys per phase
- `cache_key_v1("What is ML?")` ≠ `cache_key_v6("What is ML?")`
- Prevents cross-phase cache contamination ✅

---

## 8. Feature Disable Verification

| Feature | V1 | V2 | V3 | V4 | V5 | V6 | Disabled When |
|---------|----|----|----|----|----|----|---|
| Query Rewriter | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | Line 609: `rewrite_and_embed_query(..., use_query_rewriter=False)` |
| Hybrid Search | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | Line 617: `search_documents(..., use_hybrid=False)` → only semantic |
| Chat History | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Line 608: `history = []` if flag is False |
| Reranker | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Line 612: `use_hybrid=False` sets retrieve_k = top_k; Line 85 skips reranking |
| Guardrails | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | Line 632: `use_guardrails=False` → no score threshold check |

---

## 9. Tested Scenarios

All verified in `tests/test_phase_flags.py`:

✅ **Test 1**: Phase keys V1-V6 present in PHASE_FLAGS
✅ **Test 2**: resolve_phase_flags() returns correct dictionaries for each phase
✅ **Test 3**: get_query_settings() properly resolves phases
✅ **Test 4**: Cache keys differ per phase (prevents cross-phase hits)
✅ **Test 5**: Guardrail abstention triggers at score 0.2 (< 0.35 threshold) when enabled
✅ **Test 6**: Normal LLM flow occurs at score 0.95 even when guardrails enabled
✅ **Test 7**: Guardrails don't trigger in V5 (guardrails disabled)

---

## 10. Complete Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STREAMLIT UI (ui.py)                                         │
│ User selects phase (V1-V6) and asks a question             │
│ Sends: {"question": "...", "phase": "V6", ...}             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ FASTAPI /query ENDPOINT (api.py:583)                       │
│ 1. Extract phase from request → "V6"                       │
│ 2. Call get_query_settings(req)                            │
│    → resolve_phase_flags("V6")                             │
│    → Returns PHASE_FLAGS["V6"]                             │
│    → {use_query_rewriter: True, use_hybrid: True, ...}    │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌──────────────┐
    │ History │  │ Query   │  │ Search &     │
    │ Filter  │  │ Rewrite │  │ Rerank       │
    │         │  │         │  │              │
    │ use_... │  │ use_... │  │ use_hybrid   │
    │ history │  │ rewriter│  │ use_reranker │
    └────┬────┘  └────┬────┘  └──────┬───────┘
         │             │              │
         ▼             ▼              ▼
   [filtered or ] [rewritten or]  [retrieve+rerank
    full history] original query   or just semantic]
         │             │              │
         └─────────────┼──────────────┘
                       ▼
         ┌──────────────────────────────┐
         │ generate_answer_payload()    │
         │ Calls rag_answer() with:     │
         │ - use_reranker               │
         │ - use_guardrails             │
         └────────┬─────────────────────┘
                  ▼
    ┌──────────────────────────────────┐
    │ rag_pipeline.rag_answer()        │
    │                                  │
    │ If use_reranker:                 │
    │   Cross-encode score candidates  │
    │ If use_guardrails:               │
    │   Check max_score >= 0.35        │
    │   If not → abstain               │
    │                                  │
    │ Generate answer + evaluate       │
    └────────┬─────────────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ Cache with phase in key     │
  │ (prevent cross-phase hits)  │
  └────────┬────────────────────┘
           ▼
  ┌─────────────────────────────┐
  │ Return answer to UI         │
  └─────────────────────────────┘
```

---

## 11. Conclusion

✅ **System Status**: FULLY OPERATIONAL

**All wiring verified:**
- UI phase selector → sends phase string
- QueryRequest accepts phase
- Phase resolution → correctly maps to PHASE_FLAGS
- Feature flags → properly extracted and passed through pipeline
- Each flag → correctly disables/enables the corresponding feature
- Cache isolation → prevents cross-phase contamination
- Tests → all passing and verified

**You can now:**
1. Start the backend: `python -m uvicorn src.api:app --reload`
2. Start the UI: `streamlit run ui.py`
3. Select phase V1-V6 in the dropdown
4. Submit queries and observe feature behavior change per phase
5. Run full ablation study with high confidence in feature isolation
