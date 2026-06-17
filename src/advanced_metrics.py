"""Comprehensive evaluation metrics for RAG systems.

Includes:
- Standard retrieval metrics: Recall@k, MRR, nDCG, Hit Rate
- Answer quality metrics: Exact Match, F1
- RAG-specific metrics: Faithfulness, Answer Relevance, Context Precision
"""

import logging
import re
from typing import Any, Dict, List, Optional
from collections import Counter

logger = logging.getLogger("rag")


# ============================================================================
# STANDARD RETRIEVAL METRICS
# ============================================================================

def compute_recall_at_k(
    retrieved: List[Dict[str, Any]],
    relevant_ids: List[str],
    k: int = 5,
    id_field: str = "source_file",
) -> float:
    """
    Compute Recall@k: proportion of relevant docs in top-k retrieved docs.
    
    Args:
        retrieved: List of retrieved documents
        relevant_ids: List of document IDs that are relevant to the query
        k: Cutoff for top-k
        id_field: Metadata field name for document ID
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k_retrieved = retrieved[:k]
    retrieved_ids = {
        str(item.get("metadata", {}).get(id_field, "")).strip().lower()
        for item in top_k_retrieved if item.get("metadata")
    }
    
    if not retrieved_ids:
        return 0.0
    
    hits = sum(1 for ref_id in relevant_ids 
               if str(ref_id).strip().lower() in retrieved_ids)
    
    return float(hits) / len(relevant_ids)


def compute_mrr(
    retrieved: List[Dict[str, Any]],
    relevant_ids: List[str],
    id_field: str = "source_file",
) -> float:
    """
    Compute Mean Reciprocal Rank: reciprocal of the rank of the first relevant doc.
    
    Args:
        retrieved: List of retrieved documents
        relevant_ids: List of document IDs that are relevant to the query
        id_field: Metadata field name for document ID
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    relevant_ids_set = {str(rid).strip().lower() for rid in relevant_ids}
    
    for rank, item in enumerate(retrieved, 1):
        doc_id = str(item.get("metadata", {}).get(id_field, "")).strip().lower()
        if doc_id in relevant_ids_set:
            return 1.0 / rank
    
    return 0.0


def compute_ndcg(
    retrieved: List[Dict[str, Any]],
    relevant_ids: List[str],
    k: int = 5,
    id_field: str = "source_file",
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG@k).
    
    Relevance is binary: 1 if document ID is in relevant_ids, 0 otherwise.
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    nDCG = DCG / IDCG (where IDCG is the ideal DCG with all relevant docs at top)
    
    Args:
        retrieved: List of retrieved documents
        relevant_ids: List of document IDs that are relevant to the query
        k: Cutoff for top-k
        id_field: Metadata field name for document ID
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k_retrieved = retrieved[:k]
    relevant_ids_set = {str(rid).strip().lower() for rid in relevant_ids}
    
    # Compute DCG using log2 discount
    import math

    dcg = 0.0
    for i, item in enumerate(top_k_retrieved, 1):
        doc_id = str(item.get("metadata", {}).get(id_field, "")).strip().lower()
        relevance = 1 if doc_id in relevant_ids_set else 0
        dcg += relevance / math.log2(i + 1)

    # Compute IDCG (ideal DCG: all relevant docs at top)
    num_relevant = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def compute_hit_rate(
    retrieved: List[Dict[str, Any]],
    relevant_ids: List[str],
    k: int = 5,
    id_field: str = "source_file",
) -> float:
    """
    Compute Hit Rate@k: whether at least one relevant doc is in top-k.
    
    Returns 1.0 if any relevant doc is found, 0.0 otherwise.
    
    Args:
        retrieved: List of retrieved documents
        relevant_ids: List of document IDs that are relevant to the query
        k: Cutoff for top-k
        id_field: Metadata field name for document ID
        
    Returns:
        Hit Rate score (0.0 or 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    top_k_retrieved = retrieved[:k]
    retrieved_ids = {
        str(item.get("metadata", {}).get(id_field, "")).strip().lower()
        for item in top_k_retrieved if item.get("metadata")
    }
    
    for ref_id in relevant_ids:
        if str(ref_id).strip().lower() in retrieved_ids:
            return 1.0
    
    return 0.0


# ============================================================================
# ANSWER QUALITY METRICS
# ============================================================================

def normalize_text_for_matching(text: str) -> str:
    """Normalize text for exact match and F1 comparison."""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


def compute_exact_match(answer: str, reference: str) -> float:
    """
    Compute Exact Match: whether normalized answer matches normalized reference.
    
    Args:
        answer: The generated answer
        reference: The reference/ground truth answer
        
    Returns:
        Exact Match score (0.0 or 1.0)
    """
    if not answer or not reference:
        return 0.0
    
    norm_answer = normalize_text_for_matching(answer)
    norm_reference = normalize_text_for_matching(reference)
    
    return 1.0 if norm_answer == norm_reference else 0.0


def compute_f1_score(answer: str, reference: str) -> float:
    """
    Compute token-level F1 score between answer and reference.
    
    Args:
        answer: The generated answer
        reference: The reference/ground truth answer
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not answer or not reference:
        return 0.0
    
    answer_tokens = set(normalize_text_for_matching(answer).split())
    reference_tokens = set(normalize_text_for_matching(reference).split())
    
    if not answer_tokens or not reference_tokens:
        return 0.0
    
    common_tokens = answer_tokens.intersection(reference_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(answer_tokens)
    recall = len(common_tokens) / len(reference_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# ============================================================================
# RAG-SPECIFIC QUALITY METRICS
# ============================================================================

def compute_faithfulness(
    answer: str,
    sources: List[Dict[str, Any]],
) -> float:
    """
    Compute Faithfulness: degree to which answer is supported by the context.
    
    Based on token overlap between answer and source documents.
    Higher overlap = higher faithfulness.
    
    Args:
        answer: The generated answer
        sources: List of source documents with 'text' or 'snippet' field
        
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not answer or not sources:
        return 0.0
    
    # Combine all source text
    source_text = " ".join(
        src.get("text", "") or src.get("snippet", "")
        for src in sources
    )
    
    if not source_text.strip():
        return 0.0
    
    # Tokenize
    answer_tokens = set(normalize_text_for_matching(answer).split())
    source_tokens = set(normalize_text_for_matching(source_text).split())
    
    if not answer_tokens:
        return 0.0
    
    # Compute overlap ratio
    common = answer_tokens.intersection(source_tokens)
    faithfulness = len(common) / len(answer_tokens)
    
    return min(1.0, faithfulness)


def compute_answer_relevance(
    question: str,
    answer: str,
) -> float:
    """
    Compute Answer Relevance: degree to which answer addresses the question.
    
    Based on token overlap between question and answer.
    Assumes questions should be reflected in answers.
    
    Args:
        question: The original question
        answer: The generated answer
        
    Returns:
        Answer Relevance score (0.0 to 1.0)
    """
    if not question or not answer:
        return 0.0
    
    question_tokens = set(normalize_text_for_matching(question).split())
    answer_tokens = set(normalize_text_for_matching(answer).split())
    
    if not question_tokens:
        return 0.0
    
    # Count key terms from question that appear in answer
    # Remove very common words to avoid noise
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but', 'not', 'no', 'yes', 'what', 'how', 'why', 'where', 'when', 'who'}
    
    question_key_terms = question_tokens - common_words
    answer_key_terms = answer_tokens - common_words
    
    if not question_key_terms:
        return 0.5  # Neutral if question is all common words
    
    common = question_key_terms.intersection(answer_key_terms)
    relevance = len(common) / len(question_key_terms)
    
    return min(1.0, relevance)


def compute_context_precision(
    answer: str,
    sources: List[Dict[str, Any]],
    k: Optional[int] = None,
) -> float:
    """
    Compute Context Precision: proportion of retrieved contexts that support the answer.
    
    Measures if the retrieved context is focused and relevant to answering the question.
    
    Args:
        answer: The generated answer
        sources: List of source documents with 'text' or 'snippet' field
        k: Optional cutoff (if None, uses all sources)
        
    Returns:
        Context Precision score (0.0 to 1.0)
    """
    if not sources:
        return 0.0
    
    if k is not None:
        sources = sources[:k]
    
    if not answer:
        return 0.0
    
    answer_tokens = set(normalize_text_for_matching(answer).split())
    
    if not answer_tokens:
        return 0.0
    
    # Count sources that have significant overlap with answer
    supporting_sources = 0
    for src in sources:
        source_text = src.get("text", "") or src.get("snippet", "")
        source_tokens = set(normalize_text_for_matching(source_text).split())
        
        # Threshold: source must share at least 20% of answer tokens
        if source_tokens:
            common = answer_tokens.intersection(source_tokens)
            overlap_ratio = len(common) / len(answer_tokens)
            if overlap_ratio >= 0.2:
                supporting_sources += 1
    
    context_precision = supporting_sources / len(sources)
    return min(1.0, context_precision)


# ============================================================================
# COMPREHENSIVE METRICS COMPUTATION
# ============================================================================

def compute_comprehensive_metrics(
    question: str,
    answer: str,
    retrieved: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    reference: Optional[str] = None,
    relevant_document_ids: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a RAG query-answer pair.
    
    Args:
        question: The original query question
        answer: The generated answer
        retrieved: Full list of retrieved documents (for ranking metrics)
        sources: List of sources used in the final answer
        reference: Optional ground truth answer for comparison
        relevant_document_ids: Optional list of truly relevant doc IDs
        top_k: Parameter for @k metrics
        
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # === Retrieval Metrics ===
    # Determine the set of relevant document IDs to evaluate retrieval.
    if relevant_document_ids:
        eval_ids = [str(r).strip() for r in relevant_document_ids if r]
    else:
        # Try to extract identifiers from the `sources` returned by the pipeline.
        eval_ids = []
        for src in sources:
            # Common places where a source id may live
            mid = src.get("metadata") or {}
            candidates = [
                mid.get("source_file"),
                mid.get("file"),
                src.get("file"),
                src.get("source_file"),
                src.get("id"),
            ]
            for c in candidates:
                if c:
                    eval_ids.append(str(c).strip())
        # Deduplicate and normalize
        eval_ids = list({i.lower(): i for i in eval_ids}.values())

    if eval_ids:
        metrics["recall_at_k"] = round(
            compute_recall_at_k(retrieved, eval_ids, k=top_k),
            4,
        )
        metrics["mrr"] = round(
            compute_mrr(retrieved, eval_ids),
            4,
        )
        metrics["ndcg_at_k"] = round(
            compute_ndcg(retrieved, eval_ids, k=top_k),
            4,
        )
        metrics["hit_rate"] = round(
            compute_hit_rate(retrieved, eval_ids, k=top_k),
            4,
        )
    else:
        # No identifiable relevant IDs available; leave retrieval metrics as None
        metrics["recall_at_k"] = None
        metrics["mrr"] = None
        metrics["ndcg_at_k"] = None
        metrics["hit_rate"] = None
    
    # === Answer Quality Metrics (if reference provided) ===
    if reference:
        metrics["exact_match"] = round(compute_exact_match(answer, reference), 4)
        metrics["f1_score"] = round(compute_f1_score(answer, reference), 4)
    else:
        metrics["exact_match"] = None
        metrics["f1_score"] = None
    
    # === RAG-Specific Metrics ===
    metrics["faithfulness"] = round(compute_faithfulness(answer, sources), 4)
    metrics["answer_relevance"] = round(compute_answer_relevance(question, answer), 4)
    metrics["context_precision"] = round(
        compute_context_precision(answer, sources, k=top_k),
        4
    )
    
    return metrics
