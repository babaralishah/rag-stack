"""
Test script for comprehensive metrics calculation.
Run this to verify all metrics work correctly.
"""

from src.advanced_metrics import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    compute_hit_rate,
    compute_exact_match,
    compute_f1_score,
    compute_faithfulness,
    compute_answer_relevance,
    compute_context_precision,
    compute_comprehensive_metrics,
)


def test_standard_metrics():
    """Test standard retrieval metrics."""
    print("\n" + "="*60)
    print("TESTING STANDARD RETRIEVAL METRICS")
    print("="*60)
    
    # Sample retrieved documents
    retrieved = [
        {"metadata": {"source_file": "doc1.pdf"}, "score": 0.9},
        {"metadata": {"source_file": "doc2.pdf"}, "score": 0.8},
        {"metadata": {"source_file": "doc3.pdf"}, "score": 0.7},
        {"metadata": {"source_file": "doc4.pdf"}, "score": 0.6},
        {"metadata": {"source_file": "doc5.pdf"}, "score": 0.5},
    ]
    
    # Sample relevant documents
    relevant_ids = ["doc1.pdf", "doc2.pdf", "doc6.pdf"]
    
    recall = compute_recall_at_k(retrieved, relevant_ids, k=3)
    mrr = compute_mrr(retrieved, relevant_ids)
    ndcg = compute_ndcg(retrieved, relevant_ids, k=3)
    hit_rate = compute_hit_rate(retrieved, relevant_ids, k=3)
    
    print(f"Recall@3: {recall:.3f} (expected ~0.667: 2 out of 3 relevant docs found)")
    print(f"MRR: {mrr:.3f} (expected 1.0: first relevant at position 1)")
    print(f"nDCG@3: {ndcg:.3f} (expected high: relevant docs at top)")
    print(f"Hit Rate: {hit_rate:.3f} (expected 1.0: at least one relevant found)")


def test_answer_quality_metrics():
    """Test answer quality metrics."""
    print("\n" + "="*60)
    print("TESTING ANSWER QUALITY METRICS")
    print("="*60)
    
    answer = "The capital of France is Paris, a major European city."
    reference = "Paris is the capital of France."
    
    exact_match = compute_exact_match(answer, reference)
    f1 = compute_f1_score(answer, reference)
    
    print(f"Exact Match: {exact_match:.3f} (0 = different, 1 = identical after normalization)")
    print(f"F1 Score: {f1:.3f} (0-1 scale of token overlap)")


def test_rag_metrics():
    """Test RAG-specific quality metrics."""
    print("\n" + "="*60)
    print("TESTING RAG-SPECIFIC QUALITY METRICS")
    print("="*60)
    
    question = "What is machine learning?"
    answer = "Machine learning is a subset of artificial intelligence."
    
    sources = [
        {
            "text": "Machine learning and deep learning are subsets of artificial intelligence.",
            "snippet": "Machine learning is AI"
        },
        {
            "text": "Neural networks are used in machine learning models.",
            "snippet": "Neural networks in ML"
        },
        {
            "text": "The Mona Lisa was painted by Leonardo da Vinci.",
            "snippet": "Mona Lisa painting"
        }
    ]
    
    faithfulness = compute_faithfulness(answer, sources)
    relevance = compute_answer_relevance(question, answer)
    context_precision = compute_context_precision(answer, sources)
    
    print(f"Faithfulness: {faithfulness:.3f} (0-1, higher = answer supported by sources)")
    print(f"Answer Relevance: {relevance:.3f} (0-1, higher = answer addresses question)")
    print(f"Context Precision: {context_precision:.3f} (0-1, higher = more sources support answer)")


def test_comprehensive_metrics():
    """Test comprehensive metrics computation."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE METRICS")
    print("="*60)
    
    question = "What is artificial intelligence?"
    answer = "Artificial intelligence is the simulation of human intelligence by machines."
    
    retrieved = [
        {"metadata": {"source_file": "ai_book.pdf"}, "text": "AI is intelligence demonstrated by machines"},
        {"metadata": {"source_file": "ml_guide.pdf"}, "text": "Machine learning is a type of AI"},
        {"metadata": {"source_file": "history.pdf"}, "text": "AI was founded as an academic discipline in 1956"},
    ]
    
    sources = [
        {"text": "AI is intelligence demonstrated by machines", "snippet": "AI is intelligence"},
        {"text": "Machine learning is a subset of AI", "snippet": "ML subset"},
    ]
    
    metrics = compute_comprehensive_metrics(
        question=question,
        answer=answer,
        retrieved=retrieved,
        sources=sources,
        top_k=3,
    )
    
    print("\nComprehensive Metrics Results:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n" + "🧪 RAG METRICS VALIDATION TEST SUITE" + "\n")
    
    test_standard_metrics()
    test_answer_quality_metrics()
    test_rag_metrics()
    test_comprehensive_metrics()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")
