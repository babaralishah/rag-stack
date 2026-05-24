import re
from typing import Any, Dict, List, Optional


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9']+", normalized)


def token_overlap_ratio(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize(prediction)
    reference_tokens = set(tokenize(reference))
    if not prediction_tokens or not reference_tokens:
        return 0.0

    match_count = sum(1 for token in prediction_tokens if token in reference_tokens)
    return float(match_count) / len(prediction_tokens)


def compute_support_overlap(answer: str, sources: List[Dict[str, Any]]) -> float:
    if not answer or not sources:
        return 0.0

    support_text = " ".join(str(src.get("snippet", "")) for src in sources)
    return token_overlap_ratio(answer, support_text)


def compute_reference_scores(answer: str, reference: str) -> Dict[str, float]:
    if not answer or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    answer_tokens = tokenize(answer)
    reference_tokens = tokenize(reference)
    reference_token_set = set(reference_tokens)
    if not answer_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = sum(1 for token in answer_tokens if token in reference_token_set)
    precision = float(common) / len(answer_tokens)
    recall = float(common) / len(reference_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_ragas_metrics(
    answer: str,
    sources: List[Dict[str, Any]],
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    source_scores = [float(src.get("final_score", src.get("score", 0.0))) for src in sources]
    source_scores = [score for score in source_scores if score >= 0.0]

    if source_scores:
        max_source_score = max(source_scores)
        mean_source_score = sum(source_scores) / len(source_scores)
    else:
        max_source_score = 0.0
        mean_source_score = 0.0

    source_support = compute_support_overlap(answer, sources)
    ragas_score = 0.45 * mean_source_score + 0.45 * source_support + 0.10 * max_source_score
    ragas_score = max(0.0, min(1.0, ragas_score))

    if ragas_score >= 0.75:
        label = "high"
    elif ragas_score >= 0.45:
        label = "medium"
    else:
        label = "low"

    warnings: List[str] = []
    if mean_source_score < 0.35:
        warnings.append("low_retrieval_confidence")
    if source_support < 0.35:
        warnings.append("low_source_support")
    if label == "low" and warnings == []:
        warnings.append("review_answer_quality")

    evaluation: Dict[str, Any] = {
        "ragas_score": round(ragas_score, 4),
        "source_confidence": round(mean_source_score, 4),
        "max_source_score": round(max_source_score, 4),
        "source_support": round(source_support, 4),
        "label": label,
        "warnings": warnings,
        "source_count": len(sources),
    }

    if reference is not None:
        evaluation["reference_scores"] = compute_reference_scores(answer, reference)

    return evaluation
