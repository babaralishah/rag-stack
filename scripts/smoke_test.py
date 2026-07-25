import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import run_deep_dive_pipeline, run_scanner_pipeline, SCAN_SCOPE_ALL, SCAN_SCOPE_MEME

SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "PEPE", "DOGE", "ADA", "AVAX", "LINK", "UNI", "MATIC", "SAND"]

TECH_FIELDS = ["sma20", "ema9", "ema21", "rsi14", "support", "resistance"]


def summarize_deep_dive(symbol: str) -> dict[str, Any]:
    t0 = time.time()
    result = run_deep_dive_pipeline(symbol)
    duration = round(time.time() - t0, 2)
    research = result.get("research", {}) if isinstance(result, dict) else {}
    technicals = research.get("technicals", {}) if isinstance(research, dict) else {}
    technical_complete = True
    technical_missing: dict[str, list[str]] = {}
    for interval, snapshot in technicals.items():
        if not isinstance(snapshot, dict):
            technical_missing[interval] = TECH_FIELDS
            technical_complete = False
            continue
        missing = [field for field in TECH_FIELDS if snapshot.get(field) is None]
        if missing:
            technical_complete = False
            technical_missing[interval] = missing
    fundamentals = research.get("fundamentals", {}) if isinstance(research, dict) else {}
    fundamentals_degraded = bool(fundamentals.get("fundamentals_degraded", False))
    fundamentals_missing = [k for k, v in fundamentals.items() if v is None and k not in {"fundamentals_degraded", "fundamentals_sources", "fundamentals_cache_hit", "resolved_input", "binance_symbol", "coingecko_id", "name", "symbol"}]
    if not fundamentals_missing:
        fundamentals_missing = []
    news = research.get("news", []) if isinstance(research, dict) else []
    news_count = len(news)
    llm_report = bool(result.get("summary")) and result.get("error") is None
    return {
        "symbol": symbol,
        "technical_complete": technical_complete,
        "technical_missing": technical_missing,
        "fundamentals_degraded": fundamentals_degraded,
        "fundamentals_missing": fundamentals_missing,
        "news_count": news_count,
        "news_status": research.get("news_status", "unavailable"),
        "llm_report": llm_report,
        "latency_seconds": duration,
        "error": result.get("error"),
    }


def summarize_scanner(scope: str) -> dict[str, Any]:
    t0 = time.time()
    result = run_scanner_pipeline(top_n=15, scope=scope)
    duration = round(time.time() - t0, 2)
    stats = result.get("scan_stats", {}) or {}
    counts = stats.get("counts", {}) or {}
    candidate_count = len(result.get("candidates", []) or [])
    llm_summary = bool(result.get("summary")) and result.get("error") is None
    return {
        "scope": scope,
        "candidate_count": candidate_count,
        "counts": counts,
        "llm_summary": llm_summary,
        "latency_seconds": duration,
        "error": result.get("error"),
    }


def main() -> None:
    print("=== DEEP-DIVE SMOKE TEST ===")
    deep_results = [summarize_deep_dive(symbol) for symbol in SYMBOLS]
    print(json.dumps(deep_results, indent=2))
    print("\n=== SCANNER SMOKE TEST ===")
    scanner_results = [summarize_scanner(SCAN_SCOPE_ALL), summarize_scanner(SCAN_SCOPE_MEME)]
    print(json.dumps(scanner_results, indent=2))
    print("\n=== SUMMARY TABLE ===")
    headers = ["symbol", "tech_pass", "fundamentals_degraded", "news_count", "llm_report", "latency_s"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in deep_results:
        print(
            "| "
            + " | ".join(
                [
                    row["symbol"],
                    "PASS" if row["technical_complete"] else "FAIL",
                    "YES" if row["fundamentals_degraded"] else "NO",
                    str(row["news_count"]),
                    "YES" if row["llm_report"] else "NO",
                    str(row["latency_seconds"]),
                ]
            )
            + " |"
        )
    print("\n=== SCANNER SUMMARY TABLE ===")
    headers = ["scope", "candidate_count", "entered_scoring_pool", "returned_count", "llm_summary", "latency_s"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in scanner_results:
        counts = row.get("counts", {})
        print(
            "| "
            + " | ".join(
                [
                    row["scope"],
                    str(row["candidate_count"]),
                    str(counts.get("entered_scoring_pool", "N/A")),
                    str(counts.get("returned_count", "N/A")),
                    "YES" if row["llm_summary"] else "NO",
                    str(row["latency_seconds"]),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
