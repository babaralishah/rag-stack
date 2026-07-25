from agent import run_scanner_pipeline, run_deep_dive_pipeline, SCAN_TOP_N

scan = run_scanner_pipeline(top_n=SCAN_TOP_N)
print("=== SCANNER ===")
print(f"status_error={scan.get('error')}")
print(f"message={scan.get('message')}")
print(f"returned_rows={len(scan.get('candidates', []))}")
print(f"context_row_roundtrip_ok={scan.get('context_row_roundtrip_ok')}")
if scan.get('candidates'):
    first = scan['candidates'][0]
    print("top1_symbol=", first.get('symbol'))
    print("top1_manual_review=", first.get('needs_manual_review'))

for symbol in ("BTC", "SOL"):
    result = run_deep_dive_pipeline(symbol)
    research = result.get('research', {}) if isinstance(result, dict) else {}
    print(f"=== DEEPDIVE {symbol} ===")
    print(f"status_error={result.get('error')}")
    print(f"news_status={research.get('news_status')}")
    print(f"news_sources_used={research.get('news_sources_used')}")
    print(f"news_count={len(research.get('news', [])) if isinstance(research, dict) else 0}")
    if isinstance(research, dict) and research.get('news'):
        for idx, item in enumerate(research.get('news', [])[:3], start=1):
            print(f"news_{idx}: {item.get('source')} | {item.get('title')}")
