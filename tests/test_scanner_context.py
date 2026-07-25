from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import format_scanner_context


def test_format_scanner_context_includes_all_20_rows():
    rows = []
    for idx in range(1, 21):
        rows.append(
            {
                "rank": idx,
                "symbol": f"COIN{idx}USDT",
                "score": 80.0 - idx,
                "last_price": 1.0 + idx,
                "quote_volume": 1000000 + idx,
                "volatility_pct": 2.0 + (idx * 0.1),
                "recent_15m_range_pct": 0.2 + (idx * 0.01),
                "spread_pct": 0.02,
                "reason": f"reason {idx}",
            }
        )

    context = format_scanner_context(rows)

    for idx in range(1, 21):
        assert f"COIN{idx}USDT" in context
