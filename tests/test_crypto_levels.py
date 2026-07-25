from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import _technical_snapshot, validate_technical_readiness


def _synthetic_ohlcv_frame() -> pd.DataFrame:
    closes = [
        100.0, 101.2, 99.8, 102.4, 100.9, 103.5, 101.5, 104.0, 102.8, 105.4,
        103.6, 106.1, 104.2, 107.0, 105.5, 108.0, 106.3, 109.1, 107.7, 110.0,
        108.2, 111.4, 109.6, 112.1, 110.7, 113.0, 111.6, 113.8, 112.4, 114.3,
    ]
    rows = []
    for i, close in enumerate(closes):
        open_price = closes[i - 1] if i > 0 else close - 0.4
        high = close + (1.0 + (0.2 if i % 3 == 0 else 0.0))
        low = close - (1.0 + (0.2 if i % 4 == 0 else 0.0))
        volume = 1000 + (i * 25)
        rows.append(
            {
                "open_time": i,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": volume * close,
            }
        )
    return pd.DataFrame(rows)


def test_support_resistance_stay_directional_on_synthetic_series():
    frame = _synthetic_ohlcv_frame()
    snap = _technical_snapshot(frame)

    close = snap["latest_close"]
    support = snap["support"]
    resistance = snap["resistance"]

    if support is not None and resistance is not None:
        assert support < close < resistance

    assert all(level["level"] < close for level in snap.get("support_levels", []))
    assert all(level["level"] > close for level in snap.get("resistance_levels", []))


def test_technical_readiness_blocks_heavy_missing_fields():
    research = {
        "input": "TEST",
        "technicals": {
            "1m": {"sma20": None, "ema9": None, "ema21": None, "rsi14": None, "support": None, "resistance": None},
            "15m": {"sma20": None, "ema9": None, "ema21": None, "rsi14": None, "support": None, "resistance": None},
        },
    }
    ok, reason, diagnostics = validate_technical_readiness(research, missing_threshold=0.4)
    assert ok is False
    assert "technical data unavailable" in (reason or "")
    assert diagnostics.get("missing_ratio", 0.0) > 0.4
