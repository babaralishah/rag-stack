from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import agent


def test_spike_ratio_calculation():
    original = agent.fetch_klines_resilient

    def fake_fetch(symbol: str, interval: str, limit: int = 7):
        assert interval == "1d"
        data = []
        for i in range(7):
            data.append(
                {
                    "open_time": i,
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 10.0,
                    "quote_volume": 100.0,
                    "close_time": i,
                }
            )
        return data, {"is_stale": False, "stale_age_seconds": None, "stale_badge": None}

    agent.fetch_klines_resilient = fake_fetch
    try:
        ratio, baseline, freshness = agent._compute_volume_spike_ratio("ABCUSDT", 300.0)
        assert baseline == 100.0
        assert ratio == 3.0
        assert freshness.get("is_stale") is False
    finally:
        agent.fetch_klines_resilient = original


def test_llm_fallback_no_provider_configured():
    original = agent._llm_provider_chain
    agent._llm_provider_chain = lambda: []
    try:
        try:
            agent.fetch_llm_with_fallback("sys", "ctx")
            assert False, "Expected RuntimeError when no providers are configured"
        except RuntimeError as exc:
            assert "no provider configured" in str(exc).lower()
    finally:
        agent._llm_provider_chain = original


def test_llm_fallback_precedence_groq_then_gemini():
    original_chain = agent._llm_provider_chain
    original_circuit = agent._provider_circuit_open

    calls = []

    def groq_provider(system_prompt: str, user_content: str) -> str:
        calls.append("groq")
        raise RuntimeError("groq unavailable")

    def gemini_provider(system_prompt: str, user_content: str) -> str:
        calls.append("gemini")
        return "ok from gemini"

    agent._llm_provider_chain = lambda: [
        ("Groq", "groq", groq_provider, (RuntimeError,)),
        ("Gemini", "gemini", gemini_provider, (RuntimeError,)),
    ]
    agent._provider_circuit_open = lambda provider: False

    try:
        out = agent.fetch_llm_with_fallback("sys", "ctx")
        assert out == "ok from gemini"
        assert calls[0] == "groq"
        assert "gemini" in calls
    finally:
        agent._llm_provider_chain = original_chain
        agent._provider_circuit_open = original_circuit


def test_meme_scope_filters_and_returns_spike_or_category_items():
    original_get_tickers = agent.get_all_usdt_tickers
    original_categories = agent._coingecko_categories_for_symbol
    original_pegged = agent._is_pegged_or_non_scalp_asset
    original_movement = agent._compute_15m_movement
    original_book = agent._get_symbol_book_ticker
    original_depth = agent._get_symbol_depth_snapshot
    original_spike = agent._compute_volume_spike_ratio
    original_direction = agent._recent_direction_label
    original_fetch_klines = agent.fetch_klines_resilient

    agent.get_all_usdt_tickers = lambda: [
        {
            "symbol": "DOGEUSDT",
            "base_asset": "DOGE",
            "last_price": 0.2,
            "price_change_percent": 3.0,
            "quote_volume": 10_000_000.0,
            "volume": 1.0,
            "high_price": 0.22,
            "low_price": 0.18,
            "spread_pct": None,
            "spread_source": "pending",
        },
        {
            "symbol": "SOLUSDT",
            "base_asset": "SOL",
            "last_price": 70.0,
            "price_change_percent": 2.0,
            "quote_volume": 12_000_000.0,
            "volume": 1.0,
            "high_price": 75.0,
            "low_price": 65.0,
            "spread_pct": None,
            "spread_source": "pending",
        },
        {
            "symbol": "ATOMUSDT",
            "base_asset": "ATOM",
            "last_price": 9.0,
            "price_change_percent": 0.5,
            "quote_volume": 8_000_000.0,
            "volume": 1.0,
            "high_price": 9.1,
            "low_price": 8.9,
            "spread_pct": None,
            "spread_source": "pending",
        },
    ]
    agent._coingecko_categories_for_symbol = lambda base: ["Meme"] if base == "DOGE" else ["Layer 1"]
    agent._is_pegged_or_non_scalp_asset = lambda base, quote, categories: (False, "")
    agent._compute_15m_movement = lambda klines, freshness=None: {
        "latest_range_pct": 1.0,
        "avg_range_pct": 0.8,
        "is_stale": False,
        "stale_age_seconds": None,
        "stale_badge": None,
    }
    agent._get_symbol_book_ticker = lambda symbol: {"bidPrice": "1", "askPrice": "1.001"}
    agent._get_symbol_depth_snapshot = lambda symbol, limit=50: {
        "mid_price": 1.0,
        "best_bid": 0.999,
        "best_ask": 1.001,
        "bid_depth_notional_0_5pct": 100_000.0,
        "ask_depth_notional_0_5pct": 100_000.0,
        "depth_notional_0_5pct": 200_000.0,
        "depth_imbalance": 0.0,
        "depth_source": "test",
    }
    agent._compute_volume_spike_ratio = lambda symbol, today: (2.3 if symbol == "SOLUSDT" else 1.2, 100.0, {"is_stale": False, "stale_badge": None})
    agent._recent_direction_label = lambda symbol: ("still pumping", {"is_stale": False, "stale_badge": None})
    agent.fetch_klines_resilient = lambda symbol, interval, limit=24: ([], {"is_stale": False, "stale_badge": None})

    try:
        rows, stats = agent.find_top_scalping_candidates(top_n=10, scope=agent.SCAN_SCOPE_MEME, return_stats=True)
        symbols = {row["symbol"] for row in rows}
        assert "DOGEUSDT" in symbols
        assert "SOLUSDT" in symbols
        assert "ATOMUSDT" not in symbols
        assert stats.get("scope") == agent.SCAN_SCOPE_MEME
    finally:
        agent.get_all_usdt_tickers = original_get_tickers
        agent._coingecko_categories_for_symbol = original_categories
        agent._is_pegged_or_non_scalp_asset = original_pegged
        agent._compute_15m_movement = original_movement
        agent._get_symbol_book_ticker = original_book
        agent._get_symbol_depth_snapshot = original_depth
        agent._compute_volume_spike_ratio = original_spike
        agent._recent_direction_label = original_direction
        agent.fetch_klines_resilient = original_fetch_klines


def test_stale_row_skip_logic_keeps_fresh_only():
    fresh_ms = 10_000
    stale_ms = 1

    original_validator = agent._validate_not_stale
    original_exchange = agent.get_binance_exchange_info
    original_tickers = agent.get_binance_all_tickers_raw

    def fake_validator(close_time_ms: int, max_age_seconds: int, context: str):
        if close_time_ms == stale_ms:
            raise RuntimeError("stale")

    agent._validate_not_stale = fake_validator
    agent.get_binance_exchange_info = lambda: {
        "symbols": [
            {"symbol": "OLDUSDT", "status": "TRADING", "quoteAsset": "USDT"},
            {"symbol": "NEWUSDT", "status": "TRADING", "quoteAsset": "USDT"},
        ]
    }
    agent.get_binance_all_tickers_raw = lambda: [
        {
            "symbol": "OLDUSDT",
            "lastPrice": "1.0",
            "priceChangePercent": "0.0",
            "quoteVolume": "1000000",
            "volume": "1000000",
            "highPrice": "1.1",
            "lowPrice": "0.9",
            "closeTime": str(stale_ms),
        },
        {
            "symbol": "NEWUSDT",
            "lastPrice": "1.0",
            "priceChangePercent": "0.0",
            "quoteVolume": "1000000",
            "volume": "1000000",
            "highPrice": "1.1",
            "lowPrice": "0.9",
            "closeTime": str(fresh_ms),
        },
    ]
    try:
        out = agent.get_all_usdt_tickers()
        assert len(out) == 1
        assert out[0]["symbol"] == "NEWUSDT"
    finally:
        agent._validate_not_stale = original_validator
        agent.get_binance_exchange_info = original_exchange
        agent.get_binance_all_tickers_raw = original_tickers
