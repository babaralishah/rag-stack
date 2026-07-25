# This tool produces research summaries for human review. It does not place trades and is not financial advice.

"""Crypto market research CLI with scanner and deep-dive modes.

The script only produces decision-support research for a human reviewer. It
never executes trades.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import calendar
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import feedparser
import pandas as pd
import requests
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in some test envs
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("crypto_research_agent")

T = TypeVar("T")

LOG_PATH = Path("logs") / "research_log.jsonl"
DEFAULT_TIMEOUT = 12.0
DEFAULT_RETRIES = 3

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINPAPRIKA_BASE_URL = "https://api.coinpaprika.com/v1"
CRYPTOCOMPARE_BASE_URL = "https://min-api.cryptocompare.com/data/v2"
CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1"
COINDESK_DATA_BASE_URL = "https://min-api.cryptocompare.com/data/v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEPTH_PCT_WINDOW = 0.005
SCAN_TOP_N = int(os.getenv("SCAN_TOP_N", "10"))
SCAN_SCOPE_ALL = "all"
SCAN_SCOPE_MEME = "meme_high_volatility"
SCAN_SCOPES = (SCAN_SCOPE_ALL, SCAN_SCOPE_MEME)
MEME_SPIKE_THRESHOLD = float(os.getenv("MEME_SPIKE_THRESHOLD", "2.0"))
KLINE_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}
SCANNER_SCORE_WEIGHTS = {
    "liquidity_24h": 0.30,
    "volatility_24h": 0.22,
    "movement_15m": 0.18,
    "spread_quality": 0.10,
    "orderbook_depth": 0.20,
}
SCALPING_MIN_VOLATILITY_PCT = float(os.getenv("SCALPING_MIN_VOLATILITY_PCT", "1.5"))
TECHNICAL_MISSING_FIELD_THRESHOLD = float(os.getenv("TECHNICAL_MISSING_FIELD_THRESHOLD", "0.4"))
KLINE_SUCCESS_CACHE_TTL_SECONDS = int(os.getenv("KLINE_SUCCESS_CACHE_TTL_SECONDS", "300"))
FUNDAMENTALS_CACHE_TTL_SECONDS = int(os.getenv("FUNDAMENTALS_CACHE_TTL_SECONDS", "600"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3"))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "120"))
SCORING_FORMULA = (
    "score = 100 * ("
    f"{SCANNER_SCORE_WEIGHTS['liquidity_24h']:.2f}*norm(log1p(24h_quote_volume)) + "
    f"{SCANNER_SCORE_WEIGHTS['volatility_24h']:.2f}*norm(24h_range_pct) + "
    f"{SCANNER_SCORE_WEIGHTS['movement_15m']:.2f}*norm(recent_15m_range_pct) + "
    f"{SCANNER_SCORE_WEIGHTS['spread_quality']:.2f}*norm(spread_quality_from_bid_ask) + "
    f"{SCANNER_SCORE_WEIGHTS['orderbook_depth']:.2f}*norm(depth_notional_within_0.5pct_mid)"
    ") - thin_liquidity_penalty"
)

PEGGED_ASSET_DENYLIST = {
    "USDT",
    "USDC",
    "FDUSD",
    "TUSD",
    "DAI",
    "USDP",
    "PYUSD",
    "EURI",
    "XAUT",
    "PAXG",
}

PEGGED_CATEGORY_MARKERS = (
    "stablecoin",
    "tokenized gold",
    "gold",
    "tokenized commodity",
    "tokenized stock",
    "tokenized securities",
)

MANUAL_REVIEW_CATEGORY_MARKERS = (
    "tokenized",
    "synthetic",
    "wrapped",
    "derivative",
    "stock",
    "security",
)

REQUIRED_TECH_FIELDS = ("sma20", "ema9", "ema21", "rsi14", "support", "resistance")
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

STABLE_SYMBOLS = {
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "DAI",
    "FDUSD",
    "USDP",
    "PYUSD",
    "EURT",
    "FRAX",
    "UST",
    "LUSD",
    "USDQ",
}


@dataclass(frozen=True)
class CoinIdentity:
    """Resolved market identity for a coin."""

    query: str
    binance_symbol: str | None
    coingecko_id: str | None
    coingecko_symbol: str | None
    coingecko_name: str | None


_provider_lock = threading.Lock()
_provider_state: dict[str, dict[str, Any]] = {}
_kline_success_cache: dict[tuple[str, str], dict[str, Any]] = {}
_fundamentals_cache: dict[str, dict[str, Any]] = {}
_last_scanner_stats: dict[str, Any] = {}
_last_ticker_collection_stats: dict[str, Any] = {}


def _provider_name_from_url(url: str) -> str:
    """Map URL host to a normalized provider name for logging/health."""

    lowered = (url or "").lower()
    if "binance.com" in lowered:
        return "binance"
    if "coingecko.com" in lowered:
        return "coingecko"
    if "coinpaprika.com" in lowered:
        return "coinpaprika"
    if "cryptopanic.com" in lowered:
        return "cryptopanic"
    if "cryptocompare.com" in lowered:
        return "coindesk_data"
    return "unknown"


def _exception_details(exc: Exception) -> dict[str, Any]:
    """Extract stable exception metadata for structured logs."""

    status_code = None
    response = getattr(exc, "response", None)
    response_excerpt = None
    if response is not None:
        status_code = getattr(response, "status_code", None)
        try:
            text = str(getattr(response, "text", "") or "")
            response_excerpt = text[:300] if text else None
        except Exception:
            response_excerpt = None
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "status_code": status_code,
        "response_excerpt": response_excerpt,
    }


def _log_structured(
    *,
    level: int,
    component: str,
    outcome: str,
    provider: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    latency_ms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit structured JSON logs that are easy to grep and post-process."""

    event: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "outcome": outcome,
    }
    if provider:
        event["provider"] = provider
    if symbol:
        event["symbol"] = symbol
    if timeframe:
        event["timeframe"] = timeframe
    if latency_ms is not None:
        event["latency_ms"] = round(latency_ms, 2)
    if extra:
        event.update(extra)
    logger.log(level, json.dumps(event, ensure_ascii=True, default=str))


def _provider_circuit_open(provider: str) -> bool:
    """Return True if provider circuit breaker is currently open."""

    now = time.time()
    with _provider_lock:
        state = _provider_state.get(provider, {})
        open_until = float(state.get("open_until", 0.0) or 0.0)
    return open_until > now


def _provider_record_success(provider: str) -> None:
    """Reset failure counter and store last successful timestamp."""

    with _provider_lock:
        state = _provider_state.setdefault(provider, {})
        state["failures"] = 0
        state["open_until"] = 0.0
        state["last_success_ts"] = datetime.now(timezone.utc).isoformat()


def _provider_record_failure(provider: str, details: dict[str, Any]) -> None:
    """Increment failure counter and open circuit when threshold is exceeded."""

    now = time.time()
    with _provider_lock:
        state = _provider_state.setdefault(provider, {})
        failures = int(state.get("failures", 0) or 0) + 1
        state["failures"] = failures
        state["last_error"] = details
        state["last_failure_ts"] = datetime.now(timezone.utc).isoformat()
        if failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            state["open_until"] = now + CIRCUIT_BREAKER_COOLDOWN_SECONDS


def get_provider_health_snapshot() -> dict[str, Any]:
    """Expose provider health and circuit state for API diagnostics."""

    provider_names = (
        "binance",
        "coingecko",
        "coinpaprika",
        "rss",
        "cryptopanic",
        "coindesk_data",
        "groq",
        "gemini",
    )
    now = time.time()
    with _provider_lock:
        snapshot = {name: dict(_provider_state.get(name, {})) for name in provider_names}
    payload: dict[str, Any] = {}
    for name, state in snapshot.items():
        open_until = float(state.get("open_until", 0.0) or 0.0)
        payload[name] = {
            "last_success_ts": state.get("last_success_ts"),
            "last_failure_ts": state.get("last_failure_ts"),
            "failures": int(state.get("failures", 0) or 0),
            "circuit_open": open_until > now,
            "circuit_open_seconds_left": max(0, int(open_until - now)),
            "last_error": state.get("last_error"),
        }
    return payload


def ensure_log_dir() -> None:
    """Create the log directory if it does not already exist."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(record: dict[str, Any]) -> None:
    """Append a JSON record to the local research log."""

    ensure_log_dir()
    payload = dict(record)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Any:
    """Fetch JSON with timeout and exponential backoff."""

    provider = _provider_name_from_url(url)
    if _provider_circuit_open(provider):
        _log_structured(
            level=logging.WARNING,
            component="fetch_json",
            outcome="circuit_open",
            provider=provider,
            extra={"url": url},
        )
        raise RuntimeError(f"{provider} circuit open - skipping request")

    started = time.time()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            _provider_record_success(provider)
            _log_structured(
                level=logging.INFO,
                component="fetch_json",
                outcome="success",
                provider=provider,
                latency_ms=(time.time() - started) * 1000.0,
                extra={"url": url, "attempt": attempt + 1, "status_code": response.status_code},
            )
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            details = _exception_details(exc)
            if attempt < retries - 1:
                sleep_for = 2 ** attempt
                _log_structured(
                    level=logging.WARNING,
                    component="fetch_json",
                    outcome="retry",
                    provider=provider,
                    latency_ms=(time.time() - started) * 1000.0,
                    extra={"url": url, "attempt": attempt + 1, "sleep_s": sleep_for, **details},
                )
                time.sleep(sleep_for)
            else:
                _provider_record_failure(provider, details)
                _log_structured(
                    level=logging.ERROR,
                    component="fetch_json",
                    outcome="failed",
                    provider=provider,
                    latency_ms=(time.time() - started) * 1000.0,
                    extra={"url": url, "attempt": attempt + 1, **details},
                )
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url}")


def fetch_with_fallback(providers: list[tuple[str, Callable[[], T]]] | list[Callable[[], T]]) -> T:
    """Try provider callables in order until one succeeds."""

    normalized: list[tuple[str, Callable[[], T]]] = []
    for index, provider in enumerate(providers, start=1):
        if isinstance(provider, tuple) and len(provider) == 2:
            normalized.append((str(provider[0]), provider[1]))
        else:
            normalized.append((f"provider_{index}", provider))

    last_error: Exception | None = None
    for index, provider_record in enumerate(normalized, start=1):
        provider_name, provider = provider_record
        if _provider_circuit_open(provider_name):
            _log_structured(
                level=logging.WARNING,
                component="fetch_with_fallback",
                outcome="circuit_open",
                provider=provider_name,
                extra={"index": index},
            )
            continue
        try:
            payload = provider()
            _provider_record_success(provider_name)
            _log_structured(
                level=logging.INFO,
                component="fetch_with_fallback",
                outcome="success",
                provider=provider_name,
                extra={"index": index},
            )
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            details = _exception_details(exc)
            _provider_record_failure(provider_name, details)
            _log_structured(
                level=logging.WARNING,
                component="fetch_with_fallback",
                outcome="failed",
                provider=provider_name,
                extra={"index": index, **details},
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("No providers supplied")


def _is_http_status(exc: Exception, status_code: int) -> bool:
    """Return True when an exception wraps a specific HTTP status code."""

    if not isinstance(exc, requests.HTTPError):
        return False
    response = getattr(exc, "response", None)
    return bool(response is not None and response.status_code == status_code)


def _now_ms() -> int:
    """Return current UTC timestamp in milliseconds."""

    return int(time.time() * 1000)


def _is_stale_timestamp_ms(timestamp_ms: int, max_age_seconds: int) -> bool:
    """Check whether a millisecond timestamp is older than the allowed age."""

    return (_now_ms() - timestamp_ms) > (max_age_seconds * 1000)


def _validate_not_stale(timestamp_ms: int, max_age_seconds: int, label: str) -> None:
    """Raise when a timestamp is older than the acceptable freshness window."""

    if _is_stale_timestamp_ms(timestamp_ms, max_age_seconds):
        raise RuntimeError(f"Stale market data for {label}: timestamp={timestamp_ms}")


def _groq_retryable_errors() -> tuple[type[BaseException], ...]:
    """Return Groq exception types that should trigger failover."""

    errors: list[type[BaseException]] = [requests.HTTPError, requests.Timeout, requests.ConnectionError, ValueError, RuntimeError]
    try:
        import groq as groq_module  # type: ignore

        for name in (
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "APIStatusError",
            "APIError",
        ):
            error_type = getattr(groq_module, name, None)
            if isinstance(error_type, type) and issubclass(error_type, BaseException):
                errors.append(error_type)
    except Exception:
        pass
    return tuple(dict.fromkeys(errors))


def _gemini_retryable_errors() -> tuple[type[BaseException], ...]:
    """Return Gemini exception types that should trigger failover."""

    errors: list[type[BaseException]] = [requests.HTTPError, requests.Timeout, requests.ConnectionError, ValueError, RuntimeError]
    try:
        from google import genai  # type: ignore
        try:
            from google.genai import errors as gemini_errors  # type: ignore
        except Exception:
            gemini_errors = None

        if gemini_errors is not None:
            for name in (
                "APIError",
                "ClientError",
                "ServerError",
                "TooManyRequests",
                "BadRequest",
                "Unauthorized",
                "PermissionDenied",
                "ResourceExhausted",
                "DeadlineExceeded",
            ):
                error_type = getattr(gemini_errors, name, None)
                if isinstance(error_type, type) and issubclass(error_type, BaseException):
                    errors.append(error_type)
    except Exception:
        pass
    return tuple(dict.fromkeys(errors))


def _get_groq_client() -> Any:
    """Create a Groq client from the environment."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required.")
    from groq import Groq  # type: ignore

    return Groq(api_key=api_key)


def _get_gemini_client():
    """Create a Gemini client from the environment on demand."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")
    from google import genai  # type: ignore

    return genai.Client(api_key=api_key)


def _call_groq(system_prompt: str, user_content: str) -> str:
    """Call Groq with the configured default model."""

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def _call_gemini(system_prompt: str, user_content: str) -> str:
    """Call Gemini with the configured default model."""

    client = _get_gemini_client()
    prompt = f"{system_prompt}\n\n{user_content}"
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response")
    return text.strip()


def _llm_provider_chain() -> list[tuple[str, str, Callable[[str, str], str], tuple[type[BaseException], ...]]]:
    """Build an LLM provider chain using only configured API keys."""

    providers: list[tuple[str, str, Callable[[str, str], str], tuple[type[BaseException], ...]]] = []
    if os.getenv("GROQ_API_KEY"):
        providers.append(("Groq", "groq", _call_groq, _groq_retryable_errors()))
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("Gemini", "gemini", _call_gemini, _gemini_retryable_errors()))
    return providers


def fetch_llm_with_fallback(system_prompt: str, user_content: str) -> str:
    """Call Groq first, then Gemini if Groq fails or is rate limited."""

    provider_attempts = _llm_provider_chain()
    if not provider_attempts:
        raise RuntimeError("LLM summary unavailable - no provider configured")

    last_error: Exception | None = None
    for provider_name, provider_key, provider_fn, retryable_errors in provider_attempts:
        if _provider_circuit_open(provider_key):
            _log_structured(
                level=logging.WARNING,
                component="llm",
                outcome="circuit_open",
                provider=provider_key,
            )
            continue
        for attempt in range(1, 3):
            try:
                logger.info("LLM attempt %s/%s via %s", attempt, 2, provider_name)
                content = provider_fn(system_prompt, user_content)
                _provider_record_success(provider_key)
                return content
            except retryable_errors as exc:
                last_error = exc
                _provider_record_failure(provider_key, _exception_details(exc))
                logger.warning("%s attempt %s failed: %s", provider_name, attempt, exc)
                if attempt < 2:
                    time.sleep(2 * (2 ** (attempt - 1)))
            except Exception as exc:
                last_error = exc
                _provider_record_failure(provider_key, _exception_details(exc))
                logger.warning("%s attempt %s failed with non-retryable error: %s", provider_name, attempt, exc)
                if attempt < 2:
                    time.sleep(2 * (2 ** (attempt - 1)))
                else:
                    break
        logger.warning("Switching to fallback provider after %s failures.", provider_name)
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM fallback chain exhausted without an error")


@lru_cache(maxsize=1)
def get_binance_exchange_info() -> dict[str, Any]:
    """Fetch Binance exchange metadata."""

    try:
        return fetch_json(f"{BINANCE_BASE_URL}/exchangeInfo")
    except Exception as exc:
        if _is_http_status(exc, 451):
            logger.warning(
                "Binance exchangeInfo is region-blocked (HTTP 451). Falling back to non-Binance providers for symbol discovery."
            )
            return {"symbols": []}
        raise


@lru_cache(maxsize=1)
def get_binance_all_tickers_raw() -> list[dict[str, Any]]:
    """Fetch the Binance 24h ticker snapshot for all symbols."""

    data = fetch_json(f"{BINANCE_BASE_URL}/ticker/24hr")
    if not isinstance(data, list):
        raise ValueError("Unexpected Binance ticker payload")
    fresh = []
    for row in data:
        if not isinstance(row, dict):
            continue
        close_time = int(_to_float(row.get("closeTime"))) if row.get("closeTime") is not None else 0
        if close_time and _is_stale_timestamp_ms(close_time, 900):
            continue
        fresh.append(row)
    if not fresh:
        raise RuntimeError("All Binance ticker rows were stale")
    return fresh


def _get_symbol_book_ticker(symbol: str) -> dict[str, Any]:
    """Fetch real-time bid/ask for one symbol from Binance bookTicker endpoint."""

    payload = fetch_json(f"{BINANCE_BASE_URL}/ticker/bookTicker", params={"symbol": symbol.upper()})
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected bookTicker payload for {symbol}")
    return payload


def _get_symbol_depth_snapshot(symbol: str, limit: int = 50) -> dict[str, Any]:
    """Fetch depth and compute near-mid liquidity totals within +/-0.5%."""

    payload = fetch_json(f"{BINANCE_BASE_URL}/depth", params={"symbol": symbol.upper(), "limit": limit})
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected depth payload for {symbol}")

    bids = payload.get("bids", [])
    asks = payload.get("asks", [])
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        raise ValueError(f"Depth snapshot missing bids/asks for {symbol}")

    best_bid = _to_float(bids[0][0]) if isinstance(bids[0], list) and len(bids[0]) >= 2 else 0.0
    best_ask = _to_float(asks[0][0]) if isinstance(asks[0], list) and len(asks[0]) >= 2 else 0.0
    if best_bid <= 0 or best_ask <= 0:
        raise ValueError(f"Invalid best bid/ask in depth snapshot for {symbol}")

    mid = (best_bid + best_ask) / 2.0
    low_cut = mid * (1.0 - DEPTH_PCT_WINDOW)
    high_cut = mid * (1.0 + DEPTH_PCT_WINDOW)

    bid_notional = 0.0
    ask_notional = 0.0
    for level in bids:
        if not isinstance(level, list) or len(level) < 2:
            continue
        px = _to_float(level[0])
        qty = _to_float(level[1])
        if px >= low_cut:
            bid_notional += px * qty
    for level in asks:
        if not isinstance(level, list) or len(level) < 2:
            continue
        px = _to_float(level[0])
        qty = _to_float(level[1])
        if px <= high_cut:
            ask_notional += px * qty

    total_notional = bid_notional + ask_notional
    imbalance = ((bid_notional - ask_notional) / total_notional) if total_notional > 0 else 0.0
    return {
        "mid_price": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth_notional_0_5pct": bid_notional,
        "ask_depth_notional_0_5pct": ask_notional,
        "depth_notional_0_5pct": total_notional,
        "depth_imbalance": imbalance,
        "depth_source": "binance_depth_limit_50_0.5pct_mid",
    }
    return data


@lru_cache(maxsize=1)
def get_binance_book_tickers_raw() -> list[dict[str, Any]]:
    """Fetch Binance best bid/ask snapshots."""

    data = fetch_json(f"{BINANCE_BASE_URL}/ticker/bookTicker")
    if not isinstance(data, list):
        raise ValueError("Unexpected Binance bookTicker payload")
    return data


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a safe default."""

    try:
        if value in (None, "", "null"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_float(value: Any) -> float | None:
    """Convert a value to float, preserving None for missing fields."""

    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbol(value: str) -> str:
    """Normalize a human token into an uppercase alphanumeric symbol."""

    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _looks_like_stablecoin(symbol: str, name: str | None = None) -> bool:
    """Return True when the asset is likely a stablecoin."""

    symbol_upper = symbol.upper()
    if symbol_upper in STABLE_SYMBOLS:
        return True
    name_upper = (name or "").upper()
    stable_markers = ("STABLE", "USD", "USDT", "USDC", "DAI", "TETHER", "PYUSD")
    return any(marker in symbol_upper for marker in stable_markers) or any(
        marker in name_upper for marker in stable_markers
    )


def _extract_quote_asset(symbol: str) -> str:
    """Extract quote asset suffix from common quote currencies."""

    known_quotes = ("USDT", "USDC", "FDUSD", "BUSD", "BTC", "ETH")
    symbol_upper = symbol.upper()
    for quote in known_quotes:
        if symbol_upper.endswith(quote):
            return quote
    return ""


@lru_cache(maxsize=512)
def _coingecko_categories_for_symbol(base_symbol: str) -> tuple[str, ...]:
    """Resolve CoinGecko categories for a base ticker symbol."""

    normalized = _normalize_symbol(base_symbol)
    if not normalized:
        return ()
    for market in _iter_coingecko_markets():
        if _normalize_symbol(str(market.get("symbol", ""))) != normalized:
            continue
        coin_id = market.get("id")
        if not coin_id:
            continue
        try:
            detail = _coingecko_detail(str(coin_id))
            categories = detail.get("categories", []) if isinstance(detail, dict) else []
            if not isinstance(categories, list):
                return ()
            return tuple(str(cat) for cat in categories if str(cat).strip())
        except Exception as exc:
            logger.warning("CoinGecko category lookup failed for %s: %s", normalized, exc)
            return ()
    return ()


def _is_pegged_or_non_scalp_asset(base_asset: str, quote_asset: str, categories: Iterable[str]) -> tuple[bool, str | None]:
    """Detect pegged assets from symbol denylist and CoinGecko categories."""

    base_upper = base_asset.upper()
    quote_upper = quote_asset.upper()
    if base_upper in PEGGED_ASSET_DENYLIST:
        return True, f"base asset {base_upper} is in pegged denylist"
    # The scanner universe is USDT pairs; quote-asset exclusion is kept for non-USDT universes.
    if quote_upper and quote_upper != "USDT" and quote_upper in PEGGED_ASSET_DENYLIST:
        return True, f"quote asset {quote_upper} is in pegged denylist"

    category_blob = " ".join(str(cat).lower() for cat in categories)
    for marker in PEGGED_CATEGORY_MARKERS:
        if marker in category_blob:
            return True, f"CoinGecko category matched pegged marker '{marker}'"
    return False, None


def _needs_manual_review(base_asset: str, categories: Iterable[str]) -> bool:
    """Flag unusual assets for manual review instead of silent inclusion."""

    if not re.fullmatch(r"[A-Z]{2,8}", base_asset.upper()):
        return True
    category_blob = " ".join(str(cat).lower() for cat in categories)
    return any(marker in category_blob for marker in MANUAL_REVIEW_CATEGORY_MARKERS)


def _split_levels_by_price(levels: list[dict[str, float]], close_price: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Split levels by current price so support stays below and resistance above."""

    supports = [lvl for lvl in levels if _to_float(lvl.get("level")) < close_price]
    resistances = [lvl for lvl in levels if _to_float(lvl.get("level")) > close_price]
    return supports, resistances


def _validate_support_resistance(
    *,
    close_price: float,
    support: float | None,
    resistance: float | None,
    label: str,
) -> tuple[float | None, float | None]:
    """Validate directional constraints and drop invalid support/resistance levels."""

    valid_support = support if support is not None and support < close_price else None
    valid_resistance = resistance if resistance is not None and resistance > close_price else None
    if support is not None and valid_support is None:
        logger.error("Invalid support dropped for %s: support=%s close=%s", label, support, close_price)
    if resistance is not None and valid_resistance is None:
        logger.error("Invalid resistance dropped for %s: resistance=%s close=%s", label, resistance, close_price)

    if valid_support is not None and valid_resistance is not None:
        try:
            assert valid_support < close_price < valid_resistance
        except AssertionError:
            logger.error(
                "Support/resistance assertion failed for %s: support=%s close=%s resistance=%s",
                label,
                valid_support,
                close_price,
                valid_resistance,
            )
            valid_support = None
            valid_resistance = None
    return valid_support, valid_resistance


def validate_technical_readiness(
    research: dict[str, Any],
    *,
    missing_threshold: float = TECHNICAL_MISSING_FIELD_THRESHOLD,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Gate deep-dive LLM calls when technical fields are mostly missing."""

    technicals = research.get("technicals", {}) if isinstance(research, dict) else {}
    if not isinstance(technicals, dict) or not technicals:
        symbol = str(research.get("input", "asset")) if isinstance(research, dict) else "asset"
        return False, f"technical data unavailable for {symbol} - retry", {"missing_ratio": 1.0}

    total = 0
    missing = 0
    missing_by_timeframe: dict[str, float] = {}
    for timeframe, snapshot in technicals.items():
        if not isinstance(snapshot, dict):
            continue
        tf_total = len(REQUIRED_TECH_FIELDS)
        tf_missing = sum(1 for field in REQUIRED_TECH_FIELDS if snapshot.get(field) is None)
        total += tf_total
        missing += tf_missing
        missing_by_timeframe[str(timeframe)] = (tf_missing / tf_total) if tf_total else 1.0

    ratio = (missing / total) if total else 1.0
    if ratio > missing_threshold:
        symbol = str(research.get("input", "asset")) if isinstance(research, dict) else "asset"
        return (
            False,
            f"technical data unavailable for {symbol} - retry",
            {
                "missing_ratio": round(ratio, 4),
                "threshold": missing_threshold,
                "missing_by_timeframe": missing_by_timeframe,
            },
        )
    return True, None, {"missing_ratio": round(ratio, 4), "missing_by_timeframe": missing_by_timeframe}


def _build_binance_symbol(base_symbol: str) -> str:
    """Convert a base ticker like BTC into a Binance USDT pair."""

    normalized = _normalize_symbol(base_symbol)
    if normalized.endswith("USDT"):
        return normalized
    return f"{normalized}USDT"


@lru_cache(maxsize=8)
def _coingecko_markets_page(page: int) -> tuple[dict[str, Any], ...]:
    """Fetch a page of CoinGecko market data."""

    headers: dict[str, str] = {}
    demo_key = os.getenv("COINGECKO_API_KEY")
    if demo_key:
        headers["x-cg-demo-api-key"] = demo_key
    payload = fetch_json(
        f"{COINGECKO_BASE_URL}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false",
        },
        headers=headers or None,
    )
    if not isinstance(payload, list):
        raise ValueError("Unexpected CoinGecko markets payload")
    return tuple(item for item in payload if isinstance(item, dict))


def _iter_coingecko_markets(max_pages: int = 4) -> Iterable[dict[str, Any]]:
    """Iterate over a limited number of CoinGecko market pages."""

    for page in range(1, max_pages + 1):
        try:
            yield from _coingecko_markets_page(page)
        except Exception as exc:
            logger.warning("CoinGecko markets page %s unavailable: %s", page, exc)
            break


@lru_cache(maxsize=1)
def _coinpaprika_coins() -> tuple[dict[str, Any], ...]:
    """Fetch the CoinPaprika coin directory for fallback resolution."""

    payload = fetch_json(f"{COINPAPRIKA_BASE_URL}/coins")
    if not isinstance(payload, list):
        raise ValueError("Unexpected CoinPaprika coins payload")
    return tuple(item for item in payload if isinstance(item, dict))


def _resolve_coinpaprika_id(symbol_or_name: str) -> str | None:
    """Resolve a CoinPaprika coin id from a symbol or project name."""

    normalized = _normalize_symbol(symbol_or_name)
    for item in _coinpaprika_coins():
        if not item.get("is_active", True):
            continue
        symbol = _normalize_symbol(str(item.get("symbol", "")))
        name = _normalize_symbol(str(item.get("name", "")))
        if normalized in {symbol, name} or normalized == symbol or normalized in name:
            return str(item.get("id"))
    return None


def _coingecko_usdt_tickers_provider() -> list[dict[str, Any]]:
    """Build Binance-like ticker rows from CoinGecko market data."""

    tickers: list[dict[str, Any]] = []
    for market in _iter_coingecko_markets():
        symbol = str(market.get("symbol", "")).upper()
        name = str(market.get("name", ""))
        if not symbol or _looks_like_stablecoin(symbol, name):
            continue
        last_price = _to_float(market.get("current_price"))
        if last_price <= 0:
            continue
        high_price = _to_float(market.get("high_24h"), last_price)
        low_price = _to_float(market.get("low_24h"), last_price)
        volume = _to_float(market.get("total_volume"))
        tickers.append(
            {
                "symbol": f"{symbol}USDT",
                "base_asset": symbol,
                "last_price": last_price,
                "price_change_percent": _to_float(market.get("price_change_percentage_24h")),
                "quote_volume": volume,
                "volume": volume,
                "high_price": high_price,
                "low_price": low_price,
                "spread_pct": 0.0,
            }
        )
    if not tickers:
        raise ValueError("No CoinGecko market rows available")
    return tickers


def _coinpaprika_usdt_tickers_provider() -> list[dict[str, Any]]:
    """Best-effort CoinPaprika fallback for USDT-style ticker rows."""

    if _provider_circuit_open("coinpaprika"):
        raise RuntimeError("coinpaprika circuit open - skipping provider")

    tickers: list[dict[str, Any]] = []
    for item in _coinpaprika_coins():
        if not item.get("is_active", True):
            continue
        symbol = str(item.get("symbol", "")).upper()
        name = str(item.get("name", ""))
        if not symbol or _looks_like_stablecoin(symbol, name):
            continue
        coin_id = item.get("id")
        if not coin_id:
            continue
        try:
            payload = fetch_json(f"{COINPAPRIKA_BASE_URL}/tickers/{coin_id}")
        except Exception as exc:
            logger.warning("CoinPaprika ticker lookup failed for %s: %s", coin_id, exc)
            continue
        if not isinstance(payload, dict):
            continue
        usd = payload.get("quotes", {}).get("USD", {})
        last_price = _to_float(usd.get("price"))
        if last_price <= 0:
            continue
        tickers.append(
            {
                "symbol": f"{symbol}USDT",
                "base_asset": symbol,
                "last_price": last_price,
                "price_change_percent": _to_float(usd.get("percent_change_24h")),
                "quote_volume": _to_float(usd.get("volume_24h")),
                "volume": _to_float(payload.get("circulating_supply")),
                "high_price": _to_float(usd.get("high_24h"), last_price),
                "low_price": _to_float(usd.get("low_24h"), last_price),
                "spread_pct": 0.0,
            }
        )
        if len(tickers) >= 120:
            break
    if not tickers:
        raise ValueError("No CoinPaprika ticker rows available")
    return tickers


def resolve_coin_identity(symbol_or_name: str) -> CoinIdentity:
    """Resolve a user input into Binance and CoinGecko identities."""

    query = symbol_or_name.strip()
    normalized = _normalize_symbol(query)
    binance_symbol = _build_binance_symbol(normalized)

    exchange_symbols = {
        item.get("symbol")
        for item in get_binance_exchange_info().get("symbols", [])
        if item.get("status") == "TRADING" and item.get("quoteAsset") == "USDT"
    }
    if binance_symbol not in exchange_symbols:
        binance_symbol = None

    matched_market: dict[str, Any] | None = None
    for market in _iter_coingecko_markets():
        market_symbol = _normalize_symbol(str(market.get("symbol", "")))
        market_name = _normalize_symbol(str(market.get("name", "")))
        if normalized in {market_symbol, market_name} or normalized in market_name or normalized == market_symbol:
            matched_market = market
            break

    coingecko_id = matched_market.get("id") if matched_market else None
    coingecko_symbol = matched_market.get("symbol") if matched_market else None
    coingecko_name = matched_market.get("name") if matched_market else None
    if coingecko_id is None and normalized:
        for market in _iter_coingecko_markets():
            if _normalize_symbol(str(market.get("symbol", ""))) == normalized:
                matched_market = market
                coingecko_id = market.get("id")
                coingecko_symbol = market.get("symbol")
                coingecko_name = market.get("name")
                break

    return CoinIdentity(
        query=query,
        binance_symbol=binance_symbol,
        coingecko_id=coingecko_id,
        coingecko_symbol=coingecko_symbol,
        coingecko_name=coingecko_name,
    )


def get_all_usdt_tickers() -> list[dict[str, Any]]:
    """Return normalized Binance USDT tickers for actively trading pairs."""

    def binance_provider() -> list[dict[str, Any]]:
        global _last_ticker_collection_stats

        exchange_info = get_binance_exchange_info()
        active_symbols = {
            item.get("symbol")
            for item in exchange_info.get("symbols", [])
            if item.get("status") == "TRADING" and item.get("quoteAsset") == "USDT"
        }
        tickers: list[dict[str, Any]] = []
        skipped_stable = 0
        skipped_nonpositive_price = 0
        skipped_stale = 0
        skipped_inactive_symbol = 0
        raw_rows = 0
        for raw in get_binance_all_tickers_raw():
            raw_rows += 1
            symbol = raw.get("symbol")
            if symbol not in active_symbols:
                skipped_inactive_symbol += 1
                continue
            base_asset = str(symbol or "")[:-4]
            if _looks_like_stablecoin(base_asset):
                skipped_stable += 1
                continue
            price = _to_float(raw.get("lastPrice"))
            if price <= 0:
                skipped_nonpositive_price += 1
                continue
            close_time = int(_to_float(raw.get("closeTime"))) if raw.get("closeTime") is not None else 0
            if close_time:
                try:
                    _validate_not_stale(close_time, 900, f"ticker:{symbol}")
                except Exception as exc:
                    skipped_stale += 1
                    _log_structured(
                        level=logging.WARNING,
                        component="ticker_freshness",
                        outcome="stale_row_skipped",
                        provider="binance",
                        symbol=str(symbol),
                        extra=_exception_details(exc),
                    )
                    continue
            tickers.append(
                {
                    "symbol": symbol,
                    "base_asset": base_asset,
                    "last_price": price,
                    "price_change_percent": _to_float(raw.get("priceChangePercent")),
                    "quote_volume": _to_float(raw.get("quoteVolume")),
                    "volume": _to_float(raw.get("volume")),
                    "high_price": _to_float(raw.get("highPrice")),
                    "low_price": _to_float(raw.get("lowPrice")),
                    "spread_pct": None,
                    "spread_source": "pending_symbol_book_ticker",
                }
            )
        _last_ticker_collection_stats = {
            "provider": "binance",
            "raw_rows": int(raw_rows),
            "active_symbol_count": int(len(active_symbols)),
            "accepted_rows": int(len(tickers)),
            "skipped_inactive_symbol": int(skipped_inactive_symbol),
            "skipped_stable": int(skipped_stable),
            "skipped_nonpositive_price": int(skipped_nonpositive_price),
            "skipped_stale": int(skipped_stale),
        }
        if not tickers:
            raise ValueError("No Binance tickers available")
        return tickers

    return fetch_with_fallback([
        ("binance", binance_provider),
        ("coingecko", _coingecko_usdt_tickers_provider),
        ("coinpaprika", _coinpaprika_usdt_tickers_provider),
    ])


def get_klines(symbol: str, interval: str, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch and parse Binance klines into a friendly list of dictionaries."""

    payload = fetch_json(
        f"{BINANCE_BASE_URL}/klines",
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
    )
    if not isinstance(payload, list):
        raise ValueError("Unexpected Binance kline payload")
    parsed: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 12:
            continue
        parsed.append(
            {
                "open_time": int(row[0]),
                "open": _to_float(row[1]),
                "high": _to_float(row[2]),
                "low": _to_float(row[3]),
                "close": _to_float(row[4]),
                "volume": _to_float(row[5]),
                "close_time": int(row[6]),
                "quote_volume": _to_float(row[7]),
                "trade_count": int(row[8]) if str(row[8]).isdigit() else 0,
                "taker_buy_base_volume": _to_float(row[9]),
                "taker_buy_quote_volume": _to_float(row[10]),
                "ignored": row[11],
            }
        )
    if parsed:
        latest_close_time = int(parsed[-1]["close_time"])
        max_age_seconds = (KLINE_INTERVAL_SECONDS.get(interval, 60) * 3) + 120
        _validate_not_stale(latest_close_time, max_age_seconds, f"kline:{symbol}:{interval}")
    return parsed


def _kline_cache_key(symbol: str, interval: str) -> tuple[str, str]:
    """Normalize cache keys for per-symbol/timeframe kline snapshots."""

    return (symbol.upper(), interval)


def _store_kline_cache(symbol: str, interval: str, klines: list[dict[str, Any]]) -> None:
    """Persist the last successful kline fetch for short-lived fallback use."""

    _kline_success_cache[_kline_cache_key(symbol, interval)] = {
        "klines": list(klines),
        "updated_at": time.time(),
    }


def _get_kline_cache(symbol: str, interval: str) -> tuple[list[dict[str, Any]] | None, float | None]:
    """Return cached klines and age in seconds when still within cache TTL."""

    record = _kline_success_cache.get(_kline_cache_key(symbol, interval))
    if not record:
        return None, None
    updated_at = float(record.get("updated_at", 0.0) or 0.0)
    if updated_at <= 0:
        return None, None
    age_s = max(0.0, time.time() - updated_at)
    if age_s > KLINE_SUCCESS_CACHE_TTL_SECONDS:
        return None, None
    cached = record.get("klines")
    if not isinstance(cached, list) or not cached:
        return None, None
    return list(cached), age_s


def fetch_klines_resilient(symbol: str, interval: str, limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch live klines and fallback to short-lived cache when fresh fetch fails."""

    started = time.time()
    try:
        klines = get_klines(symbol, interval, limit=limit)
        _store_kline_cache(symbol, interval, klines)
        _log_structured(
            level=logging.INFO,
            component="kline_fetch",
            outcome="live",
            provider="binance",
            symbol=symbol.upper(),
            timeframe=interval,
            latency_ms=(time.time() - started) * 1000.0,
            extra={"rows": len(klines)},
        )
        return klines, {
            "is_stale": False,
            "stale_age_seconds": 0,
            "stale_badge": None,
            "source": "live",
        }
    except Exception as exc:
        details = _exception_details(exc)
        _log_structured(
            level=logging.ERROR,
            component="kline_fetch",
            outcome="failed",
            provider="binance",
            symbol=symbol.upper(),
            timeframe=interval,
            latency_ms=(time.time() - started) * 1000.0,
            extra=details,
        )
        cached, age_s = _get_kline_cache(symbol, interval)
        if cached:
            stale_age = int(age_s or 0)
            _log_structured(
                level=logging.WARNING,
                component="kline_fetch",
                outcome="cache_fallback",
                provider="binance",
                symbol=symbol.upper(),
                timeframe=interval,
                extra={"stale_age_seconds": stale_age, **details},
            )
            return cached, {
                "is_stale": True,
                "stale_age_seconds": stale_age,
                "stale_badge": f"stale - last updated {stale_age}s ago",
                "source": "cache",
                "fetch_error": details,
            }
        raise


def _klines_frame(klines: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert parsed klines to a pandas DataFrame with numeric columns."""

    frame = pd.DataFrame(klines)
    if frame.empty:
        return frame
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["open", "high", "low", "close"])


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using the standard Wilder smoothing approximation."""

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    return 100.0 - (100.0 / (1.0 + rs))


def _technical_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute indicators and structural levels for one timeframe."""

    if frame.empty:
        return {
            "latest_close": None,
            "sma20": None,
            "ema9": None,
            "ema21": None,
            "rsi14": None,
            "rsi_label": "unknown",
            "support": None,
            "resistance": None,
            "support_levels": [],
            "resistance_levels": [],
            "trend": "unknown",
            "trend_label": "unknown",
            "wick_flags": [],
            "wick_flag_rate_pct": None,
            "wick_flag_baseline_pct": None,
            "wick_risk_unusual": None,
            "manipulation_count": None,
        }

    enriched = frame.copy()
    enriched["sma20"] = enriched["close"].rolling(window=20, min_periods=1).mean()
    enriched["ema9"] = enriched["close"].ewm(span=9, adjust=False).mean()
    enriched["ema21"] = enriched["close"].ewm(span=21, adjust=False).mean()
    enriched["rsi14"] = _rsi(enriched["close"], 14)
    enriched["body"] = (enriched["close"] - enriched["open"]).abs()
    enriched["upper_wick"] = enriched["high"] - enriched[["open", "close"]].max(axis=1)
    enriched["lower_wick"] = enriched[["open", "close"]].min(axis=1) - enriched["low"]
    enriched["wick_size"] = enriched[["upper_wick", "lower_wick"]].max(axis=1)
    enriched["wick_mean20"] = enriched["wick_size"].rolling(window=20, min_periods=5).mean()
    enriched["wick_std20"] = enriched["wick_size"].rolling(window=20, min_periods=5).std().fillna(0.0)
    enriched["vol_ma20"] = enriched["volume"].rolling(window=20, min_periods=5).mean()
    enriched["wick_extreme"] = enriched["wick_size"] > (enriched["wick_mean20"] + (2.0 * enriched["wick_std20"]))
    enriched["volume_spike"] = enriched["volume"] > (1.5 * enriched["vol_ma20"].replace(0.0, pd.NA))
    enriched["wick_flag"] = enriched["wick_extreme"] & enriched["volume_spike"].fillna(False)

    recent_window = min(len(enriched), 20)
    recent_flags = enriched["wick_flag"].tail(recent_window)
    baseline_flags = enriched["wick_flag"].iloc[:-recent_window] if len(enriched) > recent_window else pd.Series(dtype=bool)
    recent_rate_pct = _to_float(recent_flags.mean()) * 100.0 if recent_window > 0 else 0.0
    baseline_rate_pct = (_to_float(baseline_flags.mean()) * 100.0) if len(baseline_flags) > 0 else recent_rate_pct
    wick_risk_unusual = (recent_rate_pct > (baseline_rate_pct * 1.5)) and (recent_rate_pct >= 5.0)

    wick_flags = enriched.loc[
        enriched["wick_flag"],
        ["open_time", "wick_size", "high", "low", "close", "volume", "vol_ma20"],
    ]

    recent = enriched.tail(min(len(enriched), 20))
    latest = enriched.iloc[-1]
    latest_close = round(_to_float(latest["close"]), 8)
    pivot_levels = _pivot_levels_from_frame(enriched, flank=2, cluster_tolerance_pct=0.004)
    all_levels = [
        *(pivot_levels.get("supports", [])),
        *(pivot_levels.get("resistances", [])),
    ]
    support_levels, resistance_levels = _split_levels_by_price(all_levels, latest_close)
    support = support_levels[0]["level"] if support_levels else float(recent["low"].min())
    resistance = resistance_levels[0]["level"] if resistance_levels else float(recent["high"].max())
    support, resistance = _validate_support_resistance(
        close_price=latest_close,
        support=support,
        resistance=resistance,
        label="technical_snapshot",
    )

    support_levels = [lvl for lvl in support_levels if _to_float(lvl.get("level")) < latest_close]
    resistance_levels = [lvl for lvl in resistance_levels if _to_float(lvl.get("level")) > latest_close]
    gap_series = (enriched["ema9"] - enriched["ema21"]).dropna()
    gap_now = _to_float(gap_series.iloc[-1]) if len(gap_series) > 0 else 0.0
    gap_prev = _to_float(gap_series.iloc[-4]) if len(gap_series) > 3 else (_to_float(gap_series.iloc[0]) if len(gap_series) > 0 else 0.0)
    gap_slope = gap_now - gap_prev
    trend_label = "sideways"
    if gap_now > 0 and gap_slope > 0:
        trend_label = "bullish_strengthening"
    elif gap_now > 0 and gap_slope <= 0:
        trend_label = "bullish_weakening"
    elif gap_now < 0 and gap_slope < 0:
        trend_label = "bearish_strengthening"
    elif gap_now < 0 and gap_slope >= 0:
        trend_label = "bearish_weakening"

    rsi_value = _to_float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else None
    rsi_label = _label_rsi(rsi_value)

    return {
        "latest_close": latest_close,
        "sma20": round(_to_float(latest["sma20"]), 8),
        "ema9": round(_to_float(latest["ema9"]), 8),
        "ema21": round(_to_float(latest["ema21"]), 8),
        "rsi14": round(rsi_value, 2) if rsi_value is not None else None,
        "rsi_label": rsi_label,
        "support": round(support, 8) if support is not None else None,
        "resistance": round(resistance, 8) if resistance is not None else None,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "trend": trend_label,
        "trend_label": trend_label,
        "wick_flags": [
            {
                "open_time": int(row.open_time),
                "wick_size": round(_to_float(row.wick_size), 8),
                "high": round(_to_float(row.high), 8),
                "low": round(_to_float(row.low), 8),
                "close": round(_to_float(row.close), 8),
                "volume": round(_to_float(row.volume), 4),
                "vol_ma20": round(_to_float(row.vol_ma20), 4),
            }
            for row in wick_flags.itertuples(index=False)
        ],
        "wick_flag_rate_pct": round(recent_rate_pct, 2),
        "wick_flag_baseline_pct": round(baseline_rate_pct, 2),
        "wick_risk_unusual": bool(wick_risk_unusual),
        "manipulation_count": int(len(wick_flags)),
    }


def _label_rsi(rsi_value: float | None) -> str:
    """Map RSI value to deterministic categorical labels."""

    if rsi_value is None:
        return "unknown"
    if rsi_value < 30:
        return "oversold"
    if rsi_value < 45:
        return "weak"
    if rsi_value <= 55:
        return "neutral"
    if rsi_value <= 70:
        return "strong"
    return "overbought"


def _pivot_levels_from_frame(frame: pd.DataFrame, flank: int = 2, cluster_tolerance_pct: float = 0.004) -> dict[str, list[dict[str, float]]]:
    """Detect fractal pivots and cluster the most touched support/resistance levels."""

    if frame.empty or len(frame) < (2 * flank + 1):
        return {"supports": [], "resistances": []}

    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for idx in range(flank, len(frame) - flank):
        left_highs = highs[idx - flank : idx]
        right_highs = highs[idx + 1 : idx + flank + 1]
        left_lows = lows[idx - flank : idx]
        right_lows = lows[idx + 1 : idx + flank + 1]
        current_high = highs[idx]
        current_low = lows[idx]
        if current_high > max(left_highs) and current_high > max(right_highs):
            swing_highs.append(float(current_high))
        if current_low < min(left_lows) and current_low < min(right_lows):
            swing_lows.append(float(current_low))

    def _cluster(levels: list[float]) -> list[dict[str, float]]:
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters: list[list[float]] = []
        for level in sorted_levels:
            placed = False
            for cluster in clusters:
                center = sum(cluster) / len(cluster)
                if abs(level - center) / max(center, 1e-9) <= cluster_tolerance_pct:
                    cluster.append(level)
                    placed = True
                    break
            if not placed:
                clusters.append([level])
        ranked = sorted(
            [{"level": sum(cluster) / len(cluster), "touches": float(len(cluster))} for cluster in clusters],
            key=lambda x: x["touches"],
            reverse=True,
        )
        return ranked[:3]

    return {
        "supports": _cluster(swing_lows),
        "resistances": _cluster(swing_highs),
    }


def _cluster_levels(levels: list[float], tolerance_pct: float = 0.004, keep_top: int = 3) -> list[dict[str, float]]:
    """Cluster raw price levels and return the most-touched aggregate levels."""

    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters: list[list[float]] = []
    for level in sorted_levels:
        placed = False
        for cluster in clusters:
            center = sum(cluster) / len(cluster)
            if abs(level - center) / max(center, 1e-9) <= tolerance_pct:
                cluster.append(level)
                placed = True
                break
        if not placed:
            clusters.append([level])
    ranked = sorted(
        [{"level": round(sum(cluster) / len(cluster), 8), "touches": float(len(cluster))} for cluster in clusters],
        key=lambda x: x["touches"],
        reverse=True,
    )
    return ranked[:keep_top]


def _compute_15m_movement(klines: list[dict[str, Any]], *, freshness: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure the recent 15m candle range and average movement."""

    frame = _klines_frame(klines)
    if frame.empty:
        return {
            "latest_range_pct": 0.0,
            "avg_range_pct": 0.0,
            "is_stale": bool((freshness or {}).get("is_stale", False)),
            "stale_age_seconds": (freshness or {}).get("stale_age_seconds"),
            "stale_badge": (freshness or {}).get("stale_badge"),
        }
    frame["range_pct"] = ((frame["high"] - frame["low"]) / frame["close"].replace(0.0, pd.NA)) * 100.0
    latest_range_pct = _to_float(frame.iloc[-1]["range_pct"])
    avg_range_pct = _to_float(frame["range_pct"].tail(8).mean())
    return {
        "latest_range_pct": latest_range_pct,
        "avg_range_pct": avg_range_pct,
        "is_stale": bool((freshness or {}).get("is_stale", False)),
        "stale_age_seconds": (freshness or {}).get("stale_age_seconds"),
        "stale_badge": (freshness or {}).get("stale_badge"),
    }


def _ticker_volatility_pct(ticker: dict[str, Any]) -> float:
    """Compute volatility with fallbacks when high/low are unavailable."""

    last_price = _to_float(ticker.get("last_price"))
    high_price = _to_float(ticker.get("high_price"))
    low_price = _to_float(ticker.get("low_price"))
    if last_price > 0 and high_price > 0 and low_price > 0 and high_price >= low_price:
        return ((high_price - low_price) / last_price) * 100.0
    return abs(_to_float(ticker.get("price_change_percent")))


def _score_candidate(
    ticker: dict[str, Any],
    *,
    max_volume: float,
    max_volatility: float,
    max_range: float,
    min_spread: float,
    max_depth: float,
) -> float:
    """Compute a normalized scalping score from raw market statistics."""

    components = _score_components(
        ticker,
        max_volume=max_volume,
        max_volatility=max_volatility,
        max_range=max_range,
        min_spread=min_spread,
        max_depth=max_depth,
    )
    score = sum(components["weighted"].values()) - components["thin_liquidity_penalty"]
    return round(score * 100.0, 2)


def _score_components(
    ticker: dict[str, Any],
    *,
    max_volume: float,
    max_volatility: float,
    max_range: float,
    min_spread: float,
    max_depth: float,
) -> dict[str, Any]:
    """Return normalized and weighted score components for scanner transparency."""

    volume_norm = math.log1p(ticker["quote_volume"]) / math.log1p(max_volume) if max_volume > 0 else 0.0
    volatility_pct = _ticker_volatility_pct(ticker)
    volatility_norm = volatility_pct / max_volatility if max_volatility > 0 else 0.0
    movement_norm = ticker.get("latest_range_pct", 0.0) / max_range if max_range > 0 else 0.0
    spread_norm = 1.0 - min(1.0, ticker.get("spread_pct", 0.0) / max(min_spread, 0.01))
    depth_norm = ticker.get("depth_notional_0_5pct", 0.0) / max_depth if max_depth > 0 else 0.0
    thin_liquidity_penalty = 0.0
    if volatility_pct > 12.0 and depth_norm < 0.20:
        thin_liquidity_penalty = 0.20
    weighted = {
        "liquidity_24h": SCANNER_SCORE_WEIGHTS["liquidity_24h"] * volume_norm,
        "volatility_24h": SCANNER_SCORE_WEIGHTS["volatility_24h"] * volatility_norm,
        "movement_15m": SCANNER_SCORE_WEIGHTS["movement_15m"] * movement_norm,
        "spread_quality": SCANNER_SCORE_WEIGHTS["spread_quality"] * spread_norm,
        "orderbook_depth": SCANNER_SCORE_WEIGHTS["orderbook_depth"] * depth_norm,
    }
    return {
        "normalized": {
            "liquidity_24h": volume_norm,
            "volatility_24h": volatility_norm,
            "movement_15m": movement_norm,
            "spread_quality": spread_norm,
            "orderbook_depth": depth_norm,
        },
        "weighted": weighted,
        "thin_liquidity_penalty": thin_liquidity_penalty,
        "total_score": round((sum(weighted.values()) - thin_liquidity_penalty) * 100.0, 2),
    }


def _is_meme_category(categories: Iterable[str]) -> bool:
    """Classify meme/high-volatility categories from CoinGecko labels."""

    blob = " ".join(str(cat).lower() for cat in categories)
    markers = (
        "meme",
        "dog",
        "dog-themed",
        "dog themed",
        "frog",
        "cat",
        "animal",
    )
    return any(marker in blob for marker in markers)


def _compute_volume_spike_ratio(symbol: str, today_quote_volume: float) -> tuple[float | None, float | None, dict[str, Any]]:
    """Compare today's quote volume against trailing 7 daily candles."""

    klines, freshness = fetch_klines_resilient(symbol, "1d", limit=7)
    frame = _klines_frame(klines)
    if frame.empty or "quote_volume" not in frame.columns:
        return None, None, freshness
    trailing_avg = _to_float(frame["quote_volume"].mean())
    if trailing_avg <= 0:
        return None, trailing_avg, freshness
    return (today_quote_volume / trailing_avg), trailing_avg, freshness


def _recent_direction_label(symbol: str) -> tuple[str, dict[str, Any]]:
    """Label whether recent hourly candles are still pumping, flat, or reversing."""

    klines, freshness = fetch_klines_resilient(symbol, "1h", limit=8)
    frame = _klines_frame(klines)
    if frame.empty or len(frame) < 4:
        return "unknown", freshness
    closes = frame["close"].tail(4).tolist()
    moves = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    up_count = sum(1 for value in moves if value > 0)
    down_count = sum(1 for value in moves if value < 0)
    if up_count >= 2 and moves[-1] > 0:
        return "still pumping", freshness
    if down_count >= 2 and moves[-1] < 0:
        return "showing reversal signs", freshness
    return "flat since spike", freshness


def find_top_scalping_candidates(
    top_n: int = SCAN_TOP_N,
    *,
    scope: str = SCAN_SCOPE_ALL,
    return_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, int]]:
    """Scan Binance USDT pairs and rank the best scalp candidates."""

    scope_value = scope if scope in SCAN_SCOPES else SCAN_SCOPE_ALL
    tickers = [item for item in get_all_usdt_tickers() if item["quote_volume"] > 0]
    if not tickers:
        return []

    quote_volumes = [item["quote_volume"] for item in tickers]
    max_volume = max(quote_volumes)
    volatilities = [_ticker_volatility_pct(item) for item in tickers if item.get("last_price", 0) > 0]
    max_volatility = max(volatilities) if volatilities else 0.0
    min_spread = 0.05

    filtered = [item for item in tickers if item["quote_volume"] >= 250_000 and item["last_price"] > 0]
    filtered.sort(key=lambda item: (item["quote_volume"], _ticker_volatility_pct(item)), reverse=True)
    initial_pool = filtered[:40]

    enriched: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    skipped_pegged = 0
    skipped_volatility_floor = 0
    for ticker in initial_pool:
        symbol = str(ticker.get("symbol", ""))
        base_asset = str(ticker.get("base_asset", ""))
        quote_asset = _extract_quote_asset(symbol)
        categories = _coingecko_categories_for_symbol(base_asset)
        is_pegged, pegged_reason = _is_pegged_or_non_scalp_asset(base_asset, quote_asset, categories)
        volatility_pct = _ticker_volatility_pct(ticker)
        if is_pegged:
            skipped_pegged += 1
            exclusion = {"symbol": symbol, "reason": f"pegged_exclusion: {pegged_reason}"}
            excluded.append(exclusion)
            logger.info("Excluded scanner symbol %s - %s", symbol, exclusion["reason"])
            continue
        if volatility_pct < SCALPING_MIN_VOLATILITY_PCT:
            skipped_volatility_floor += 1
            exclusion = {
                "symbol": symbol,
                "reason": f"below_volatility_floor: {volatility_pct:.4f}% < {SCALPING_MIN_VOLATILITY_PCT:.4f}%",
            }
            excluded.append(exclusion)
            logger.info("Excluded scanner symbol %s - %s", symbol, exclusion["reason"])
            continue
        try:
            movement_klines, movement_freshness = fetch_klines_resilient(ticker["symbol"], "15m", limit=24)
            movement = _compute_15m_movement(movement_klines, freshness=movement_freshness)
        except Exception as exc:
            details = _exception_details(exc)
            _log_structured(
                level=logging.ERROR,
                component="scanner_15m",
                outcome="failed",
                provider="binance",
                symbol=ticker["symbol"],
                timeframe="15m",
                extra=details,
            )
            movement = {
                "latest_range_pct": 0.0,
                "avg_range_pct": 0.0,
                "is_stale": False,
                "stale_age_seconds": None,
                "stale_badge": None,
            }
        try:
            book = _get_symbol_book_ticker(ticker["symbol"])
            bid = _to_float(book.get("bidPrice"))
            ask = _to_float(book.get("askPrice"))
            spread_pct = (((ask - bid) / ask) * 100.0) if ask > 0 and bid > 0 and ask >= bid else 0.0
            spread_source = "binance_bookTicker_symbol_bid_ask"
        except Exception as exc:
            logger.warning("bookTicker spread fetch failed for %s: %s", ticker["symbol"], exc)
            spread_pct = 0.0
            spread_source = "spread_unavailable"

        try:
            depth_snapshot = _get_symbol_depth_snapshot(ticker["symbol"], limit=50)
        except Exception as exc:
            logger.warning("Depth fetch failed for %s: %s", ticker["symbol"], exc)
            depth_snapshot = {
                "mid_price": ticker["last_price"],
                "best_bid": None,
                "best_ask": None,
                "bid_depth_notional_0_5pct": 0.0,
                "ask_depth_notional_0_5pct": 0.0,
                "depth_notional_0_5pct": 0.0,
                "depth_imbalance": 0.0,
                "depth_source": "depth_unavailable",
            }

        merged = dict(ticker)
        merged.update(movement)
        merged["spread_pct"] = spread_pct
        merged["spread_source"] = spread_source
        merged["coingecko_categories"] = list(categories)
        merged["needs_manual_review"] = _needs_manual_review(base_asset, categories)
        merged.update(depth_snapshot)

        spike_ratio = None
        avg_daily_quote_volume_7d = None
        direction_label = "unknown"
        daily_freshness = {"is_stale": False, "stale_age_seconds": None, "stale_badge": None}
        direction_freshness = {"is_stale": False, "stale_age_seconds": None, "stale_badge": None}
        try:
            spike_ratio, avg_daily_quote_volume_7d, daily_freshness = _compute_volume_spike_ratio(
                ticker["symbol"],
                _to_float(ticker.get("quote_volume")),
            )
        except Exception as exc:
            _log_structured(
                level=logging.WARNING,
                component="scanner_spike",
                outcome="failed",
                provider="binance",
                symbol=ticker["symbol"],
                timeframe="1d",
                extra=_exception_details(exc),
            )
        try:
            direction_label, direction_freshness = _recent_direction_label(ticker["symbol"])
        except Exception as exc:
            _log_structured(
                level=logging.WARNING,
                component="scanner_direction",
                outcome="failed",
                provider="binance",
                symbol=ticker["symbol"],
                timeframe="1h",
                extra=_exception_details(exc),
            )

        merged["today_quote_volume"] = _to_float(ticker.get("quote_volume"))
        merged["avg_daily_quote_volume_7d"] = avg_daily_quote_volume_7d
        merged["spike_ratio"] = spike_ratio
        merged["recent_direction_label"] = direction_label
        merged["is_meme_category"] = _is_meme_category(categories)
        merged["stale_badges"] = [
            badge
            for badge in (
                movement.get("stale_badge"),
                daily_freshness.get("stale_badge"),
                direction_freshness.get("stale_badge"),
            )
            if badge
        ]
        enriched.append(merged)

    if excluded:
        append_jsonl({"mode": "scanner_exclusions", "input": None, "output": {"excluded": excluded}})

    overall_max_range = max((item.get("latest_range_pct", 0.0) for item in enriched), default=0.0)
    max_depth = max((item.get("depth_notional_0_5pct", 0.0) for item in enriched), default=0.0)
    spread_values = [item.get("spread_pct", 0.0) for item in enriched if item.get("spread_pct", 0.0) > 0]
    min_spread = min(spread_values) if spread_values else 0.05
    for item in enriched:
        components = _score_components(
            item,
            max_volume=max_volume,
            max_volatility=max_volatility,
            max_range=overall_max_range,
            min_spread=min_spread,
            max_depth=max_depth,
        )
        item["score_components"] = components
        item["score"] = float(components.get("total_score", 0.0))

    if scope_value == SCAN_SCOPE_MEME:
        meme_filtered = [
            item
            for item in enriched
            if item.get("is_meme_category")
            or (_to_optional_float(item.get("spike_ratio")) is not None and _to_float(item.get("spike_ratio")) >= MEME_SPIKE_THRESHOLD)
        ]
        ranked = sorted(
            meme_filtered,
            key=lambda item: (
                _to_float(item.get("spike_ratio"), 0.0),
                _ticker_volatility_pct(item),
                _to_float(item.get("quote_volume"), 0.0),
            ),
            reverse=True,
        )[:top_n]
    else:
        ranked = sorted(enriched, key=lambda item: item["score"], reverse=True)[:top_n]

    results: list[dict[str, Any]] = []
    for index, item in enumerate(ranked, start=1):
        volatility_pct = _ticker_volatility_pct(item)
        reason_parts = [
            f"rank #{index}",
            f"24h quote volume ${item['quote_volume']:,.0f}",
            f"24h range {volatility_pct:.2f}%",
            f"15m range {item.get('latest_range_pct', 0.0):.2f}%",
            f"spread {item.get('spread_pct', 0.0):.4f}%",
        ]
        if item.get("spread_pct", 0.0) <= min_spread * 1.5:
            reason_parts.append("tight spread supports scalp execution")
        if volatility_pct >= max_volatility * 0.75:
            reason_parts.append("above-average intraday volatility")
        if scope_value == SCAN_SCOPE_MEME and item.get("spike_ratio") is not None:
            reason_parts.append(f"volume spike {_to_float(item.get('spike_ratio')):.2f}x vs 7d baseline")
        results.append(
            {
                "rank": index,
                "symbol": item["symbol"],
                "base_asset": item["base_asset"],
                "score": item["score"],
                "last_price": round(item["last_price"], 8),
                "quote_volume": round(item["quote_volume"], 2),
                "today_quote_volume": round(_to_float(item.get("today_quote_volume")), 2),
                "avg_daily_quote_volume_7d": round(_to_float(item.get("avg_daily_quote_volume_7d")), 2)
                if item.get("avg_daily_quote_volume_7d") is not None
                else None,
                "spike_ratio": round(_to_float(item.get("spike_ratio")), 3) if item.get("spike_ratio") is not None else None,
                "price_change_percent": round(item["price_change_percent"], 2),
                "volatility_pct": round(volatility_pct, 2),
                "recent_15m_range_pct": round(item.get("latest_range_pct", 0.0), 2),
                "spread_pct": round(item.get("spread_pct", 0.0), 4),
                "spread_source": item.get("spread_source", "unknown"),
                "depth_notional_0_5pct": round(item.get("depth_notional_0_5pct", 0.0), 2),
                "depth_source": item.get("depth_source", "unknown"),
                "recent_direction_label": item.get("recent_direction_label", "unknown"),
                "scope": scope_value,
                "needs_manual_review": bool(item.get("needs_manual_review", False)),
                "coingecko_categories": item.get("coingecko_categories", []),
                "stale_badges": item.get("stale_badges", []),
                "score_components": item.get("score_components", {}),
                "reason": "; ".join(reason_parts),
            }
        )
    score_component_preview = [
        {
            "symbol": row.get("symbol"),
            "score": row.get("score"),
            "components": row.get("score_components", {}),
        }
        for row in results[: min(5, len(results))]
    ]
    stats_payload: dict[str, Any] = {
        "requested_top_n": int(top_n),
        "scope": scope_value,
        "counts": {
            "eligible_tickers_after_basic_filter": int(len(tickers)),
            "entered_initial_pool": int(len(initial_pool)),
            "skipped_pegged": int(skipped_pegged),
            "skipped_volatility_floor": int(skipped_volatility_floor),
            "entered_scoring_pool": int(len(enriched)),
            "returned_count": int(len(results)),
            "skipped_stale_or_bad_rows": int(_to_float(_last_ticker_collection_stats.get("skipped_stale"), 0.0))
            + int(_to_float(_last_ticker_collection_stats.get("skipped_nonpositive_price"), 0.0)),
        },
        "thresholds": {
            "scalping_min_volatility_pct": SCALPING_MIN_VOLATILITY_PCT,
            "meme_spike_threshold": MEME_SPIKE_THRESHOLD,
            "meme_category_markers": ["meme", "dog", "dog-themed", "dog themed", "frog", "cat", "animal"],
        },
        "ticker_collection": dict(_last_ticker_collection_stats),
        "score_component_preview": score_component_preview,
    }
    global _last_scanner_stats
    _last_scanner_stats = stats_payload
    if return_stats:
        return results, stats_payload
    return results


def _coingecko_detail(coin_id: str) -> dict[str, Any]:
    """Fetch detailed CoinGecko coin fundamentals."""

    headers: dict[str, str] = {}
    demo_key = os.getenv("COINGECKO_API_KEY")
    if demo_key:
        headers["x-cg-demo-api-key"] = demo_key
    payload = fetch_json(
        f"{COINGECKO_BASE_URL}/coins/{coin_id}",
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        },
        headers=headers or None,
    )
    if not isinstance(payload, dict):
        raise ValueError("Unexpected CoinGecko coin detail payload")
    return payload


def _coinpaprika_fundamentals(coin_id: str) -> dict[str, Any]:
    """Fetch CoinPaprika fundamentals when a mapped coin id is available."""

    payload = fetch_json(f"{COINPAPRIKA_BASE_URL}/tickers/{coin_id}")
    if not isinstance(payload, dict):
        raise ValueError("Unexpected CoinPaprika payload")
    return payload


def get_coin_fundamentals(symbol_or_name: str) -> dict[str, Any]:
    """Resolve a coin and return its fundamentals from the available providers."""

    identity = resolve_coin_identity(symbol_or_name)
    coinpaprika_id = _resolve_coinpaprika_id(symbol_or_name)
    cache_key = (identity.coingecko_id or coinpaprika_id or _normalize_symbol(symbol_or_name)).lower()
    now = time.time()
    cache_entry = _fundamentals_cache.get(cache_key)
    if cache_entry and float(cache_entry.get("expires_at", 0.0) or 0.0) > now:
        cached = dict(cache_entry.get("payload") or {})
        cached["fundamentals_cache_hit"] = True
        return cached
    if not identity.coingecko_id and not coinpaprika_id:
        raise ValueError(f"Could not resolve coin id for {symbol_or_name!r}")

    def cg_provider() -> dict[str, Any]:
        return _coingecko_detail(identity.coingecko_id or "")

    def cp_provider() -> dict[str, Any]:
        return _coinpaprika_fundamentals(coinpaprika_id or identity.coingecko_id or "")

    payload = fetch_with_fallback([("coingecko", cg_provider), ("coinpaprika", cp_provider)])
    result: dict[str, Any]
    if "market_data" in payload:
        market_data = payload.get("market_data", {})
        community_data = payload.get("community_data", {})
        developer_data = payload.get("developer_data", {})
        current_price = _to_optional_float(market_data.get("current_price", {}).get("usd"))
        ath = _to_optional_float(market_data.get("ath", {}).get("usd"))
        max_supply = _to_optional_float(market_data.get("max_supply"))
        fdv = (current_price * max_supply) if current_price is not None and max_supply is not None else None
        pct_from_ath = None
        if current_price is not None and ath not in (None, 0):
            pct_from_ath = ((current_price - ath) / ath) * 100.0
        result = {
            "resolved_input": identity.query,
            "binance_symbol": identity.binance_symbol,
            "coingecko_id": identity.coingecko_id,
            "name": payload.get("name"),
            "symbol": payload.get("symbol"),
            "current_price": current_price,
            "market_cap": _to_optional_float(market_data.get("market_cap", {}).get("usd")),
            "volume_24h": _to_optional_float(market_data.get("total_volume", {}).get("usd")),
            "circulating_supply": _to_optional_float(market_data.get("circulating_supply")),
            "total_supply": _to_optional_float(market_data.get("total_supply")),
            "max_supply": max_supply,
            "ath": ath,
            "ath_date": market_data.get("ath_date", {}).get("usd"),
            "atl": _to_optional_float(market_data.get("atl", {}).get("usd")),
            "atl_date": market_data.get("atl_date", {}).get("usd"),
            "fdv_estimated": fdv,
            "pct_from_ath": pct_from_ath,
            "categories": payload.get("categories", []),
            "homepage": (payload.get("links", {}).get("homepage") or [None])[0],
            "blockchain_site": (payload.get("links", {}).get("blockchain_site") or [None])[0],
            "community_score": _to_optional_float(community_data.get("facebook_likes") or community_data.get("reddit_average_comments_48h")),
            "developer_score": _to_optional_float(developer_data.get("forks")),
            "market_cap_rank": _to_optional_float(payload.get("market_cap_rank")),
            "coingecko_rank": _to_optional_float(payload.get("market_cap_rank")),
            "price_change_24h_pct": _to_optional_float(market_data.get("price_change_percentage_24h")),
            "fundamentals_degraded": False,
            "fundamentals_sources": ["coingecko"],
            "fundamentals_cache_hit": False,
        }
        _fundamentals_cache[cache_key] = {"payload": dict(result), "expires_at": now + FUNDAMENTALS_CACHE_TTL_SECONDS}
        return result
    if "quotes" in payload:
        usd = payload.get("quotes", {}).get("USD", {})
        current_price = _to_optional_float(usd.get("price"))
        ath = _to_optional_float(usd.get("ath_price"))
        max_supply = _to_optional_float(payload.get("max_supply"))
        fdv = (current_price * max_supply) if current_price is not None and max_supply is not None else None
        pct_from_ath = None
        if current_price is not None and ath not in (None, 0):
            pct_from_ath = ((current_price - ath) / ath) * 100.0
        result = {
            "resolved_input": identity.query,
            "binance_symbol": identity.binance_symbol,
            "coingecko_id": identity.coingecko_id,
            "name": payload.get("name"),
            "symbol": payload.get("symbol"),
            "current_price": current_price,
            "market_cap": _to_optional_float(usd.get("market_cap")),
            "volume_24h": _to_optional_float(usd.get("volume_24h")),
            "circulating_supply": _to_optional_float(payload.get("circulating_supply")),
            "total_supply": _to_optional_float(payload.get("total_supply")),
            "max_supply": max_supply,
            "ath": ath,
            "ath_date": usd.get("ath_date"),
            "atl": _to_optional_float(usd.get("atl_price")),
            "atl_date": usd.get("atl_date"),
            "fdv_estimated": fdv,
            "pct_from_ath": pct_from_ath,
            "categories": payload.get("tags", []),
            "homepage": payload.get("website_link"),
            "blockchain_site": payload.get("explorer"),
            "community_score": None,
            "developer_score": None,
            "market_cap_rank": _to_optional_float(payload.get("rank")),
            "coingecko_rank": _to_optional_float(payload.get("rank")),
            "price_change_24h_pct": _to_optional_float(usd.get("percent_change_24h")),
            "fundamentals_degraded": False,
            "fundamentals_sources": ["coinpaprika"],
            "fundamentals_cache_hit": False,
        }
        _fundamentals_cache[cache_key] = {"payload": dict(result), "expires_at": now + FUNDAMENTALS_CACHE_TTL_SECONDS}
        return result
    raise ValueError("Unable to parse fundamentals payload")


def _minimal_fundamentals_from_binance(symbol_or_name: str) -> dict[str, Any]:
    """Return minimal fundamentals from Binance ticker as last resort fallback."""

    identity = resolve_coin_identity(symbol_or_name)
    if not identity.binance_symbol:
        raise ValueError("No Binance symbol available for minimal fundamentals fallback")
    payload = fetch_json(f"{BINANCE_BASE_URL}/ticker/24hr", params={"symbol": identity.binance_symbol})
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Binance ticker payload for minimal fundamentals")
    last_price = _to_optional_float(payload.get("lastPrice"))
    return {
        "resolved_input": identity.query,
        "binance_symbol": identity.binance_symbol,
        "coingecko_id": identity.coingecko_id,
        "name": identity.coingecko_name or identity.query,
        "symbol": identity.coingecko_symbol or identity.query,
        "current_price": last_price,
        "market_cap": None,
        "volume_24h": _to_optional_float(payload.get("quoteVolume")),
        "circulating_supply": None,
        "total_supply": None,
        "max_supply": None,
        "ath": None,
        "ath_date": None,
        "atl": None,
        "atl_date": None,
        "fdv_estimated": None,
        "pct_from_ath": None,
        "categories": [],
        "homepage": None,
        "blockchain_site": None,
        "community_score": None,
        "developer_score": None,
        "market_cap_rank": None,
        "coingecko_rank": None,
        "price_change_24h_pct": _to_optional_float(payload.get("priceChangePercent")),
        "fundamentals_degraded": True,
        "fundamentals_sources": ["binance_minimal"],
        "fundamentals_cache_hit": False,
    }


def _matches_coin_news(text: str, coin_name: str, coin_symbol: str | None) -> bool:
    """Match coin-specific mentions with word-boundary checks to reduce false positives."""

    haystack = text or ""
    tokens = [coin_name.strip(), (coin_symbol or "").strip()]
    for token in tokens:
        if not token:
            continue
        pattern = rf"\b{re.escape(token)}\b"
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return True
    return False


def _dedupe_news_by_url(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate news items by URL while preserving order."""

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        url = str(item.get("url") or "").strip()
        key = url.lower() if url else f"{item.get('title')}|{item.get('published_time')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_feed_published(entry: Any) -> str | None:
    """Normalize RSS entry timestamps to ISO-8601 UTC strings."""

    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed is None:
        return None
    try:
        ts = calendar.timegm(parsed)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _rss_news_provider(coin_name: str, coin_symbol: str | None, limit: int = 8) -> tuple[list[dict[str, Any]], bool]:
    """Fetch keyless crypto RSS feeds and return coin-matched items or fallback headlines."""

    if _provider_circuit_open("rss"):
        raise RuntimeError("rss circuit open - skipping request")

    combined: list[dict[str, Any]] = []
    feed_failures = 0
    for source_name, feed_url in RSS_FEEDS.items():
        parsed = feedparser.parse(feed_url)
        entries = getattr(parsed, "entries", []) or []
        if not entries:
            feed_failures += 1
            continue
        for entry in entries[:30]:
            title = str(getattr(entry, "title", "") or "")
            summary = str(getattr(entry, "summary", "") or "")
            link = str(getattr(entry, "link", "") or "")
            combined.append(
                {
                    "title": title,
                    "source": source_name,
                    "published_time": _parse_feed_published(entry),
                    "url": link,
                    "summary": summary,
                    "is_general_market_news": False,
                }
            )

    if feed_failures == len(RSS_FEEDS) or not combined:
        _provider_record_failure("rss", {"exception_type": "RuntimeError", "exception_message": "RSS layer unavailable", "status_code": None})
        raise RuntimeError("RSS layer unavailable")

    _provider_record_success("rss")

    combined = _dedupe_news_by_url(combined)
    combined.sort(key=lambda item: item.get("published_time") or "", reverse=True)

    matched = [
        item
        for item in combined
        if _matches_coin_news(
            f"{item.get('title', '')} {item.get('summary', '')}",
            coin_name,
            coin_symbol,
        )
    ]
    if matched:
        return matched[:limit], False

    general = []
    for item in combined[:3]:
        labeled = dict(item)
        labeled["is_general_market_news"] = True
        labeled["summary"] = (
            f"general market news, not {(coin_symbol or coin_name)}-specific. "
            f"{str(item.get('summary', '')).strip()}"
        ).strip()
        general.append(labeled)
    return general, True


def _cryptopanic_news_provider(coin_symbol: str | None, limit: int = 8) -> tuple[list[dict[str, Any]], bool]:
    """Fetch optional CryptoPanic tagged news when API key is present."""

    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    if not api_key or not (coin_symbol or "").strip():
        return [], False
    params = {"auth_token": api_key, "public": "true", "currencies": (coin_symbol or "").upper()}
    payload = fetch_json(f"{CRYPTOPANIC_BASE_URL}/posts/", params=params)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        raise ValueError("Unexpected CryptoPanic payload")
    mapped = [
        {
            "title": item.get("title"),
            "source": (item.get("source") or {}).get("title") if isinstance(item.get("source"), dict) else item.get("source"),
            "published_time": item.get("published_at"),
            "url": item.get("url"),
            "summary": item.get("title"),
            "is_general_market_news": False,
        }
        for item in results[:limit]
    ]
    return mapped, True


def _coindesk_data_news_provider(coin_name: str, coin_symbol: str | None, limit: int = 8) -> tuple[list[dict[str, Any]], bool]:
    """Fetch optional CoinDesk Data (CryptoCompare) news when key is configured."""

    api_key = os.getenv("COINDESK_DATA_API_KEY")
    if not api_key:
        return [], False

    headers = {"authorization": f"Apikey {api_key}"}
    payload = fetch_json(f"{COINDESK_DATA_BASE_URL}/news/", params={"lang": "EN"}, headers=headers)
    articles = payload.get("Data", []) if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        raise ValueError("Unexpected CoinDesk Data news payload")

    mapped: list[dict[str, Any]] = []
    for item in articles:
        title = str(item.get("title", ""))
        body = str(item.get("body", ""))
        if not _matches_coin_news(f"{title} {body}", coin_name, coin_symbol):
            continue
        mapped.append(
            {
                "title": item.get("title"),
                "source": (item.get("source_info", {}) or {}).get("name") or item.get("source") or "CoinDesk Data",
                "published_time": datetime.fromtimestamp(int(item.get("published_on", 0)), tz=timezone.utc).isoformat()
                if item.get("published_on")
                else None,
                "url": item.get("url"),
                "summary": item.get("body"),
                "is_general_market_news": False,
            }
        )
        if len(mapped) >= limit:
            break
    return mapped, True


def _select_news_items(coin_name: str, coin_symbol: str | None) -> dict[str, Any]:
    """Fetch news using RSS-first fallback chain with optional keyed enrichments."""

    sources_used: list[str] = []
    news_items: list[dict[str, Any]] = []
    rss_general_fallback = False

    try:
        rss_items, rss_general_fallback = _rss_news_provider(coin_name, coin_symbol, limit=8)
        news_items.extend(rss_items)
        sources_used.append("rss")
    except Exception as exc:
        logger.warning("RSS news provider unavailable: %s", exc)
        return {
            "news": [],
            "news_status": "unavailable",
            "news_sources_used": [],
            "news_note": "rss_unavailable",
        }

    try:
        cp_items, used = _cryptopanic_news_provider(coin_symbol, limit=8)
        if used and cp_items:
            news_items.extend(cp_items)
            sources_used.append("cryptopanic")
    except Exception as exc:
        logger.warning("CryptoPanic news provider failed: %s", exc)

    try:
        cd_items, used = _coindesk_data_news_provider(coin_name, coin_symbol, limit=8)
        if used and cd_items:
            news_items.extend(cd_items)
            sources_used.append("coindesk_data")
    except Exception as exc:
        logger.warning("CoinDesk Data news provider failed: %s", exc)

    news_items = _dedupe_news_by_url(news_items)
    news_items.sort(key=lambda item: item.get("published_time") or "", reverse=True)
    news_items = news_items[:8]

    note = "general_market_news_fallback" if rss_general_fallback else "coin_specific_news"
    return {
        "news": news_items,
        "news_status": "ok",
        "news_sources_used": sources_used,
        "news_note": note,
    }


def research_coin(symbol_or_name: str) -> dict[str, Any]:
    """Run the full deep-dive research pipeline for one coin."""

    identity = resolve_coin_identity(symbol_or_name)
    if not identity.binance_symbol and not identity.coingecko_id:
        raise ValueError(f"Could not resolve {symbol_or_name!r} to a supported market identity")

    technicals: dict[str, dict[str, Any]] = {}
    technical_fetch_errors: list[dict[str, str]] = []
    for interval in ("1m", "15m", "1h", "4h", "1d"):
        if identity.binance_symbol:
            try:
                klines, freshness = fetch_klines_resilient(identity.binance_symbol, interval, limit=120)
            except Exception as exc:
                details = _exception_details(exc)
                technical_fetch_errors.append(
                    {
                        "symbol": identity.binance_symbol,
                        "timeframe": interval,
                        "error": str(exc),
                        "exception_type": str(details.get("exception_type")),
                        "status_code": str(details.get("status_code")),
                    }
                )
                technicals[interval] = {
                    "latest_close": None,
                    "sma20": None,
                    "ema9": None,
                    "ema21": None,
                    "rsi14": None,
                    "rsi_label": "unknown",
                    "support": None,
                    "resistance": None,
                    "support_levels": [],
                    "resistance_levels": [],
                    "trend": "unknown",
                    "trend_label": "unknown",
                    "wick_flags": [],
                    "wick_flag_rate_pct": None,
                    "wick_flag_baseline_pct": None,
                    "wick_risk_unusual": None,
                    "manipulation_count": None,
                    "fetch_failed": True,
                    "is_stale": False,
                    "stale_age_seconds": None,
                    "stale_badge": None,
                }
                continue
            technical_snapshot = _technical_snapshot(_klines_frame(klines))
            technical_snapshot["fetch_failed"] = False
            technical_snapshot["is_stale"] = bool(freshness.get("is_stale", False))
            technical_snapshot["stale_age_seconds"] = freshness.get("stale_age_seconds")
            technical_snapshot["stale_badge"] = freshness.get("stale_badge")
            technicals[interval] = technical_snapshot
        else:
            technicals[interval] = {
                "latest_close": None,
                "sma20": None,
                "ema9": None,
                "ema21": None,
                "rsi14": None,
                "rsi_label": "unknown",
                "support": None,
                "resistance": None,
                "trend": "unknown",
                "trend_label": "unknown",
                "wick_flags": [],
                "wick_flag_rate_pct": None,
                "wick_flag_baseline_pct": None,
                "wick_risk_unusual": None,
                "manipulation_count": None,
                "fetch_failed": True,
                "is_stale": False,
                "stale_age_seconds": None,
                "stale_badge": None,
            }

    all_level_candidates: list[float] = []
    for interval_data in technicals.values():
        if not isinstance(interval_data, dict):
            continue
        for item in interval_data.get("support_levels", []) or []:
            if isinstance(item, dict) and item.get("level") is not None:
                all_level_candidates.append(_to_float(item.get("level")))
        for item in interval_data.get("resistance_levels", []) or []:
            if isinstance(item, dict) and item.get("level") is not None:
                all_level_candidates.append(_to_float(item.get("level")))
    if not all_level_candidates:
        for interval in technicals:
            if not isinstance(technicals[interval], dict):
                continue
            raw_support = technicals[interval].get("support")
            raw_resistance = technicals[interval].get("resistance")
            if raw_support is not None:
                all_level_candidates.append(_to_float(raw_support))
            if raw_resistance is not None:
                all_level_candidates.append(_to_float(raw_resistance))

    close_marker = None
    for preferred_tf in ("15m", "1h", "4h", "1d", "1m"):
        tf = technicals.get(preferred_tf)
        if isinstance(tf, dict) and tf.get("latest_close") is not None:
            close_marker = _to_float(tf.get("latest_close"))
            break

    clustered_levels = _cluster_levels([v for v in all_level_candidates if v > 0], tolerance_pct=0.004, keep_top=6)
    if close_marker and close_marker > 0:
        support_levels, resistance_levels = _split_levels_by_price(clustered_levels, close_marker)
    else:
        support_levels, resistance_levels = [], []

    top_support = support_levels[0]["level"] if support_levels else None
    top_resistance = resistance_levels[0]["level"] if resistance_levels else None
    if close_marker and close_marker > 0:
        top_support, top_resistance = _validate_support_resistance(
            close_price=close_marker,
            support=top_support,
            resistance=top_resistance,
            label=f"clustered:{identity.query}",
        )
        if top_support is None:
            support_levels = []
        if top_resistance is None:
            resistance_levels = []

    support_levels = support_levels[:3]
    resistance_levels = resistance_levels[:3]

    fundamentals_error = None
    try:
        fundamentals = get_coin_fundamentals(symbol_or_name)
    except Exception as exc:
        fundamentals_error = str(exc)
        _log_structured(
            level=logging.WARNING,
            component="fundamentals",
            outcome="provider_failed",
            symbol=identity.binance_symbol,
            extra=_exception_details(exc),
        )
        try:
            fundamentals = _minimal_fundamentals_from_binance(symbol_or_name)
            fundamentals["fundamentals_note"] = "fallback:binance_ticker_minimal"
            fundamentals["fundamentals_provider_error"] = fundamentals_error
        except Exception as fallback_exc:
            _log_structured(
                level=logging.ERROR,
                component="fundamentals",
                outcome="fallback_failed",
                symbol=identity.binance_symbol,
                extra=_exception_details(fallback_exc),
            )
            raise
    if "fundamentals_degraded" not in fundamentals:
        fundamentals["fundamentals_degraded"] = False
    coin_name = fundamentals.get("name") or identity.coingecko_name or identity.query
    coin_symbol = fundamentals.get("symbol") or identity.coingecko_symbol or identity.query
    news_payload = _select_news_items(str(coin_name), str(coin_symbol) if coin_symbol else None)
    news = news_payload.get("news", [])
    news_status = news_payload.get("news_status", "unavailable")

    depth_snapshot: dict[str, Any] = {}
    if identity.binance_symbol:
        try:
            depth_snapshot = _get_symbol_depth_snapshot(identity.binance_symbol, limit=50)
        except Exception as exc:
            logger.warning("Depth fetch failed for deep-dive %s: %s", identity.binance_symbol, exc)
            depth_snapshot = {
                "depth_notional_0_5pct": 0.0,
                "bid_depth_notional_0_5pct": 0.0,
                "ask_depth_notional_0_5pct": 0.0,
                "depth_source": "depth_unavailable",
            }

    research = {
        "input": symbol_or_name,
        "resolved": {
            "binance_symbol": identity.binance_symbol,
            "coingecko_id": identity.coingecko_id,
            "coingecko_name": identity.coingecko_name,
            "coingecko_symbol": identity.coingecko_symbol,
        },
        "technicals": technicals,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "fundamentals": fundamentals,
        "news": news,
        "news_status": news_status,
        "news_status_detail": "OK" if news_status == "ok" else "NO NEWS DATA RETRIEVED",
        "news_sources_used": news_payload.get("news_sources_used", []),
        "news_note": news_payload.get("news_note"),
        "depth": depth_snapshot,
        "technical_fetch_errors": technical_fetch_errors,
        "fundamentals_error": fundamentals_error,
    }
    return research


def format_scanner_context(candidates: list[dict[str, Any]]) -> str:
    """Convert scanner results into a compact Markdown context block."""

    if not candidates:
        return "No scanner candidates were found."
    frame = pd.DataFrame(candidates)
    columns = [
        "rank",
        "symbol",
        "score",
        "last_price",
        "quote_volume",
        "avg_daily_quote_volume_7d",
        "spike_ratio",
        "volatility_pct",
        "recent_15m_range_pct",
        "recent_direction_label",
        "spread_pct",
        "reason",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[columns]
    try:
        table = frame.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows: list[str] = []
        for row in frame.itertuples(index=False):
            row_values = [str(getattr(row, col)) for col in columns]
            rows.append("| " + " | ".join(row_values) + " |")
        table = "\n".join([header, separator, *rows])
    return "# Scanner Results\n\n" + table


def format_deepdive_context(research: dict[str, Any]) -> str:
    """Convert deep-dive research into Markdown for an LLM prompt."""

    lines = [
        f"# Deep Dive: {research.get('input')}",
        "",
        "## Resolved Identity",
        f"- Binance Symbol: {research.get('resolved', {}).get('binance_symbol')}",
        f"- CoinGecko ID: {research.get('resolved', {}).get('coingecko_id')}",
        f"- CoinGecko Name: {research.get('resolved', {}).get('coingecko_name')}",
        f"- CoinGecko Symbol: {research.get('resolved', {}).get('coingecko_symbol')}",
        "",
        "## Technical Summary",
    ]
    for interval, data in research.get("technicals", {}).items():
        lines.append(
            f"- {interval}: close={data.get('latest_close')}, SMA20={data.get('sma20')}, EMA9={data.get('ema9')}, EMA21={data.get('ema21')}, RSI14={data.get('rsi14')}, RSI_label={data.get('rsi_label')}, support={data.get('support')}, resistance={data.get('resistance')}, trend_label={data.get('trend_label')}, wick_flag_rate_pct={data.get('wick_flag_rate_pct')}, wick_flag_baseline_pct={data.get('wick_flag_baseline_pct')}, wick_risk_unusual={data.get('wick_risk_unusual')}"
        )
    lines.extend(
        [
            "",
            "## Support / Resistance",
            f"- Supports (clustered pivots): {research.get('support_levels', [])}",
            f"- Resistances (clustered pivots): {research.get('resistance_levels', [])}",
            "",
            "## Fundamentals",
        ]
    )
    fundamentals = research.get("fundamentals", {})
    for key in ("market_cap", "circulating_supply", "total_supply", "max_supply", "ath", "atl", "categories", "homepage", "blockchain_site", "community_score", "developer_score", "coingecko_rank", "price_change_24h_pct"):
        lines.append(f"- {key}: {fundamentals.get(key)}")
    lines.extend(
        [
            "",
            "## Orderbook Depth",
            f"- depth_notional_0_5pct: {research.get('depth', {}).get('depth_notional_0_5pct')}",
            f"- bid_depth_notional_0_5pct: {research.get('depth', {}).get('bid_depth_notional_0_5pct')}",
            f"- ask_depth_notional_0_5pct: {research.get('depth', {}).get('ask_depth_notional_0_5pct')}",
            f"- depth_source: {research.get('depth', {}).get('depth_source')}",
            "",
            "## News Status",
            f"- {research.get('news_status_detail', 'NO NEWS DATA RETRIEVED')}",
            f"- sources_used: {research.get('news_sources_used', [])}",
            f"- note: {research.get('news_note')}",
        ]
    )
    lines.extend(["", "## News"])
    if research.get("news_status") == "unavailable" or not research.get("news"):
        lines.append("- NO NEWS DATA RETRIEVED")
    else:
        for item in research.get("news", []):
            lines.append(
                f"- {item.get('published_time')} | {item.get('source')} | {item.get('title')} | {item.get('url')}"
            )
    return "\n".join(lines)


def run_scanner_pipeline(top_n: int = SCAN_TOP_N, *, scope: str = SCAN_SCOPE_ALL) -> dict[str, Any]:
    """Run scanner retrieval and summary with partial-failure handling."""

    candidates, scan_stats = find_top_scalping_candidates(top_n=top_n, scope=scope, return_stats=True)
    context = format_scanner_context(candidates)
    symbols = [str(item.get("symbol", "")) for item in candidates]
    roundtrip_ok = all(symbol and symbol in context for symbol in symbols)
    summary = ""
    error = None
    try:
        summary = summarize_scanner(context, scope=scope)
    except Exception as exc:
        err = str(exc)
        if "no provider configured" in err.lower():
            error = "LLM summary unavailable - no provider configured"
            summary = "LLM summary unavailable - no provider configured"
        else:
            error = f"scanner summary unavailable: {exc}"
        logger.warning(error)
    return {
        "candidates": candidates,
        "summary": summary,
        "scoring_formula": SCORING_FORMULA,
        "scan_stats": scan_stats,
        "context_row_roundtrip_ok": bool(roundtrip_ok),
        "message": (
            f"{(scan_stats.get('counts') or {}).get('entered_scoring_pool', len(candidates))} candidates passed filters "
            f"out of {(scan_stats.get('counts') or {}).get('eligible_tickers_after_basic_filter', 0)} scanned "
            f"(requested top {scan_stats.get('requested_top_n', top_n)}, scope={scan_stats.get('scope')})."
        ),
        "scope": scope,
        "error": error,
    }


def run_deep_dive_pipeline(symbol_or_name: str) -> dict[str, Any]:
    """Run deep-dive research and report generation with validation gates."""

    research = research_coin(symbol_or_name)
    ready, reason, diagnostics = validate_technical_readiness(research)
    if not ready:
        return {
            "research": research,
            "summary": "",
            "error": reason,
            "technical_gate": diagnostics,
        }
    context = format_deepdive_context(research)
    summary = ""
    summary_error = None
    try:
        summary = summarize_deepdive(context, research=research)
    except Exception as exc:
        err = str(exc)
        if "no provider configured" in err.lower():
            summary_error = "LLM summary unavailable - no provider configured"
            summary = "LLM summary unavailable - no provider configured"
        else:
            summary_error = f"deep-dive summary unavailable: {exc}"
        logger.warning(summary_error)
    return {
        "research": research,
        "summary": summary,
        "error": summary_error,
        "technical_gate": diagnostics,
    }


def _extract_context_rsi_labels(context: str) -> list[str]:
    """Extract deterministic RSI labels from deep-dive context."""

    labels = re.findall(r"RSI_label=([a-z_]+)", context)
    return [label.lower() for label in labels]


def _rsi_word_mismatch(report: str, context: str) -> bool:
    """Check report claims against deterministic RSI labels in context."""

    report_lower = report.lower()
    labels = _extract_context_rsi_labels(context)
    if "oversold" in report_lower and "oversold" not in labels:
        return True
    if "overbought" in report_lower and "overbought" not in labels:
        return True
    return False


def _missing_no_news_phrase(report: str, context: str) -> bool:
    """Ensure explicit no-news phrasing when the context has no news data."""

    if "NO NEWS DATA RETRIEVED" not in context:
        return False
    return "no news data available" not in report.lower()


def summarize_scanner(context: str, *, scope: str = SCAN_SCOPE_ALL) -> str:
    """Summarize scanner data with the LLM fallback engine."""

    meme_clause = ""
    if scope == SCAN_SCOPE_MEME:
        meme_clause = (
            " In meme/high-volatility mode, prioritize spike_ratio as the primary signal. "
            "If context says a coin's volume is Xx its 7-day average, explain whether that implies momentum is still building or extended, "
            "using recent_direction_label only. Explicitly call out wick-risk and thin-depth risk for manipulation-prone assets."
        )
    system_prompt = (
        "You are a crypto market research assistant. Given this scanner data, explain in plain language why each coin is or is not currently favorable for scalping, citing specific numbers from context only. Low volatility + high liquidity is good for market-making, not scalping. A scalp candidate needs BOTH tight spread/depth AND sufficient volatility to create capturable price movement. Do not recommend near-zero-volatility assets regardless of how tight their spread is. Do not invent data not in context. End with a one-line caveat that volatility cuts both ways."
        + meme_clause
    )
    return fetch_llm_with_fallback(system_prompt, context)


def summarize_deepdive(context: str, research: dict[str, Any] | None = None) -> str:
    """Summarize deep-dive research with the LLM fallback engine."""

    if research is not None:
        is_ready, reason, diagnostics = validate_technical_readiness(research)
        if not is_ready:
            logger.error("Deep-dive LLM gated: %s diagnostics=%s", reason, diagnostics)
            raise RuntimeError(reason or "technical data unavailable")

    system_prompt = (
        "You are a crypto research assistant producing a decision-support report, not a signal. Given this technical + fundamental + news context for one coin, produce: (1) Technical Read per timeframe, (2) Key support/resistance levels, (3) Fundamental Snapshot, (4) News Read — anything bullish/bearish in the last 48h, (5) Risk Flags (wick manipulation, thin liquidity, bad news), (6) a plain-language summary of what would need to be true for a long vs short case. Do not output a single directive BUY/SELL — lay out both cases with the evidence for each so the user decides. RSI and trend labels in context are deterministic ground truth; do not reinterpret thresholds. If context contains NO NEWS DATA RETRIEVED, explicitly state 'no news data available' and do not speculate on sentiment or recent events."
    )
    report = fetch_llm_with_fallback(system_prompt, context)
    if _rsi_word_mismatch(report, context):
        correction_prompt = (
            context
            + "\n\nValidation note: your previous response used oversold/overbought terms that conflict with deterministic RSI labels. Regenerate with strict adherence to RSI_label values."
        )
        regenerated = fetch_llm_with_fallback(system_prompt, correction_prompt)
        if _rsi_word_mismatch(regenerated, context):
            return (
                regenerated
                + "\n\n[Validation Flag] RSI language mismatch detected against deterministic RSI labels."
            )
        return regenerated
    if _missing_no_news_phrase(report, context):
        correction_prompt = (
            context
            + "\n\nValidation note: context contains NO NEWS DATA RETRIEVED. Regenerate and explicitly include the exact phrase: no news data available. Do not infer sentiment or recent events."
        )
        regenerated = fetch_llm_with_fallback(system_prompt, correction_prompt)
        if _missing_no_news_phrase(regenerated, context):
            return regenerated + "\n\n[Validation Flag] No-news phrase missing despite NO NEWS DATA RETRIEVED context."
        return regenerated
    return report


def _print_scanner_results(candidates: list[dict[str, Any]], summary: str, message: str | None = None) -> None:
    """Render scanner output to the terminal."""

    if candidates:
        frame = pd.DataFrame(candidates)
        print("\n=== TOP SCALPING CANDIDATES ===")
        if message:
            print(message)
        print(frame[["rank", "symbol", "score", "last_price", "quote_volume", "volatility_pct", "recent_15m_range_pct", "spread_pct", "spread_source", "depth_notional_0_5pct", "depth_source", "reason"]].to_string(index=False))
        print("\n=== SCORING FORMULA ===")
        print(SCORING_FORMULA)
        print("thin_liquidity_penalty applies when volatility is high and near-mid orderbook depth is thin.")
    else:
        print("No scanner candidates found.")
    print("\n=== LLM SUMMARY ===")
    print(summary)


def _print_deepdive_results(research: dict[str, Any], summary: str) -> None:
    """Render deep-dive output to the terminal."""

    print("\n=== TECHNICAL SUMMARY ===")
    for interval, data in research.get("technicals", {}).items():
        print(
            f"{interval}: close={data.get('latest_close')} | SMA20={data.get('sma20')} | EMA9={data.get('ema9')} | EMA21={data.get('ema21')} | RSI14={data.get('rsi14')} ({data.get('rsi_label')}) | support={data.get('support')} | resistance={data.get('resistance')} | trend={data.get('trend_label')} | wick_flag_rate_pct={data.get('wick_flag_rate_pct')} | wick_risk_unusual={data.get('wick_risk_unusual')}"
        )
    print("\n=== CLUSTERED SUPPORT/RESISTANCE ===")
    print(f"Supports: {research.get('support_levels', [])}")
    print(f"Resistances: {research.get('resistance_levels', [])}")
    print("\n=== FUNDAMENTALS ===")
    print(json.dumps(research.get("fundamentals", {}), indent=2, ensure_ascii=False, default=str))
    print("\n=== ORDERBOOK DEPTH ===")
    print(json.dumps(research.get("depth", {}), indent=2, ensure_ascii=False, default=str))
    print("\n=== NEWS STATUS ===")
    print(research.get("news_status", "NO NEWS DATA RETRIEVED"))
    print(f"Sources used: {research.get('news_sources_used', [])}")
    if research.get("news_note"):
        print(f"News note: {research.get('news_note')}")
    print("\n=== NEWS ===")
    for item in research.get("news", []):
        print(f"- {item.get('published_time')} | {item.get('source')} | {item.get('title')} | {item.get('url')}")
    print("\n=== LLM SUMMARY ===")
    print(summary)


def run_scanner_mode() -> None:
    """Execute scanner mode end to end."""

    payload = run_scanner_pipeline(top_n=SCAN_TOP_N)
    candidates = payload.get("candidates", [])
    summary = payload.get("summary", "")
    _print_scanner_results(candidates, summary, message=str(payload.get("message", "")))
    append_jsonl({"mode": "scanner", "input": None, "output": payload})


def run_deep_dive_mode() -> None:
    """Prompt the user for a coin and print a full research report."""

    symbol_or_name = input("Enter a coin ticker or name: ").strip()
    payload = run_deep_dive_pipeline(symbol_or_name)
    research = payload.get("research", {})
    summary = payload.get("summary", "")
    if payload.get("error"):
        raise RuntimeError(str(payload.get("error")))
    _print_deepdive_results(research, summary)
    append_jsonl({"mode": "deep_dive", "input": symbol_or_name, "output": payload})


def main() -> None:
    """Run the interactive CLI loop."""

    while True:
        print("\n1) Scan top scalping candidates")
        print("2) Deep-dive a specific coin")
        print("q) Quit")
        choice = input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            break
        if choice == "1":
            try:
                run_scanner_mode()
            except Exception as exc:
                logger.exception("Scanner mode failed")
                print(f"Scanner mode failed: {exc}")
        elif choice == "2":
            try:
                run_deep_dive_mode()
            except Exception as exc:
                logger.exception("Deep-dive mode failed")
                print(f"Deep-dive mode failed: {exc}")
        else:
            print("Invalid choice. Please enter 1, 2, or q.")


if __name__ == "__main__":
    main()