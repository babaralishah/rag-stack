"""Standalone Streamlit dashboard for crypto market research.

This tool produces research summaries for human review. It does not place
trades and is not financial advice.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent import (
    SCAN_SCOPE_ALL,
    SCAN_SCOPE_MEME,
    SCAN_TOP_N,
    find_top_scalping_candidates,
    format_deepdive_context,
    format_scanner_context,
    research_coin,
    summarize_deepdive,
    summarize_scanner,
)


st.set_page_config(page_title="Crypto Scalping Research Dashboard", layout="wide")


st.markdown(
    """
    <div style="
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(12,18,39,0.96), rgba(24,47,89,0.92));
        border: 1px solid rgba(135, 170, 255, 0.22);
        box-shadow: 0 14px 45px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    ">
        <div style="font-size: 0.88rem; letter-spacing: 0.12em; text-transform: uppercase; color: #9ec3ff;">Research Dashboard</div>
        <div style="font-size: 2rem; font-weight: 800; color: white; margin-top: 0.2rem;">📊 Crypto Scalping Research Dashboard</div>
        <div style="font-size: 0.98rem; color: rgba(255,255,255,0.82); margin-top: 0.35rem;">Real-time scanning and multi-timeframe decision support powered by your free Groq/Gemini stack.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "This tool produces research summaries for human review. It does not place trades and is not financial advice.",
    icon="⚠️",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(18, 26, 45, 0.98), rgba(13, 18, 31, 0.98));
            border: 1px solid rgba(135, 170, 255, 0.16);
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid rgba(135, 170, 255, 0.16);
            border-radius: 16px;
            overflow: hidden;
        }
        .section-card {
            background: linear-gradient(180deg, rgba(18, 26, 45, 0.98), rgba(13, 18, 31, 0.98));
            border: 1px solid rgba(135, 170, 255, 0.16);
            border-radius: 18px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.22);
        }
        .small-label {
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #91a7ff;
            margin-bottom: 0.35rem;
        }
        .report-box {
            background: linear-gradient(180deg, rgba(15, 20, 34, 0.98), rgba(10, 13, 23, 0.98));
            border: 1px solid rgba(135, 170, 255, 0.16);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            white-space: pre-wrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

scanner_tab, deep_dive_tab = st.tabs(["Market Scanner", "Coin Deep-Dive"])


def _display_scanner_summary(candidates: list[dict[str, object]], scope: str) -> None:
    """Render scanner summary text in a readable card."""

    scanner_context = format_scanner_context(candidates)
    summary = summarize_scanner(scanner_context, scope=scope)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">LLM Scanner Analysis</div>', unsafe_allow_html=True)
    st.markdown(f"<div class=\"report-box\">{summary}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _scanner_table_frame(candidates: list[dict[str, object]]) -> pd.DataFrame:
    """Build scanner dataframe with mode-relevant columns for display."""

    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates).copy()
    preferred_columns = [
        "rank",
        "symbol",
        "score",
        "last_price",
        "today_quote_volume",
        "avg_daily_quote_volume_7d",
        "spike_ratio",
        "volatility_pct",
        "recent_direction_label",
        "spread_pct",
        "depth_notional_0_5pct",
    ]
    for col in preferred_columns:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[preferred_columns]
    frame.rename(
        columns={
            "rank": "Rank",
            "symbol": "Symbol",
            "score": "Score",
            "last_price": "Price",
            "today_quote_volume": "Today Vol",
            "avg_daily_quote_volume_7d": "7d Avg Vol",
            "spike_ratio": "Spike Ratio",
            "volatility_pct": "24h Range %",
            "recent_direction_label": "Direction",
            "spread_pct": "Spread %",
            "depth_notional_0_5pct": "Depth Notional",
        },
        inplace=True,
    )
    return frame


def _deep_dive_rows(research: dict[str, object]) -> pd.DataFrame:
    """Build a table of structural levels for the deep-dive view."""

    rows: list[dict[str, object]] = []
    technicals = research.get("technicals", {}) if isinstance(research, dict) else {}
    if isinstance(technicals, dict):
        for interval in ("1m", "15m", "1h", "4h", "1d"):
            snapshot = technicals.get(interval, {})
            if not isinstance(snapshot, dict):
                snapshot = {}
            rows.append(
                {
                    "Timeframe": interval,
                    "Close": snapshot.get("latest_close"),
                    "SMA20": snapshot.get("sma20"),
                    "EMA9": snapshot.get("ema9"),
                    "EMA21": snapshot.get("ema21"),
                    "RSI14": snapshot.get("rsi14"),
                    "Support": snapshot.get("support"),
                    "Resistance": snapshot.get("resistance"),
                    "Trend": snapshot.get("trend_label") if snapshot.get("trend_label") else "unknown",
                    "Wick Risk": (
                        "unknown"
                        if snapshot.get("manipulation_count") is None
                        else ("High" if snapshot.get("manipulation_count", 0) else "Low")
                    ),
                }
            )
    return pd.DataFrame(rows)


with scanner_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">Scanner Mode</div>', unsafe_allow_html=True)
    st.markdown("Scan the market for the strongest current scalping candidates using live exchange data.")
    st.markdown("</div>", unsafe_allow_html=True)

    control_left, control_right = st.columns([1, 1], gap="large")
    with control_left:
        selected_top_n = st.selectbox(
            "Top-N",
            options=[5, 10, 15, 20],
            index=[5, 10, 15, 20].index(SCAN_TOP_N) if SCAN_TOP_N in [5, 10, 15, 20] else 1,
            help="Controls how many ranked candidates are returned.",
        )
    with control_right:
        selected_scope_label = st.radio(
            "Asset Scope",
            options=["Meme / High-Volatility", "All Assets"],
            index=0,
            horizontal=True,
            help="Meme mode prioritizes volume-spike behavior for high-volatility candidates.",
        )
    selected_scope = SCAN_SCOPE_MEME if selected_scope_label == "Meme / High-Volatility" else SCAN_SCOPE_ALL

    run_scan = st.button("🚀 Run Real-Time Market Scan", type="primary", use_container_width=True)
    if run_scan:
        try:
            with st.spinner("Scanning the market and ranking the top scalping candidates..."):
                candidates = find_top_scalping_candidates(top_n=int(selected_top_n), scope=selected_scope)
        except Exception as exc:
            st.error(f"Scanner failed to load market data: {exc}")
            candidates = []
        if candidates:
            scanner_df = _scanner_table_frame(candidates)
            st.dataframe(scanner_df, use_container_width=True, hide_index=True)
            _display_scanner_summary(candidates, selected_scope)
        else:
            st.info("No scanner candidates were returned for the current market snapshot.")


with deep_dive_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">Deep-Dive Mode</div>', unsafe_allow_html=True)
    st.markdown("Review one asset across multiple timeframes, fundamentals, and news before deciding whether to act.")
    st.markdown("</div>", unsafe_allow_html=True)

    symbol = st.text_input("Coin symbol or name", value="PEPE")
    run_deep_dive = st.button("🔍 Run Comprehensive Asset Deep-Dive", type="primary", use_container_width=True)

    if run_deep_dive and symbol.strip():
        try:
            with st.spinner(f"Researching {symbol.strip().upper()} across multiple timeframes..."):
                research = research_coin(symbol.strip())
        except Exception as exc:
            st.error(f"Deep-dive failed for '{symbol.strip()}': {exc}")
            research = None

        if not isinstance(research, dict):
            st.stop()

        fundamentals = research.get("fundamentals", {}) if isinstance(research, dict) else {}
        technical_df = _deep_dive_rows(research)

        left_col, right_col = st.columns([1, 1.35], gap="large")

        with left_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="small-label">Asset Fundamentals</div>', unsafe_allow_html=True)
            st.metric("Market Cap", f"${fundamentals.get('market_cap', 0):,.0f}" if fundamentals.get("market_cap") else "N/A")
            st.metric(
                "Circulating Supply",
                f"{fundamentals.get('circulating_supply', 0):,.0f}" if fundamentals.get("circulating_supply") else "N/A",
            )
            st.metric("ATH", f"${fundamentals.get('ath', 0):,.8f}" if fundamentals.get("ath") else "N/A")
            st.metric("Total Supply", f"{fundamentals.get('total_supply', 0):,.0f}" if fundamentals.get("total_supply") else "N/A")
            st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="small-label">Structural Pricing</div>', unsafe_allow_html=True)
            st.dataframe(technical_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Live News Feed")
        news_items = research.get("news", []) if isinstance(research, dict) else []
        if news_items:
            for item in news_items:
                title = item.get("title", "Untitled") if isinstance(item, dict) else str(item)
                source = item.get("source", "Unknown") if isinstance(item, dict) else "Unknown"
                published = item.get("published_time", "") if isinstance(item, dict) else ""
                url = item.get("url", "") if isinstance(item, dict) else ""
                st.markdown(
                    f"- **{title}**  \n"
                    f"  Source: {source}  \n"
                    f"  Published: {published}  \n"
                    f"  [Read more]({url})"
                )
        else:
            st.info("No matching news items were found for this asset.")

        deep_context = format_deepdive_context(research)
        try:
            deep_report = summarize_deepdive(deep_context, research=research)
        except Exception as exc:
            deep_report = f"technical data unavailable, please retry. ({exc})"
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">LLM Decision Support Report</div>', unsafe_allow_html=True)
        st.markdown(f"<div class=\"report-box\">{deep_report}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
