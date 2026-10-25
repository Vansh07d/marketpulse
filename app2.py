"""MarketPulse dashboard - reads signals, history, news and latency from Redis (written by realtime_worker.py)."""

import os
from datetime import datetime

import pandas as pd
import streamlit as st

import latency
from redis_client import get_redis

st.set_page_config(page_title="MarketPulse", page_icon="📈", layout="wide")

DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "INFY").strip().upper()
REFRESH_INTERVAL_SECONDS = max(3, int(os.getenv("REFRESH_INTERVAL_SECONDS", "10")))
INDIAN_SYMBOLS = {"INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "TATAMOTORS"}
HISTORY_POINTS = 300
ICONS = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}


@st.cache_resource
def get_redis_client():
    return get_redis()


redis_client = get_redis_client()


def format_timestamp(value) -> str:
    ts = latency.parse_ts(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else ("Unknown" if not value else str(value))


def money(value, currency: str) -> str:
    return f"{currency}{float(value):,.2f}" if value is not None else "N/A"


def history_frame(symbol: str) -> pd.DataFrame:
    rows = []
    for item in reversed(redis_client.get_signal_history(symbol, HISTORY_POINTS)):  # stored newest-first
        ts = latency.parse_ts(item.get("updated_at") or item.get("timestamp"))
        if ts:
            rows.append({
                "time": ts,
                "Signal score": item.get("signal_score"),
                "News sentiment": item.get("news_sentiment"),
                "Price momentum": item.get("price_momentum"),
                "Market momentum": item.get("market_momentum"),
                "Price": item.get("price"),
            })
    return pd.DataFrame(rows).set_index("time") if rows else pd.DataFrame()


def render_header_and_metrics(symbol: str, signal_data: dict, currency: str) -> None:
    label = str(signal_data.get("signal", "HOLD")).upper()
    st.markdown(
        f'<div style="padding:20px;border-radius:12px;border:1px solid #444;margin-bottom:20px">'
        f"<h2>{symbol}</h2><h1>{ICONS.get(label, '🟡')} {label}</h1><p>Market Signal</p></div>",
        unsafe_allow_html=True,
    )
    metrics = [("Signal Score", "signal_score"), ("News Sentiment", "news_sentiment"),
               ("Price Momentum", "price_momentum"), ("Market Momentum", "market_momentum")]
    for col, (name, key) in zip(st.columns(4), metrics):
        col.metric(name, f"{float(signal_data.get(key, 0.0)):+.3f}")

    st.subheader("📊 Market Data")
    market = signal_data.get("market_data") or {}
    pct = market.get("change_percent")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", money(market.get("price"), currency))
    c2.metric("Change", f"{float(pct):+.2f}%" if pct is not None else "N/A")
    c3.metric("Previous Close", money(market.get("previous_close"), currency))
    c4.metric("Data Provider", str(market.get("provider", "Unknown")))


def render_history(symbol: str) -> None:
    st.subheader("📈 Signal History")
    df = history_frame(symbol)
    if len(df) < 2:
        st.info("History builds up as the worker completes more cycles.")
        return
    left, right = st.columns(2)
    with left:
        st.caption("Signal score and components (-1 to +1)")
        st.line_chart(df[["Signal score", "News sentiment", "Price momentum", "Market momentum"]])
    with right:
        st.caption("Price")
        st.line_chart(df[["Price"]].dropna())
    st.caption(f"{len(df)} snapshots, newest {format_timestamp(df.index[-1].isoformat())}")


def render_latency(symbol: str) -> None:
    st.subheader("⏱️ Pipeline Latency")
    records = latency.get_latencies(redis_client, symbol)
    if not records:
        st.info("Latency is recorded when new articles are processed.")
        return
    summary = latency.summarize(records)
    names = {"end_to_end_s": "End-to-end", "ingest_lag_s": "Publish → ingest",
             "inference_s": "FinBERT", "signal_s": "Scored → signal"}
    for col, key in zip(st.columns(4), names):
        col.metric(f"{names[key]} p50", latency.format_seconds(summary[key]["p50"]))
        col.caption(f"p95: {latency.format_seconds(summary[key]['p95'])} · n={summary[key]['count']}")
    st.caption("End-to-end includes the provider's own delay and our poll interval.")


def render_news(symbol: str) -> None:
    st.subheader("📰 Latest News")
    news = redis_client.get_news(symbol) or []
    if not news:
        st.info("No news available yet.")
        return
    for article in news[:10]:
        sentiment = article.get("sentiment") or {}
        st.markdown(f"### {article.get('title', 'Untitled')}")
        a, b, c = st.columns(3)
        a.write(f"**Sentiment:** {str(sentiment.get('label', 'unknown')).upper()}")
        b.write(f"**Score:** {float(sentiment.get('score', 0.0)):+.3f}")
        c.write(f"**Source:** {article.get('source', 'Unknown')}")
        if article.get("published_at"):
            st.caption(f"Published: {format_timestamp(article['published_at'])}")
        if article.get("url"):
            st.markdown(f"[Read article]({article['url']})")
        st.divider()


@st.fragment(run_every=REFRESH_INTERVAL_SECONDS)
def dashboard(symbol: str) -> None:
    signal_data = redis_client.get_signal(symbol)
    if not signal_data:
        st.warning(f"No signal available for {symbol} yet.")
        st.info("Make sure realtime_worker.py is running and has completed one successful cycle. "
                "This page keeps checking automatically.")
        return
    render_header_and_metrics(symbol, signal_data, "₹" if symbol in INDIAN_SYMBOLS else "$")
    render_history(symbol)
    render_latency(symbol)
    render_news(symbol)
    st.caption(f"Signal updated: {format_timestamp(signal_data.get('updated_at') or signal_data.get('timestamp'))}")


st.title("📈 MarketPulse")
st.caption("Real-time financial sentiment and market signal dashboard")

st.sidebar.header("⚙️ Dashboard")
symbol = st.sidebar.text_input("Stock Symbol", value=DEFAULT_SYMBOL).strip().upper() or DEFAULT_SYMBOL
st.sidebar.button("🔄 Refresh Now")   # any click reruns the script
st.sidebar.markdown("---")
st.sidebar.caption(f"Auto-refresh: every {REFRESH_INTERVAL_SECONDS} seconds")

if not redis_client.health_check():
    st.error("❌ Redis is not available. Make sure Redis is running.")
    st.stop()

dashboard(symbol)