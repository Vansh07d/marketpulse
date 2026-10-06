import streamlit as st
from datetime import datetime, timezone
import time

from redis_client import get_redis


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SYMBOL = "INFY"

REFRESH_INTERVAL_SECONDS = 10


# ============================================================
# REDIS
# ============================================================

@st.cache_resource
def get_redis_client():
    return get_redis()


redis_client = get_redis_client()


# ============================================================
# HELPERS
# ============================================================

def format_timestamp(timestamp):
    """Format ISO timestamp for display."""

    if not timestamp:
        return "Unknown"

    try:
        dt = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    except Exception:
        return str(timestamp)


def get_signal(symbol):
    """Get latest signal from Redis."""

    try:
        return redis_client.get_signal(symbol)
    except Exception as exc:
        st.error(f"Redis signal error: {exc}")
        return None


def get_news(symbol):
    """Get latest news from Redis."""

    try:
        return redis_client.get_news(symbol) or []
    except Exception as exc:
        st.error(f"Redis news error: {exc}")
        return []


def signal_color(signal):
    """Return display color for signal."""

    signal = str(signal).upper()

    if signal == "BUY":
        return "green"

    if signal == "SELL":
        return "red"

    return "orange"


def signal_emoji(signal):
    """Return emoji for signal."""

    signal = str(signal).upper()

    if signal == "BUY":
        return "🟢"

    if signal == "SELL":
        return "🔴"

    return "🟡"


# ============================================================
# HEADER
# ============================================================

st.title("📈 MarketPulse")

st.caption(
    "Real-time financial sentiment and market signal dashboard"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard")

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value=DEFAULT_SYMBOL,
).strip().upper()

if not symbol:
    symbol = DEFAULT_SYMBOL


if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()


st.sidebar.markdown("---")

st.sidebar.caption(
    f"Auto-refresh: every {REFRESH_INTERVAL_SECONDS} seconds"
)


# ============================================================
# REDIS HEALTH
# ============================================================

if not redis_client.health_check():

    st.error(
        "❌ Redis is not available. "
        "Make sure Redis is running."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

signal_data = get_signal(symbol)
news_data = get_news(symbol)


# ============================================================
# NO SIGNAL YET
# ============================================================

if not signal_data:

    st.warning(
        f"No signal available for {symbol} yet."
    )

    st.info(
        "Make sure realtime_worker.py is running "
        "and has completed at least one successful cycle."
    )

    st.stop()


# ============================================================
# SIGNAL DATA
# ============================================================

signal = signal_data.get(
    "signal",
    "HOLD",
)

signal_score = float(
    signal_data.get(
        "signal_score",
        0.0,
    )
)

news_sentiment = float(
    signal_data.get(
        "news_sentiment",
        0.0,
    )
)

price_momentum = float(
    signal_data.get(
        "price_momentum",
        0.0,
    )
)

market_momentum = float(
    signal_data.get(
        "market_momentum",
        0.0,
    )
)

market_data = signal_data.get(
    "market_data",
    {},
)


# ============================================================
# MAIN SIGNAL CARD
# ============================================================

emoji = signal_emoji(signal)

st.markdown(
    f"""
    <div style="
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 20px;
    ">
        <h2>{symbol}</h2>
        <h1>{emoji} {signal}</h1>
        <p>Market Signal</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Signal Score",
        f"{signal_score:+.3f}",
    )


with col2:
    st.metric(
        "News Sentiment",
        f"{news_sentiment:+.3f}",
    )


with col3:
    st.metric(
        "Price Momentum",
        f"{price_momentum:+.3f}",
    )


with col4:
    st.metric(
        "Market Momentum",
        f"{market_momentum:+.3f}",
    )


# ============================================================
# MARKET DATA
# ============================================================

st.subheader("📊 Market Data")

market_col1, market_col2, market_col3, market_col4 = st.columns(4)


price = market_data.get("price")

previous_close = market_data.get(
    "previous_close"
)

change_percent = market_data.get(
    "change_percent"
)

provider = market_data.get(
    "provider",
    "Unknown",
)


with market_col1:

    if price is not None:
        st.metric(
            "Current Price",
            f"₹{float(price):,.2f}",
        )
    else:
        st.metric(
            "Current Price",
            "N/A",
        )


with market_col2:

    if change_percent is not None:
        st.metric(
            "Change",
            f"{float(change_percent):+.2f}%",
        )
    else:
        st.metric(
            "Change",
            "N/A",
        )


with market_col3:

    if previous_close is not None:
        st.metric(
            "Previous Close",
            f"₹{float(previous_close):,.2f}",
        )
    else:
        st.metric(
            "Previous Close",
            "N/A",
        )


with market_col4:

    st.metric(
        "Data Provider",
        str(provider),
    )


# ============================================================
# NEWS
# ============================================================

st.subheader("📰 Latest News")


if not news_data:

    st.info(
        "No news available yet."
    )

else:

    for article in news_data[:10]:

        title = article.get(
            "title",
            "Untitled",
        )

        source = article.get(
            "source",
            "Unknown",
        )

        sentiment = article.get(
            "sentiment",
            {},
        )

        label = sentiment.get(
            "label",
            "UNKNOWN",
        )

        score = sentiment.get(
            "score",
            0.0,
        )

        published_at = article.get(
            "published_at"
        )

        url = article.get(
            "url"
        )

        st.markdown(
            f"### {title}"
        )

        news_col1, news_col2, news_col3 = st.columns(3)

        with news_col1:
            st.write(
                f"**Sentiment:** {label}"
            )

        with news_col2:
            st.write(
                f"**Score:** {float(score):+.3f}"
            )

        with news_col3:
            st.write(
                f"**Source:** {source}"
            )

        if published_at:
            st.caption(
                f"Published: "
                f"{format_timestamp(published_at)}"
            )

        if url:
            st.markdown(
                f"[Read article]({url})"
            )

        st.divider()


# ============================================================
# LAST UPDATE
# ============================================================

updated_at = signal_data.get(
    "updated_at"
)

if updated_at:

    st.caption(
        f"Signal updated: "
        f"{format_timestamp(updated_at)}"
    )

else:

    st.caption(
        "Signal update time unavailable."
    )


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(
    REFRESH_INTERVAL_SECONDS
)

st.rerun()