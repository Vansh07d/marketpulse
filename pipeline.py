"""
MarketPulse - End-to-End Pipeline Test

Flow:

    Fresh News
        ↓
    FinBERT
        ↓
    Market Data
        ↓
    Signal Engine
        ↓
    BUY / HOLD / SELL

This script intentionally does NOT use Redis yet.
It validates that all three core services work together
with real data before introducing background processing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from news_service import fetch_company_news
from sentiment_service import get_finbert
from market_service import fetch_market_data
from signal_service import generate_signal


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SYMBOL = "INFY"

# Number of fresh articles to send to FinBERT.
MAX_ARTICLES_FOR_SENTIMENT = 10


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("marketpulse.pipeline")


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_article(
    index: int,
    article: Dict[str, Any],
) -> None:

    print(
        f"\n[{index}] "
        f"{article.get('title', 'Untitled')}"
    )

    print(
        f"Provider   : "
        f"{article.get('provider')}"
    )

    print(
        f"Source     : "
        f"{article.get('source')}"
    )

    print(
        f"Published  : "
        f"{article.get('published_at')}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print_section(
        "MARKETPULSE END-TO-END PIPELINE TEST"
    )

    print(
        f"Testing symbol: {TEST_SYMBOL}"
    )

    # ========================================================
    # STEP 1 — NEWS
    # ========================================================

    print_section(
        "STEP 1 — FETCHING FRESH NEWS"
    )

    try:

        articles = fetch_company_news(
            TEST_SYMBOL
        )

    except Exception as exc:

        logger.exception(
            "News pipeline failed: %s",
            exc,
        )

        return

    if not articles:

        print(
            "No fresh articles found."
        )

        print(
            "\nCannot continue sentiment test "
            "without news."
        )

        return

    # Limit the number of headlines sent
    # to FinBERT.

    articles = articles[
        :MAX_ARTICLES_FOR_SENTIMENT
    ]

    print(
        f"Fresh articles found: "
        f"{len(articles)}"
    )

    for index, article in enumerate(
        articles,
        start=1,
    ):
        print_article(
            index,
            article,
        )

    # ========================================================
    # STEP 2 — FINBERT
    # ========================================================

    print_section(
        "STEP 2 — RUNNING FINBERT"
    )

    headlines = [
        article.get("title", "").strip()
        for article in articles
        if article.get("title")
    ]

    if not headlines:

        print(
            "No valid headlines available."
        )

        return

    print(
        f"Running FinBERT on "
        f"{len(headlines)} headlines..."
    )

    try:

        finbert = get_finbert()

        sentiment_results = (
            finbert.predict_batch(
                headlines
            )
        )

    except Exception as exc:

        logger.exception(
            "FinBERT inference failed: %s",
            exc,
        )

        return

    # --------------------------------------------------------
    # Display individual sentiment
    # --------------------------------------------------------

    for index, result in enumerate(
        sentiment_results,
        start=1,
    ):

        print(
            f"\n[{index}] "
            f"{result.get('label', 'unknown').upper()}"
        )

        print(
            f"Score      : "
            f"{result.get('score', 0):+.4f}"
        )

        print(
            f"Positive   : "
            f"{result.get('positive', 0):.4f}"
        )

        print(
            f"Neutral    : "
            f"{result.get('neutral', 0):.4f}"
        )

        print(
            f"Negative   : "
            f"{result.get('negative', 0):.4f}"
        )

    # ========================================================
    # STEP 3 — MARKET DATA
    # ========================================================

    print_section(
        "STEP 3 — FETCHING MARKET DATA"
    )

    try:

        market_data = fetch_market_data(
            TEST_SYMBOL
        )

    except Exception as exc:

        logger.exception(
            "Market data pipeline failed: %s",
            exc,
        )

        return

    if not market_data:

        print(
            "No market data available."
        )

        return

    print(
        f"Provider          : "
        f"{market_data.get('provider')}"
    )

    print(
        f"Price             : "
        f"{market_data.get('price')}"
    )

    print(
        f"Previous Close    : "
        f"{market_data.get('previous_close')}"
    )

    print(
        f"Change            : "
        f"{market_data.get('change_percent')}%"
    )

    print(
        f"Volume            : "
        f"{market_data.get('volume')}"
    )

    # ========================================================
    # STEP 4 — SIGNAL
    # ========================================================

    print_section(
        "STEP 4 — GENERATING MARKETPULSE SIGNAL"
    )

    try:

        signal = generate_signal(
            symbol=TEST_SYMBOL,
            sentiment_results=sentiment_results,
            market_data=market_data,
            benchmark_data=None,
        )

    except Exception as exc:

        logger.exception(
            "Signal generation failed: %s",
            exc,
        )

        return

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print_section(
        "FINAL MARKETPULSE RESULT"
    )

    print(
        f"Symbol              : "
        f"{signal.get('symbol')}"
    )

    print(
        f"Signal              : "
        f"{signal.get('signal')}"
    )

    print(
        f"Signal Score        : "
        f"{signal.get('signal_score'):+.4f}"
    )

    print()

    print(
        f"News Sentiment      : "
        f"{signal.get('news_sentiment'):+.4f}"
    )

    print(
        f"Price Momentum      : "
        f"{signal.get('price_momentum'):+.4f}"
    )

    print(
        f"Market Momentum     : "
        f"{signal.get('market_momentum'):+.4f}"
    )

    print()

    print(
        f"News Component      : "
        f"{signal.get('news_component'):+.4f}"
    )

    print(
        f"Price Component     : "
        f"{signal.get('price_component'):+.4f}"
    )

    print(
        f"Market Component    : "
        f"{signal.get('market_component'):+.4f}"
    )

    print()

    print(
        f"Current Price       : "
        f"{signal.get('price')}"
    )

    print(
        f"Price Change        : "
        f"{signal.get('change_percent')}%"
    )

    print(
        f"Timestamp           : "
        f"{signal.get('timestamp')}"
    )

    print()
    print("=" * 70)

    print(
        "END-TO-END PIPELINE TEST COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()