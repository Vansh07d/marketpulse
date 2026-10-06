"""
MarketPulse - Signal Service

Combines:
    1. News sentiment from FinBERT
    2. Stock price momentum
    3. Market momentum

into a single BUY / HOLD / SELL signal.

Initial weighting:
    News sentiment  = 60%
    Price momentum  = 25%
    Market momentum = 15%

IMPORTANT:
These weights and thresholds are initial heuristics.
They must be backtested before being treated as predictive.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

NEWS_WEIGHT = 0.60
PRICE_WEIGHT = 0.25
MARKET_WEIGHT = 0.15

BUY_THRESHOLD = 0.40
SELL_THRESHOLD = -0.40

# Percentage change at which price momentum reaches
# approximately +1 or -1.

PRICE_MOMENTUM_SCALE = 5.0
MARKET_MOMENTUM_SCALE = 3.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("marketpulse.signal")


# ============================================================
# HELPERS
# ============================================================

def clamp(
    value: float,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> float:
    """
    Keep a value within a specified range.
    """

    return max(
        minimum,
        min(maximum, value),
    )


def utc_now() -> str:
    """
    Return current UTC timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# NEWS SENTIMENT
# ============================================================

def calculate_news_sentiment(
    sentiment_results: List[Dict[str, Any]],
) -> float:
    """
    Calculate average FinBERT sentiment.

    Each result is expected to contain:

        {
            "score": positive_probability - negative_probability
        }

    Score range:
        -1 = strongly negative
        +1 = strongly positive
    """

    if not sentiment_results:
        return 0.0

    scores = []

    for result in sentiment_results:

        score = result.get("score")

        if score is None:
            continue

        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue

    if not scores:
        return 0.0

    return clamp(
        sum(scores) / len(scores)
    )


# ============================================================
# PRICE MOMENTUM
# ============================================================

def calculate_price_momentum(
    change_percent: Optional[float],
) -> float:
    """
    Convert percentage price change into a bounded
    momentum score between -1 and +1.

    Example with scale = 5:

        +5% → +1
        +2.5% → +0.5
         0% → 0
        -2.5% → -0.5
        -5% → -1
    """

    if change_percent is None:
        return 0.0

    try:
        change = float(change_percent)
    except (TypeError, ValueError):
        return 0.0

    return clamp(
        change / PRICE_MOMENTUM_SCALE
    )


# ============================================================
# MARKET MOMENTUM
# ============================================================

def calculate_market_momentum(
    stock_change_percent: Optional[float],
    market_change_percent: Optional[float],
) -> float:
    """
    Calculate relative market momentum.

    We care about how the stock is performing relative
    to the broader market.

    Example:

        Stock  = +4%
        Market = +1%

        Relative movement = +3%

    The result is normalized to approximately -1 to +1.
    """

    if stock_change_percent is None:
        return 0.0

    if market_change_percent is None:
        return 0.0

    try:
        stock_change = float(
            stock_change_percent
        )

        market_change = float(
            market_change_percent
        )

    except (TypeError, ValueError):
        return 0.0

    relative_change = (
        stock_change - market_change
    )

    return clamp(
        relative_change / MARKET_MOMENTUM_SCALE
    )


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_signal(
    signal_score: float,
) -> str:
    """
    Convert numerical signal score into:

        BUY
        HOLD
        SELL
    """

    if signal_score >= BUY_THRESHOLD:
        return "BUY"

    if signal_score <= SELL_THRESHOLD:
        return "SELL"

    return "HOLD"


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    *,
    symbol: str,
    sentiment_results: List[Dict[str, Any]],
    market_data: Dict[str, Any],
    benchmark_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a MarketPulse trading signal.

    Parameters
    ----------
    symbol:
        Internal company symbol.

    sentiment_results:
        FinBERT results for recent news.

    market_data:
        Output from market_service.py.

    benchmark_data:
        Market benchmark data, such as NIFTY or S&P 500.

    Returns
    -------
    Dictionary containing signal components and final signal.
    """

    symbol = symbol.upper().strip()

    # --------------------------------------------------------
    # 1. NEWS SENTIMENT
    # --------------------------------------------------------

    news_sentiment = calculate_news_sentiment(
        sentiment_results
    )

    # --------------------------------------------------------
    # 2. PRICE MOMENTUM
    # --------------------------------------------------------

    stock_change = market_data.get(
        "change_percent"
    )

    price_momentum = calculate_price_momentum(
        stock_change
    )

    # --------------------------------------------------------
    # 3. MARKET MOMENTUM
    # --------------------------------------------------------

    market_change = None

    if benchmark_data:

        market_change = benchmark_data.get(
            "change_percent"
        )

    market_momentum = calculate_market_momentum(
        stock_change,
        market_change,
    )

    # --------------------------------------------------------
    # 4. WEIGHTED SIGNAL
    # --------------------------------------------------------

    news_component = (
        NEWS_WEIGHT * news_sentiment
    )

    price_component = (
        PRICE_WEIGHT * price_momentum
    )

    market_component = (
        MARKET_WEIGHT * market_momentum
    )

    signal_score = (
        news_component
        + price_component
        + market_component
    )

    signal_score = clamp(
        signal_score
    )

    # --------------------------------------------------------
    # 5. CLASSIFICATION
    # --------------------------------------------------------

    signal = classify_signal(
        signal_score
    )

    # --------------------------------------------------------
    # 6. RESULT
    # --------------------------------------------------------

    result = {
        "symbol": symbol,

        "signal": signal,

        "signal_score": round(
            signal_score,
            4,
        ),

        "news_sentiment": round(
            news_sentiment,
            4,
        ),

        "price_momentum": round(
            price_momentum,
            4,
        ),

        "market_momentum": round(
            market_momentum,
            4,
        ),

        "news_component": round(
            news_component,
            4,
        ),

        "price_component": round(
            price_component,
            4,
        ),

        "market_component": round(
            market_component,
            4,
        ),

        "price": market_data.get(
            "price"
        ),

        "change_percent": stock_change,

        "volume": market_data.get(
            "volume"
        ),

        "market_change_percent": market_change,

        "timestamp": utc_now(),
    }

    logger.info(
        "%s | %s | score=%+.4f | "
        "news=%+.4f | price=%+.4f | market=%+.4f",
        symbol,
        signal,
        signal_score,
        news_sentiment,
        price_momentum,
        market_momentum,
    )

    return result


# ============================================================
# SIMPLE SIGNAL WITHOUT BENCHMARK
# ============================================================

def generate_signal_without_benchmark(
    *,
    symbol: str,
    sentiment_results: List[Dict[str, Any]],
    market_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a signal when benchmark data isn't available.

    Market momentum becomes 0.

    This is useful for testing and development.
    """

    return generate_signal(
        symbol=symbol,
        sentiment_results=sentiment_results,
        market_data=market_data,
        benchmark_data=None,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARKETPULSE SIGNAL SERVICE")
    print("=" * 60)

    # --------------------------------------------------------
    # Fake FinBERT results
    # --------------------------------------------------------

    test_sentiment = [
        {
            "text": "Infosys reports strong quarterly growth",
            "label": "positive",
            "score": 0.92,
        },
        {
            "text": "Infosys receives major new contract",
            "label": "positive",
            "score": 0.81,
        },
        {
            "text": "Infosys faces margin pressure",
            "label": "negative",
            "score": -0.35,
        },
    ]

    # --------------------------------------------------------
    # Fake market data
    # --------------------------------------------------------

    test_market_data = {
        "symbol": "INFY",
        "price": 1035.0,
        "previous_close": 995.0,
        "change_percent": 4.0201,
        "volume": 2245124,
        "provider": "alpha_vantage",
    }

    # --------------------------------------------------------
    # Fake benchmark
    # --------------------------------------------------------

    test_benchmark = {
        "symbol": "NIFTY",
        "price": 25000.0,
        "change_percent": 1.20,
    }

    # --------------------------------------------------------
    # Generate signal
    # --------------------------------------------------------

    result = generate_signal(
        symbol="INFY",
        sentiment_results=test_sentiment,
        market_data=test_market_data,
        benchmark_data=test_benchmark,
    )

    print()

    for key, value in result.items():
        print(
            f"{key:24}: {value}"
        )

    print("=" * 60)