"""
MarketPulse - Signal Service

Combines FinBERT news sentiment (with recency decay), stock price momentum and
relative market momentum into one BUY / HOLD / SELL signal.

Weights (news 60 / price 25 / market 15) and thresholds are initial heuristics;
backtest them before treating the signal as predictive.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

NEWS_WEIGHT, PRICE_WEIGHT, MARKET_WEIGHT = 0.60, 0.25, 0.15
BUY_THRESHOLD, SELL_THRESHOLD = 0.40, -0.40
PRICE_MOMENTUM_SCALE = 5.0    # % move that maps to +/-1
MARKET_MOMENTUM_SCALE = 3.0   # % outperformance vs benchmark that maps to +/-1
NEWS_HALF_LIFE_HOURS = max(1.0, float(os.getenv("NEWS_HALF_LIFE_HOURS", "6")))
# Extra "neutral" weight in the news average. 0 keeps the plain decay-weighted mean;
# ~0.5 makes old news fade toward neutral instead of keeping full strength.
NEWS_PRIOR_WEIGHT = max(0.0, float(os.getenv("NEWS_PRIOR_WEIGHT", "0")))

logger = logging.getLogger("marketpulse.signal")


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _decay_weight(published_at: Any, now: datetime) -> float:
    if not published_at:
        return 1.0
    try:
        ts = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (now - ts.astimezone(timezone.utc)).total_seconds() / 3600)
        return math.exp(-math.log(2) * age_hours / NEWS_HALF_LIFE_HOURS)
    except (ValueError, TypeError):
        return 1.0


def calculate_news_sentiment(sentiment_results: List[Dict[str, Any]]) -> float:
    """Recency-weighted mean FinBERT score in [-1, 1]."""
    now = datetime.now(timezone.utc)
    total = weights = 0.0
    for result in sentiment_results or []:
        score = _to_float(result.get("score"))
        if score is None:
            continue
        w = _decay_weight(result.get("published_at"), now)
        total += clamp(score) * w
        weights += w
    return clamp(total / (weights + NEWS_PRIOR_WEIGHT)) if weights else 0.0


def calculate_price_momentum(change_percent: Optional[float]) -> float:
    change = _to_float(change_percent)
    return 0.0 if change is None else clamp(change / PRICE_MOMENTUM_SCALE)


def calculate_market_momentum(stock_change: Optional[float], market_change: Optional[float]) -> float:
    stock, market = _to_float(stock_change), _to_float(market_change)
    return 0.0 if stock is None or market is None else clamp((stock - market) / MARKET_MOMENTUM_SCALE)


def classify_signal(signal_score: float) -> str:
    if signal_score >= BUY_THRESHOLD:
        return "BUY"
    if signal_score <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def generate_signal(
    *,
    symbol: str,
    sentiment_results: List[Dict[str, Any]],
    market_data: Dict[str, Any],
    benchmark_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    symbol = symbol.upper().strip()
    stock_change = market_data.get("change_percent")
    market_change = (benchmark_data or {}).get("change_percent")

    news = calculate_news_sentiment(sentiment_results)
    price = calculate_price_momentum(stock_change)
    market = calculate_market_momentum(stock_change, market_change)

    news_c, price_c, market_c = NEWS_WEIGHT * news, PRICE_WEIGHT * price, MARKET_WEIGHT * market
    score = clamp(news_c + price_c + market_c)
    signal = classify_signal(score)
    now = utc_now()

    logger.info("%s | %s | score=%+.4f | news=%+.4f | price=%+.4f | market=%+.4f",
                symbol, signal, score, news, price, market)
    return {
        "symbol": symbol,
        "signal": signal,
        "signal_score": round(score, 4),
        "news_sentiment": round(news, 4),
        "price_momentum": round(price, 4),
        "market_momentum": round(market, 4),
        "news_component": round(news_c, 4),
        "price_component": round(price_c, 4),
        "market_component": round(market_c, 4),
        "price": market_data.get("price"),
        "change_percent": stock_change,
        "volume": market_data.get("volume"),
        "market_change_percent": market_change,
        "market_data": market_data,   # read by the dashboard
        "updated_at": now,            # read by the dashboard
        "timestamp": now,
    }


if __name__ == "__main__":
    demo = generate_signal(
        symbol="INFY",
        sentiment_results=[{"score": 0.92}, {"score": 0.81}, {"score": -0.35}],
        market_data={"price": 1035.0, "change_percent": 4.02, "volume": 2245124},
        benchmark_data={"change_percent": 1.2},
    )
    for key, value in demo.items():
        print(f"{key:24}: {value}")