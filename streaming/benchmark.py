"""
Benchmark change for non-Indian symbols (the legacy code only had NIFTY, so US stocks always got a
market-momentum component of 0 and the signal silently lost 15% of its weight).

    BENCHMARK_US=^GSPC     Yahoo ticker used for symbols that are not in INDIAN_SYMBOLS
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence

logger = logging.getLogger("marketpulse.benchmark")
TTL_SECONDS = int(os.getenv("QUOTE_CACHE_SECONDS", "300"))
_cache: Dict[str, Any] = {"time": 0.0, "value": None}


def pct_change(closes: Sequence[float]) -> Optional[float]:
    """% change of the last close vs the one before it (None if unusable)."""
    vals = [float(c) for c in closes if c == c and c]       # drops NaN and zeros
    if len(vals) < 2 or vals[-2] <= 0:
        return None
    return (vals[-1] / vals[-2] - 1.0) * 100.0


def _download(ticker: str) -> Optional[Sequence[float]]:
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
    return None if hist.empty else hist["Close"].tolist()


def fetch_us_benchmark(download: Callable[[str], Optional[Sequence[float]]] = _download,
                       now: Callable[[], float] = time.time) -> Optional[Dict[str, Any]]:
    """Cached {'change_percent': x, 'symbol': ticker}; last good value if Yahoo fails."""
    ticker = os.getenv("BENCHMARK_US", "^GSPC")
    if _cache["value"] and now() - _cache["time"] < TTL_SECONDS:
        return _cache["value"]
    try:
        change = pct_change(download(ticker) or [])
    except Exception as exc:  # noqa: BLE001 - provider failures must never stop the aggregator
        logger.warning("US benchmark %s failed: %s", ticker, exc)
        change = None
    if change is not None:
        _cache["time"], _cache["value"] = now(), {"change_percent": round(change, 4), "symbol": ticker}
        logger.info("US benchmark %s: %+.2f%%", ticker, change)
    elif not _cache["value"]:
        logger.warning("US benchmark %s unavailable (no usable closes); market momentum will be 0", ticker)
    return _cache["value"]