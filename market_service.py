"""
MarketPulse - Market Data Service

Fetches current market data with provider fallbacks.

Provider priority:
    1. Finnhub
    2. Alpha Vantage
    3. EODHD

The service normalizes all providers into one common format so
the signal engine does not need to know which API supplied the data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()

REQUEST_TIMEOUT = int(
    os.getenv("MARKET_REQUEST_TIMEOUT", "15")
)

FINNHUB_URL = "https://finnhub.io/api/v1/quote"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
EODHD_URL = "https://eodhd.com/api/real-time"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("marketpulse.market")


# ============================================================
# SYMBOL MAPPING
# ============================================================

# Internal symbols → provider-specific symbols

FINNHUB_SYMBOLS = {
    # India — NSE
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "TATAMOTORS": "TATAMOTORS.NS",

    # US
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "TSLA": "TSLA",
}


# Alpha Vantage generally needs exchange-qualified
# symbols for Indian equities.

ALPHA_SYMBOLS = {
    "INFY": "INFY.BSE",
    "TCS": "TCS.BSE",
    "RELIANCE": "RELIANCE.BSE",
    "HDFCBANK": "HDFCBANK.BSE",
    "ICICIBANK": "ICICIBANK.BSE",
    "TATAMOTORS": "TATAMOTORS.BSE",

    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "TSLA": "TSLA",
}


# EODHD uses exchange-qualified symbols.

EODHD_SYMBOLS = {
    "INFY": "INFY.BSE",
    "TCS": "TCS.BSE",
    "RELIANCE": "RELIANCE.BSE",
    "HDFCBANK": "HDFCBANK.BSE",
    "ICICIBANK": "ICICIBANK.BSE",
    "TATAMOTORS": "TATAMOTORS.BSE",

    "AAPL": "AAPL.US",
    "MSFT": "MSFT.US",
    "NVDA": "NVDA.US",
    "GOOGL": "GOOGL.US",
    "AMZN": "AMZN.US",
    "META": "META.US",
    "TSLA": "TSLA.US",
}


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_change_percent(
    current: Optional[float],
    previous: Optional[float],
) -> Optional[float]:
    """Calculate percentage change."""
    if current is None or previous is None:
        return None

    if previous == 0:
        return None

    return ((current - previous) / previous) * 100


def normalize_market_data(
    *,
    symbol: str,
    price: Optional[float],
    previous_close: Optional[float],
    change_percent: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    open_price: Optional[float] = None,
    volume: Optional[float] = None,
    provider: str,
    provider_symbol: str,
) -> Dict[str, Any]:
    """
    Normalize provider-specific market data into one structure.
    """

    if change_percent is None:
        change_percent = calculate_change_percent(
            price,
            previous_close,
        )

    return {
        "symbol": symbol,
        "price": price,
        "previous_close": previous_close,
        "change_percent": change_percent,
        "open": open_price,
        "high": high,
        "low": low,
        "volume": volume,
        "provider": provider,
        "provider_symbol": provider_symbol,
        "timestamp": utc_now(),
    }


# ============================================================
# FINNHUB
# ============================================================

def fetch_finnhub_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch latest quote from Finnhub.
    """

    if not FINNHUB_API_KEY:
        logger.warning("Finnhub API key not configured.")
        return None

    provider_symbol = FINNHUB_SYMBOLS.get(
        symbol.upper(),
        symbol.upper(),
    )

    params = {
        "symbol": provider_symbol,
        "token": FINNHUB_API_KEY,
    }

    try:
        response = requests.get(
            FINNHUB_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        logger.warning(
            "%s | Finnhub request failed: %s",
            symbol,
            exc,
        )
        return None

    except ValueError:
        logger.warning(
            "%s | Finnhub returned invalid JSON",
            symbol,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "%s | Unexpected Finnhub response format",
            symbol,
        )
        return None

    # Finnhub quote fields:
    #
    # c  = current price
    # d  = change
    # dp = percentage change
    # h  = high
    # l  = low
    # o  = open
    # pc = previous close
    # t  = timestamp

    price = safe_float(data.get("c"))
    previous_close = safe_float(data.get("pc"))

    if price is None:
        logger.info(
            "%s | Finnhub returned no current price",
            symbol,
        )
        return None

    change_percent = safe_float(data.get("dp"))

    result = normalize_market_data(
        symbol=symbol,
        price=price,
        previous_close=previous_close,
        change_percent=change_percent,
        high=safe_float(data.get("h")),
        low=safe_float(data.get("l")),
        open_price=safe_float(data.get("o")),
        provider="finnhub",
        provider_symbol=provider_symbol,
    )

    logger.info(
        "%s | Finnhub: %.2f (%+.2f%%)",
        symbol,
        price,
        change_percent if change_percent is not None else 0,
    )

    return result


# ============================================================
# ALPHA VANTAGE
# ============================================================

def fetch_alpha_vantage_quote(
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch latest quote from Alpha Vantage.
    """

    if not ALPHA_VANTAGE_API_KEY:
        logger.warning(
            "Alpha Vantage API key not configured."
        )
        return None

    provider_symbol = ALPHA_SYMBOLS.get(
        symbol.upper(),
        symbol.upper(),
    )

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": provider_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        logger.warning(
            "%s | Alpha Vantage request failed: %s",
            symbol,
            exc,
        )
        return None

    except ValueError:
        logger.warning(
            "%s | Alpha Vantage returned invalid JSON",
            symbol,
        )
        return None

    if not isinstance(data, dict):
        return None

    if "Error Message" in data:
        logger.warning(
            "%s | Alpha Vantage error: %s",
            symbol,
            data["Error Message"],
        )
        return None

    if "Note" in data:
        logger.warning(
            "%s | Alpha Vantage rate limit: %s",
            symbol,
            data["Note"],
        )
        return None

    quote = data.get("Global Quote")

    if not isinstance(quote, dict):
        logger.info(
            "%s | Alpha Vantage returned no quote",
            symbol,
        )
        return None

    price = safe_float(
        quote.get("05. price")
    )

    previous_close = safe_float(
        quote.get("08. previous close")
    )

    if price is None:
        return None

    change_percent_raw = quote.get(
        "10. change percent"
    )

    change_percent = None

    if change_percent_raw:
        try:
            change_percent = float(
                str(change_percent_raw)
                .replace("%", "")
                .strip()
            )
        except ValueError:
            pass

    result = normalize_market_data(
        symbol=symbol,
        price=price,
        previous_close=previous_close,
        change_percent=change_percent,
        high=safe_float(
            quote.get("03. high")
        ),
        low=safe_float(
            quote.get("04. low")
        ),
        open_price=safe_float(
            quote.get("02. open")
        ),
        volume=safe_float(
            quote.get("06. volume")
        ),
        provider="alpha_vantage",
        provider_symbol=provider_symbol,
    )

    logger.info(
        "%s | Alpha Vantage: %.2f",
        symbol,
        price,
    )

    return result


# ============================================================
# EODHD
# ============================================================

def fetch_eodhd_quote(
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch latest quote from EODHD.
    """

    if not EODHD_API_KEY:
        logger.warning(
            "EODHD API key not configured."
        )
        return None

    provider_symbol = EODHD_SYMBOLS.get(
        symbol.upper(),
        symbol.upper(),
    )

    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
    }

    url = f"{EODHD_URL}/{provider_symbol}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:
        logger.warning(
            "%s | EODHD request failed: %s",
            symbol,
            exc,
        )
        return None

    except ValueError:
        logger.warning(
            "%s | EODHD returned invalid JSON",
            symbol,
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "%s | Unexpected EODHD response",
            symbol,
        )
        return None

    if "error" in data:
        logger.warning(
            "%s | EODHD error: %s",
            symbol,
            data["error"],
        )
        return None

    price = safe_float(data.get("close"))

    previous_close = safe_float(
        data.get("previousClose")
    )

    if price is None:
        logger.info(
            "%s | EODHD returned no price",
            symbol,
        )
        return None

    change_percent = safe_float(
        data.get("change_p")
    )

    result = normalize_market_data(
        symbol=symbol,
        price=price,
        previous_close=previous_close,
        change_percent=change_percent,
        high=safe_float(data.get("high")),
        low=safe_float(data.get("low")),
        open_price=safe_float(data.get("open")),
        volume=safe_float(data.get("volume")),
        provider="eodhd",
        provider_symbol=provider_symbol,
    )

    logger.info(
        "%s | EODHD: %.2f",
        symbol,
        price,
    )

    return result


# ============================================================
# PROVIDER FALLBACK
# ============================================================

def fetch_market_data(
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch market data using provider fallback.

    Indian stocks:
        Alpha Vantage → EODHD

    US stocks:
        Finnhub → Alpha Vantage → EODHD
    """

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    logger.info(
        "Fetching market data for %s...",
        symbol,
    )

    # Indian equities
    indian_symbols = {
        "INFY",
        "TCS",
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "TATAMOTORS",
    }

    # --------------------------------------------------------
    # INDIA
    # --------------------------------------------------------

    if symbol in indian_symbols:

        result = fetch_alpha_vantage_quote(symbol)

        if result:
            return result

        result = fetch_eodhd_quote(symbol)

        if result:
            return result

    # --------------------------------------------------------
    # US / OTHER
    # --------------------------------------------------------

    else:

        result = fetch_finnhub_quote(symbol)

        if result:
            return result

        result = fetch_alpha_vantage_quote(symbol)

        if result:
            return result

        result = fetch_eodhd_quote(symbol)

        if result:
            return result

    logger.error(
        "%s | All market providers failed.",
        symbol,
    )

    return None


# ============================================================
# MULTIPLE SYMBOLS
# ============================================================

def fetch_market_data_for_symbols(
    symbols: list[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch market data for multiple symbols.
    """

    results: Dict[str, Dict[str, Any]] = {}

    for symbol in symbols:
        data = fetch_market_data(symbol)

        if data:
            results[symbol.upper()] = data

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARKETPULSE MARKET DATA SERVICE")
    print("=" * 60)

    print(
        f"Finnhub        : "
        f"{'READY' if FINNHUB_API_KEY else 'MISSING'}"
    )

    print(
        f"Alpha Vantage  : "
        f"{'READY' if ALPHA_VANTAGE_API_KEY else 'MISSING'}"
    )

    print(
        f"EODHD          : "
        f"{'READY' if EODHD_API_KEY else 'MISSING'}"
    )

    print("=" * 60)

    test_symbol = "INFY"

    print(
        f"\nTesting market pipeline with "
        f"{test_symbol}...\n"
    )

    result = fetch_market_data(test_symbol)

    print("\n" + "=" * 60)

    if result:

        print("MARKET DATA FOUND")
        print("=" * 60)

        for key, value in result.items():
            print(f"{key:18}: {value}")

    else:

        print(
            "NO MARKET DATA FOUND FROM ANY PROVIDER."
        )

    print("=" * 60)