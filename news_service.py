"""
MarketPulse - News Service

Provider priority: Finnhub -> StockData.org -> Alpha Vantage -> EODHD (fallback only
when the others return nothing, to protect its smaller quota).

Env vars: FINNHUB_API_KEY, STOCKDATA_API_KEY, ALPHA_VANTAGE_API_KEY, EODHD_API_KEY
Optional: NEWS_LIMIT_PER_SOURCE=10, NEWS_REQUEST_TIMEOUT=15,
          NEWS_LOOKBACK_HOURS=24, DEFAULT_NEWS_LANGUAGE=en
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(Path(__file__).resolve().parent / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


FINNHUB_API_KEY = _env("FINNHUB_API_KEY")
STOCKDATA_API_KEY = _env("STOCKDATA_API_KEY")
ALPHA_VANTAGE_API_KEY = _env("ALPHA_VANTAGE_API_KEY")
EODHD_API_KEY = _env("EODHD_API_KEY")

NEWS_LIMIT_PER_SOURCE = int(_env("NEWS_LIMIT_PER_SOURCE", "10"))
NEWS_REQUEST_TIMEOUT = int(_env("NEWS_REQUEST_TIMEOUT", "15"))
NEWS_LOOKBACK_HOURS = int(_env("NEWS_LOOKBACK_HOURS", "24"))
DEFAULT_NEWS_LANGUAGE = _env("DEFAULT_NEWS_LANGUAGE", "en")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("marketpulse.news")

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "MarketPulse/1.0 (financial-news-analysis-project)",
        "Accept": "application/json",
    }
)

# ============================================================
# COMPANY CONFIGURATION
# ============================================================
# Internal symbols are kept separate from provider-specific symbols.


def _co(name: str, aliases: List[str], fh: str, sd: str, av: str, eod: str) -> Dict[str, Any]:
    return {
        "name": name,
        "aliases": aliases,
        "finnhub": fh,
        "stockdata": sd,
        "alpha": av,
        "eodhd": eod,
    }


COMPANIES: Dict[str, Dict[str, Any]] = {
    # US
    "AAPL": _co("Apple", ["Apple", "Apple Inc"], "AAPL", "AAPL", "AAPL", "AAPL.US"),
    "MSFT": _co("Microsoft", ["Microsoft", "Microsoft Corp"], "MSFT", "MSFT", "MSFT", "MSFT.US"),
    "NVDA": _co("NVIDIA", ["NVIDIA", "Nvidia", "NVIDIA Corporation"], "NVDA", "NVDA", "NVDA", "NVDA.US"),
    "GOOGL": _co("Alphabet", ["Google", "Alphabet", "Alphabet Inc"], "GOOGL", "GOOGL", "GOOGL", "GOOGL.US"),
    "AMZN": _co("Amazon", ["Amazon", "Amazon.com"], "AMZN", "AMZN", "AMZN", "AMZN.US"),
    "META": _co("Meta Platforms", ["Meta", "Facebook", "Meta Platforms"], "META", "META", "META", "META.US"),
    "TSLA": _co("Tesla", ["Tesla", "Tesla Inc"], "TSLA", "TSLA", "TSLA", "TSLA.US"),
    # India
    "INFY": _co("Infosys", ["Infosys", "Infosys Limited", "Infosys Technologies"], "INFY", "INFY", "INFY", "INFY.NSE"),
    "TCS": _co("Tata Consultancy Services", ["TCS", "Tata Consultancy Services", "TCS Limited"], "TCS", "TCS", "TCS", "TCS.NSE"),
    "RELIANCE": _co("Reliance Industries", ["Reliance", "Reliance Industries", "Reliance Industries Limited"], "RELIANCE", "RELIANCE", "RELIANCE", "RELIANCE.NSE"),
    "HDFCBANK": _co("HDFC Bank", ["HDFC Bank", "HDFC Bank Limited"], "HDFCBANK", "HDFCBANK", "HDFCBANK", "HDFCBANK.NSE"),
    "ICICIBANK": _co("ICICI Bank", ["ICICI Bank", "ICICI Bank Limited"], "ICICIBANK", "ICICIBANK", "ICICIBANK", "ICICIBANK.NSE"),
    "TATAMOTORS": _co("Tata Motors", ["Tata Motors", "Tata Motors Limited"], "TATAMOTORS", "TATAMOTORS", "TATAMOTORS", "TATAMOTORS.NSE"),
}

# ============================================================
# GENERAL HELPERS
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def lookback_start() -> datetime:
    """Earliest acceptable article timestamp."""
    return utc_now() - timedelta(hours=NEWS_LOOKBACK_HOURS)


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_datetime(value: Any) -> Optional[datetime]:
    """
    Parse common API datetime formats into a timezone-aware UTC datetime.

    Supports unix timestamps, Alpha Vantage (20261005T123456), ISO 8601,
    and a few common fallback formats.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OSError):
            return None

    value = safe_text(value)
    if not value:
        return None

    # Alpha Vantage: 20261005T123456
    if len(value) == 15 and "T" in value:
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # ISO format
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    logger.debug("Could not parse datetime: %s", value)
    return None


def article_id(title: str, url: str) -> str:
    """Deterministic article ID (used later by Redis to detect processed articles)."""
    raw = f"{title.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_article(
    *,
    title: Any,
    url: Any,
    source: Any,
    published_at: Any,
    company: str,
    provider: str,
    description: Any = "",
) -> Optional[Dict[str, Any]]:
    """Convert provider-specific data into the common MarketPulse article structure."""
    title = safe_text(title)
    url = safe_text(url)

    if not title:
        return None

    parsed_time = parse_datetime(published_at)

    # Drop stale articles when a timestamp exists.
    if parsed_time is not None and parsed_time < lookback_start():
        return None

    return {
        "id": article_id(title, url),
        "title": title,
        "description": safe_text(description),
        "url": url,
        "source": safe_text(source) or provider,
        "provider": provider,
        "company": company,
        "published_at": parsed_time.isoformat() if parsed_time else None,
    }


def request_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Safe GET request returning parsed JSON (dict OR list) or None on failure.

    Providers differ: Finnhub and EODHD return lists; StockData and
    Alpha Vantage return dicts.
    """
    try:
        response = SESSION.get(url, params=params, timeout=NEWS_REQUEST_TIMEOUT)

        if not response.ok:
            logger.warning(
                "HTTP %s from %s | Response: %s",
                response.status_code, url, response.text[:500],
            )
            return None

        try:
            data = response.json()
        except ValueError:
            logger.warning("Invalid JSON from %s | Response: %s", url, response.text[:500])
            return None

        if isinstance(data, (dict, list)):
            return data

        logger.warning("Unexpected JSON type from %s: %s", url, type(data).__name__)
        return None

    except requests.RequestException as exc:
        logger.warning("Request failed [%s]: %s", url, exc)
        return None


# ============================================================
# PROVIDERS (table-driven)
# ============================================================
# Per provider:
#   label    - name used in logs
#   key      - API key
#   symbol   - key in COMPANIES for the provider symbol
#   url      - endpoint
#   params   - callable(symbol) -> query params
#   expect   - expected top-level JSON type (list or dict)
#   items    - dict responses: field holding the article list (None for list responses)
#   fail     - dict keys that mean "provider refused" -> return []
#   warn     - dict keys that are logged but processing continues
#   warn_empty - log a warning if the article list is empty
#   fields   - normalized field -> candidate provider field(s), first non-empty wins

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "finnhub": {
        "label": "Finnhub",
        "key": FINNHUB_API_KEY,
        "symbol": "finnhub",
        "url": "https://finnhub.io/api/v1/company-news",
        "params": lambda s: {
            "symbol": s,
            "from": lookback_start().date().isoformat(),
            "to": utc_now().date().isoformat(),
            "token": FINNHUB_API_KEY,
        },
        "expect": list,
        "fields": {
            "title": ("headline",),
            "description": ("summary",),
            "url": ("url",),
            "source": ("source",),
            "published_at": ("datetime",),
        },
    },
    "stockdata": {
        "label": "StockData",
        "key": STOCKDATA_API_KEY,
        "symbol": "stockdata",
        "url": "https://api.stockdata.org/v1/news/all",
        "params": lambda s: {
            "api_token": STOCKDATA_API_KEY,
            "symbols": s,
            "filter_entities": "true",
            "language": DEFAULT_NEWS_LANGUAGE,
            "limit": NEWS_LIMIT_PER_SOURCE,
        },
        "expect": dict,
        "items": "data",
        "warn": ("error", "message"),
        "warn_empty": True,
        "fields": {
            "title": ("title",),
            "description": ("description", "snippet"),
            "url": ("url",),
            "source": ("source", "domain"),
            "published_at": ("published_at",),
        },
    },
    "alpha_vantage": {
        "label": "Alpha Vantage",
        "key": ALPHA_VANTAGE_API_KEY,
        "symbol": "alpha",
        "url": "https://www.alphavantage.co/query",
        "params": lambda s: {
            "function": "NEWS_SENTIMENT",
            "tickers": s,
            "limit": NEWS_LIMIT_PER_SOURCE,
            "apikey": ALPHA_VANTAGE_API_KEY,
        },
        "expect": dict,
        "items": "feed",
        "fail": ("Information", "Note", "Error Message"),
        "fields": {
            "title": ("title",),
            "description": ("summary",),
            "url": ("url",),
            "source": ("source",),
            "published_at": ("time_published",),
        },
    },
    "eodhd": {
        "label": "EODHD",
        "key": EODHD_API_KEY,
        "symbol": "eodhd",
        "url": "https://eodhd.com/api/news",
        "params": lambda s: {
            "s": s,
            "limit": NEWS_LIMIT_PER_SOURCE,
            "api_token": EODHD_API_KEY,
            "fmt": "json",
        },
        "expect": list,
        "fields": {
            "title": ("title",),
            "description": ("content",),
            "url": ("link",),
            "source": ("source",),
            "published_at": ("date",),
        },
    },
}


def _first(item: Dict[str, Any], names: tuple) -> Any:
    """Return the first truthy value among candidate fields (else the last value)."""
    value = None
    for name in names:
        value = item.get(name)
        if value:
            return value
    return value


def fetch_provider_news(provider: str, company: str) -> List[Dict[str, Any]]:
    """Generic fetch + normalize for any configured provider."""
    cfg = PROVIDERS[provider]
    label = cfg["label"]

    if not cfg["key"]:
        logger.debug("%s API key not configured", label)
        return []

    symbol = (COMPANIES.get(company) or {}).get(cfg["symbol"])
    if not symbol:
        return []

    data = request_json(cfg["url"], params=cfg["params"](symbol))

    if data is None:
        logger.warning("%s | %s returned no data", company, label)
        return []

    if not isinstance(data, cfg["expect"]):
        logger.warning(
            "%s | %s unexpected response type: %s",
            company, label, type(data).__name__,
        )
        return []

    raw_items: Any = data

    if isinstance(data, dict):
        # Provider refused the request (rate limit, bad key, etc.)
        for key in cfg.get("fail", ()):
            if key in data:
                logger.warning("%s | %s: %s", company, label, data[key])
                return []

        # Provider-level warnings; keep processing.
        if any(data.get(k) for k in cfg.get("warn", ())):
            logger.warning("%s | %s API response: %s", company, label, data)

        raw_items = data.get(cfg["items"], [])

        if not isinstance(raw_items, list):
            logger.warning("%s | %s unexpected %s field: %s", company, label, cfg["items"], raw_items)
            return []

        if not raw_items and cfg.get("warn_empty"):
            logger.warning(
                "%s | %s returned 0 articles. Response keys: %s",
                company, label, list(data.keys()),
            )
            return []

    articles: List[Dict[str, Any]] = []

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        article = normalize_article(
            company=company,
            provider=provider,
            **{field: _first(item, names) for field, names in cfg["fields"].items()},
        )

        if article:
            articles.append(article)

        if len(articles) >= NEWS_LIMIT_PER_SOURCE:
            break

    logger.info(
        "%s | %s raw=%d kept=%d (rest dropped as stale/invalid)",
        company, label, len(raw_items), len(articles),
    )

    return articles


# Thin per-provider wrappers (kept for backwards compatibility).
def fetch_finnhub_news(company: str) -> List[Dict[str, Any]]:
    return fetch_provider_news("finnhub", company)


def fetch_stockdata_news(company: str) -> List[Dict[str, Any]]:
    return fetch_provider_news("stockdata", company)


def fetch_alpha_vantage_news(company: str) -> List[Dict[str, Any]]:
    return fetch_provider_news("alpha_vantage", company)


def fetch_eodhd_news(company: str) -> List[Dict[str, Any]]:
    return fetch_provider_news("eodhd", company)


# ============================================================
# DEDUPLICATION & SORTING
# ============================================================


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate articles across providers by ID."""
    seen = set()
    unique = []

    for article in articles:
        aid = article.get("id")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        unique.append(article)

    return unique


def sort_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Newest first; articles without a timestamp go last."""
    oldest = datetime.min.replace(tzinfo=timezone.utc)

    return sorted(
        articles,
        key=lambda a: parse_datetime(a.get("published_at")) or oldest,
        reverse=True,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

PRIMARY_PROVIDERS = ("finnhub", "stockdata", "alpha_vantage")


def _try_provider(provider: str, company: str) -> List[Dict[str, Any]]:
    label = PROVIDERS[provider]["label"]
    try:
        result = fetch_provider_news(provider, company)
        logger.info("%s | %s: %d articles", company, label, len(result))
        return result
    except Exception as exc:
        logger.exception("%s | %s failed: %s", company, label, exc)
        return []


def fetch_company_news(company: str) -> List[Dict[str, Any]]:
    """
    Fetch news for one company.

    Provider strategy:
        1. Finnhub
        2. StockData
        3. Alpha Vantage
        4. EODHD only for US companies as a final fallback

    EODHD is currently excluded from the Indian news pipeline because
    its NSE symbol mapping is not working reliably for these symbols.
    """

    company = company.upper().strip()

    if company not in COMPANIES:
        logger.warning(
            "Unknown company: %s",
            company,
        )
        return []

    articles: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # PRIMARY PROVIDERS
    # --------------------------------------------------------

    for provider in PRIMARY_PROVIDERS:

        if len(articles) >= NEWS_LIMIT_PER_SOURCE:
            break

        articles.extend(
            _try_provider(
                provider,
                company,
            )
        )

    # --------------------------------------------------------
    # EODHD FALLBACK — US ONLY
    # --------------------------------------------------------

    if not articles:

        provider_symbol = (
            COMPANIES[company]
            .get("eodhd", "")
        )

        is_indian_symbol = (
            provider_symbol.endswith(".NSE")
        )

        if not is_indian_symbol:

            logger.info(
                "%s | Trying EODHD fallback...",
                company,
            )

            articles.extend(
                _try_provider(
                    "eodhd",
                    company,
                )
            )

        else:

            logger.info(
                "%s | Skipping EODHD fallback "
                "for Indian symbol.",
                company,
            )

    # --------------------------------------------------------
    # DEDUP + SORT
    # --------------------------------------------------------

    articles = sort_articles(
        deduplicate_articles(
            articles
        )
    )

    return articles[
        : NEWS_LIMIT_PER_SOURCE * 2
    ]

def fetch_news_for_companies(companies: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch news for multiple companies: {"INFY": [...], "TCS": [...]}."""
    results: Dict[str, List[Dict[str, Any]]] = {}

    for company in companies:
        company = company.upper().strip()
        try:
            results[company] = fetch_company_news(company)
        except Exception as exc:
            logger.exception("%s | News pipeline failed: %s", company, exc)
            results[company] = []

    return results


# ============================================================
# STATUS & TEST HELPERS
# ============================================================


def get_provider_status() -> Dict[str, bool]:
    """Which providers have API keys configured."""
    return {name: bool(cfg["key"]) for name, cfg in PROVIDERS.items()}


def print_provider_status() -> None:
    print("\n" + "=" * 60)
    print("MARKETPULSE NEWS PROVIDERS")
    print("=" * 60)

    for provider, configured in get_provider_status().items():
        print(f"{provider:<20} : {'READY' if configured else 'MISSING KEY'}")

    print("=" * 60)


def print_articles(articles: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(f"ARTICLES FOUND: {len(articles)}")
    print("=" * 60)

    for i, a in enumerate(articles, start=1):
        print(f"\n[{i}] {a['title']}")
        print(f"Provider : {a['provider']}")
        print(f"Source   : {a['source']}")
        print(f"Published: {a['published_at']}")
        print(f"URL      : {a['url']}")


if __name__ == "__main__":
    print_provider_status()
    print("\nTesting news pipeline with INFY...\n")
    print_articles(fetch_company_news("INFY"))