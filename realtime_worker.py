"""
MarketPulse - Real-Time Background Worker

News APIs -> Redis dedup -> new articles only -> FinBERT -> Redis news state
          -> market data + benchmark -> signal engine -> Redis

Development mode: one symbol (DEFAULT_SYMBOL, default INFY).
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any, Dict, List

from market_service import fetch_market_data, fetch_nifty_benchmark
from news_service import fetch_company_news
from redis_client import get_redis
from sentiment_service import get_finbert
from signal_service import generate_signal

SYMBOL = os.getenv("DEFAULT_SYMBOL", "INFY").strip().upper()
POLL_INTERVAL_SECONDS = max(10, int(os.getenv("POLL_INTERVAL_SECONDS", "300")))
MAX_NEW_ARTICLES_PER_CYCLE = 10
MAX_STORED_ARTICLES = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("marketpulse.worker")

running = True


def stop_worker(signum: int, frame: Any) -> None:
    global running
    logger.info("Shutdown requested - finishing the current step...")
    running = False


signal.signal(signal.SIGINT, stop_worker)
signal.signal(signal.SIGTERM, stop_worker)


def get_new_articles(redis_client: Any, symbol: str) -> List[Dict[str, Any]]:
    try:
        articles = fetch_company_news(symbol)
    except Exception:
        logger.exception("%s | News fetch failed", symbol)
        return []
    new = [a for a in articles if not redis_client.is_news_processed(a)][:MAX_NEW_ARTICLES_PER_CYCLE]
    logger.info("%s | %d articles returned, %d new", symbol, len(articles), len(new))
    return new


def process_articles(symbol: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    articles = [a for a in articles if (a.get("title") or "").strip()]
    if not articles:
        return []
    logger.info("%s | Running FinBERT on %d new headlines...", symbol, len(articles))
    try:
        results = get_finbert().predict_batch([a["title"].strip() for a in articles])
    except Exception:
        logger.exception("%s | FinBERT failed", symbol)
        return []
    for article, sentiment in zip(articles, results):
        logger.info("%s | %s | score=%+.4f", symbol, sentiment.get("label", "unknown").upper(),
                    sentiment.get("score", 0.0))
    return [{**a, "sentiment": s} for a, s in zip(articles, results)]


def update_news_state(redis_client: Any, symbol: str, new_articles: List[Dict[str, Any]]) -> None:
    seen, merged = set(), []
    for article in new_articles + (redis_client.get_news(symbol) or []):
        article_id = article.get("id") or redis_client.make_article_hash(article)
        if article_id not in seen:
            seen.add(article_id)
            merged.append(article)
    redis_client.set_news(symbol, merged[:MAX_STORED_ARTICLES])


def get_stored_sentiment(redis_client: Any, symbol: str) -> List[Dict[str, Any]]:
    return [
        {**a["sentiment"], "published_at": a.get("published_at")}
        for a in (redis_client.get_news(symbol) or [])
        if a.get("sentiment")
    ]


def run_cycle(redis_client: Any, symbol: str) -> None:
    logger.info("%s | Starting worker cycle", symbol)

    processed = process_articles(symbol, get_new_articles(redis_client, symbol))
    if processed:
        update_news_state(redis_client, symbol, processed)
        for article in processed:
            redis_client.mark_news_processed(article)
        logger.info("%s | Stored %d processed articles", symbol, len(processed))
    else:
        # re-set so the stored news keeps its TTL while nothing new arrives
        stored = redis_client.get_news(symbol)
        if stored:
            redis_client.set_news(symbol, stored)

    try:
        market_data = fetch_market_data(symbol)
    except Exception:
        logger.exception("%s | Market data failed", symbol)
        return
    if not market_data:
        logger.warning("%s | No market data available.", symbol)
        return

    sentiments = get_stored_sentiment(redis_client, symbol)
    if not sentiments:
        logger.warning("%s | No stored sentiment available for signal generation.", symbol)
        return

    try:
        benchmark = fetch_nifty_benchmark()   # cached; last good value on failure
    except Exception as exc:
        logger.warning("Benchmark failed; generating signal without it: %s", exc)
        benchmark = None

    result = generate_signal(symbol=symbol, sentiment_results=sentiments,
                             market_data=market_data, benchmark_data=benchmark)
    redis_client.set_signal(symbol, result)
    logger.info("%s | Worker cycle complete", symbol)


def main() -> None:
    print("=" * 70)
    print("MARKETPULSE REAL-TIME WORKER")
    print(f"Symbol: {SYMBOL} | Poll interval: {POLL_INTERVAL_SECONDS}s | Ctrl+C once to stop")
    print("=" * 70)

    redis_client = get_redis()
    if not redis_client.ping():
        raise SystemExit("Redis is unavailable.")
    logger.info("Redis connection successful.")

    while running:
        started = time.time()
        try:
            run_cycle(redis_client, SYMBOL)
        except Exception:
            logger.exception("Unexpected worker error")

        deadline = started + POLL_INTERVAL_SECONDS
        if running and time.time() < deadline:
            logger.info("%s | Sleeping %.0f seconds...", SYMBOL, deadline - time.time())
        while running and time.time() < deadline:   # 1s steps so Ctrl+C is instant while idle
            time.sleep(1)

    logger.info("MarketPulse worker stopped.")


if __name__ == "__main__":
    main()