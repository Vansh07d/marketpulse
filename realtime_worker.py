"""
MarketPulse - Real-Time Background Worker

Pipeline:

    News APIs
        ↓
    Redis deduplication
        ↓
    New articles only
        ↓
    FinBERT
        ↓
    Redis news state
        ↓
    Fresh market data
        ↓
    Signal engine
        ↓
    Redis

Development mode:
    INFY only
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Any, Dict, List

from news_service import fetch_company_news
from sentiment_service import get_finbert
from market_service import (
    fetch_market_data,
    fetch_nifty_benchmark,
)
from signal_service import generate_signal
from redis_client import get_redis



# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "INFY"

POLL_INTERVAL_SECONDS = 60

MAX_NEW_ARTICLES_PER_CYCLE = 10

MAX_STORED_ARTICLES = 20


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "marketpulse.worker"
)


# ============================================================
# WORKER STATE
# ============================================================

running = True


def stop_worker(
    signum: int,
    frame: Any,
) -> None:

    global running

    logger.info(
        "Shutdown signal received."
    )

    running = False


signal.signal(
    signal.SIGINT,
    stop_worker,
)

signal.signal(
    signal.SIGTERM,
    stop_worker,
)


# ============================================================
# FETCH NEW ARTICLES
# ============================================================

def get_new_articles(
    redis_client: Any,
    symbol: str,
) -> List[Dict[str, Any]]:

    logger.info(
        "%s | Fetching news...",
        symbol,
    )

    try:

        articles = fetch_company_news(
            symbol
        )

    except Exception as exc:

        logger.exception(
            "%s | News fetch failed: %s",
            symbol,
            exc,
        )

        return []

    logger.info(
        "%s | %d articles returned",
        symbol,
        len(articles),
    )

    new_articles = []

    for article in articles:

        if redis_client.is_news_processed(
            article
        ):

            logger.debug(
                "%s | Already processed: %s",
                symbol,
                article.get("title"),
            )

            continue

        new_articles.append(
            article
        )

    new_articles = new_articles[
        :MAX_NEW_ARTICLES_PER_CYCLE
    ]

    logger.info(
        "%s | %d new articles",
        symbol,
        len(new_articles),
    )

    return new_articles


# ============================================================
# PROCESS NEW ARTICLES WITH FINBERT
# ============================================================

def process_articles(
    symbol: str,
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    if not articles:

        return []

    headlines = [
        article.get(
            "title",
            "",
        ).strip()
        for article in articles
    ]

    valid_articles = []
    valid_headlines = []

    for article, headline in zip(
        articles,
        headlines,
    ):

        if not headline:
            continue

        valid_articles.append(
            article
        )

        valid_headlines.append(
            headline
        )

    if not valid_headlines:

        logger.warning(
            "%s | No valid headlines.",
            symbol,
        )

        return []

    logger.info(
        "%s | Running FinBERT on %d new headlines...",
        symbol,
        len(valid_headlines),
    )

    try:

        finbert = get_finbert()

        sentiment_results = (
            finbert.predict_batch(
                valid_headlines
            )
        )

    except Exception as exc:

        logger.exception(
            "%s | FinBERT failed: %s",
            symbol,
            exc,
        )

        return []

    processed_articles = []

    for article, sentiment in zip(
        valid_articles,
        sentiment_results,
    ):

        article_with_sentiment = dict(
            article
        )

        article_with_sentiment[
            "sentiment"
        ] = sentiment

        processed_articles.append(
            article_with_sentiment
        )

        logger.info(
            "%s | %s | score=%+.4f",
            symbol,
            sentiment.get(
                "label",
                "unknown",
            ).upper(),
            sentiment.get(
                "score",
                0.0,
            ),
        )

    return processed_articles


# ============================================================
# UPDATE REDIS NEWS STATE
# ============================================================

def update_news_state(
    redis_client: Any,
    symbol: str,
    new_articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    existing_articles = (
        redis_client.get_news(
            symbol
        )
        or []
    )

    combined = (
        new_articles
        + existing_articles
    )

    unique_articles = []

    seen_ids = set()

    for article in combined:

        article_id = article.get(
            "id"
        )

        if not article_id:

            article_id = (
                redis_client
                .make_article_hash(
                    article
                )
            )

        if article_id in seen_ids:
            continue

        seen_ids.add(
            article_id
        )

        unique_articles.append(
            article
        )

    unique_articles = (
        unique_articles[
            :MAX_STORED_ARTICLES
        ]
    )

    redis_client.set_news(
        symbol,
        unique_articles,
    )

    return unique_articles


# ============================================================
# EXTRACT STORED SENTIMENT
# ============================================================

def get_stored_sentiment(
    redis_client: Any,
    symbol: str,
) -> List[Dict[str, Any]]:

    stored_news = (
        redis_client.get_news(
            symbol
        )
        or []
    )

    sentiment_results = []

    for article in stored_news:

        sentiment = article.get(
            "sentiment"
        )

        if not sentiment:
            continue

        sentiment_results.append(
            sentiment
        )

    return sentiment_results


# ============================================================
# GENERATE + STORE SIGNAL
# ============================================================

def update_signal(
    redis_client: Any,
    symbol: str,
    sentiment_results: List[Dict[str, Any]],
    market_data: Dict[str, Any],
) -> Dict[str, Any]:

    signal_result = generate_signal(
        symbol=symbol,
        sentiment_results=sentiment_results,
        market_data=market_data,
        benchmark_data=None,
    )

    redis_client.set_signal(
        symbol,
        signal_result,
    )

    logger.info(
        "%s | %s | score=%+.4f",
        symbol,
        signal_result.get(
            "signal"
        ),
        signal_result.get(
            "signal_score",
            0.0,
        ),
    )

    return signal_result


# ============================================================
# SINGLE WORKER CYCLE
# ============================================================

def run_cycle(
    redis_client: Any,
    symbol: str,
) -> None:

    logger.info(
        "=================================================="
    )

    logger.info(
        "%s | Starting worker cycle",
        symbol,
    )

    # ========================================================
    # STEP 1 — FETCH NEW NEWS
    # ========================================================

    new_articles = get_new_articles(
        redis_client,
        symbol,
    )

    # ========================================================
    # STEP 2 — RUN FINBERT ONLY ON NEW ARTICLES
    # ========================================================

    processed_articles = process_articles(
        symbol,
        new_articles,
    )

    # ========================================================
    # STEP 3 — UPDATE REDIS NEWS STATE
    # ========================================================

    if processed_articles:

        update_news_state(
            redis_client,
            symbol,
            processed_articles,
        )

        for article in processed_articles:

            redis_client.mark_news_processed(
                article
            )

        logger.info(
            "%s | Stored %d processed articles",
            symbol,
            len(processed_articles),
        )

    else:

        logger.info(
            "%s | No new articles to process",
            symbol,
        )

    # ========================================================
    # STEP 4 — GET CURRENT MARKET DATA
    # ========================================================

    logger.info(
        "%s | Fetching market data...",
        symbol,
    )

    try:

        market_data = fetch_market_data(
            symbol
        )

    except Exception as exc:

        logger.exception(
            "%s | Market data failed: %s",
            symbol,
            exc,
        )

        return

    if not market_data:

        logger.warning(
            "%s | No market data available.",
            symbol,
        )

        return

    # ========================================================
    # STEP 5 — GET ALL CURRENT SENTIMENT FROM REDIS
    # ========================================================

    sentiment_results = (
        get_stored_sentiment(
            redis_client,
            symbol,
        )
    )

    if not sentiment_results:

        logger.warning(
            "%s | No stored sentiment available "
            "for signal generation.",
            symbol,
        )

        return

    if processed_articles:

        logger.info(
            "%s | Using newly updated sentiment state",
            symbol,
        )

    else:

        logger.info(
            "%s | No new news — reusing existing "
            "sentiment from Redis",
            symbol,
        )

    # ========================================================
    # STEP 6 — GENERATE SIGNAL
    # ========================================================

    update_signal(
        redis_client,
        symbol,
        sentiment_results,
        market_data,
    )

    logger.info(
        "%s | Worker cycle complete",
        symbol,
    )


# ============================================================
# MAIN WORKER LOOP
# ============================================================

def main() -> None:

    global running

    print()
    print("=" * 70)
    print("MARKETPULSE REAL-TIME WORKER")
    print("=" * 70)

    print(
        f"Symbol         : {SYMBOL}"
    )

    print(
        f"Poll interval  : "
        f"{POLL_INTERVAL_SECONDS} seconds"
    )

    print("=" * 70)
    print()

    # ========================================================
    # REDIS
    # ========================================================

    redis_client = get_redis()

    if not redis_client.ping():

        logger.error(
            "Redis is unavailable."
        )

        raise SystemExit(1)

    logger.info(
        "Redis connection successful."
    )

    # ========================================================
    # WORKER LOOP
    # ========================================================

    while running:

        cycle_started = time.time()

        try:

            run_cycle(
                redis_client,
                SYMBOL,
            )

        except Exception as exc:

            logger.exception(
                "Unexpected worker error: %s",
                exc,
            )

        elapsed = (
            time.time()
            - cycle_started
        )

        sleep_time = max(
            0,
            POLL_INTERVAL_SECONDS
            - elapsed,
        )

        if running:

            logger.info(
                "%s | Sleeping %.1f seconds...",
                SYMBOL,
                sleep_time,
            )

            time.sleep(
                sleep_time
            )

    logger.info(
        "MarketPulse worker stopped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()