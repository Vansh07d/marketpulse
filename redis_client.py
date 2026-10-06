from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import redis
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv(
        "REDIS_PORT",
        "6379",
    )
)

REDIS_DB = int(
    os.getenv(
        "REDIS_DB",
        "0",
    )
)

REDIS_PASSWORD = os.getenv(
    "REDIS_PASSWORD",
    None,
)


# ============================================================
# CONSTANTS
# ============================================================

SIGNAL_KEY_PREFIX = "signal:"
NEWS_KEY_PREFIX = "news:"
PROCESSED_NEWS_PREFIX = "processed:news:"


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "marketpulse.redis"
)


# ============================================================
# REDIS CLIENT
# ============================================================

class RedisClient:
    """
    Thin wrapper around redis-py.

    Keeps Redis-specific logic isolated from the rest
    of the MarketPulse application.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD,
    ) -> None:

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

    # ========================================================
    # HEALTH
    # ========================================================

    def ping(self) -> bool:
        """
        Check whether Redis is reachable.
        """

        try:

            return bool(
                self.client.ping()
            )

        except redis.RedisError as exc:

            logger.error(
                "Redis ping failed: %s",
                exc,
            )

            return False

    # ========================================================
    # GENERIC JSON OPERATIONS
    # ========================================================

    def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:

        try:

            payload = json.dumps(
                value,
                ensure_ascii=False,
            )

            self.client.set(
                key,
                payload,
                ex=ttl,
            )

            return True

        except (
            redis.RedisError,
            TypeError,
            ValueError,
        ) as exc:

            logger.error(
                "Redis SET failed for %s: %s",
                key,
                exc,
            )

            return False

    def get_json(
        self,
        key: str,
    ) -> Optional[Any]:

        try:

            value = self.client.get(
                key
            )

            if value is None:
                return None

            return json.loads(value)

        except (
            redis.RedisError,
            json.JSONDecodeError,
        ) as exc:

            logger.error(
                "Redis GET failed for %s: %s",
                key,
                exc,
            )

            return None

    def delete(
        self,
        key: str,
    ) -> bool:

        try:

            self.client.delete(key)

            return True

        except redis.RedisError as exc:

            logger.error(
                "Redis DELETE failed for %s: %s",
                key,
                exc,
            )

            return False

    # ========================================================
    # SIGNALS
    # ========================================================

    def set_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        ttl: int = 3600,
    ) -> bool:

        key = (
            f"{SIGNAL_KEY_PREFIX}"
            f"{symbol.upper()}"
        )

        return self.set_json(
            key,
            signal,
            ttl=ttl,
        )

    def get_signal(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:

        key = (
            f"{SIGNAL_KEY_PREFIX}"
            f"{symbol.upper()}"
        )

        result = self.get_json(key)

        if result is None:
            return None

        if not isinstance(
            result,
            dict,
        ):
            return None

        return result

    # ========================================================
    # NEWS
    # ========================================================

    def set_news(
        self,
        symbol: str,
        articles: List[Dict[str, Any]],
        ttl: int = 1800,
    ) -> bool:

        key = (
            f"{NEWS_KEY_PREFIX}"
            f"{symbol.upper()}"
        )

        return self.set_json(
            key,
            articles,
            ttl=ttl,
        )

    def get_news(
        self,
        symbol: str,
    ) -> Optional[List[Dict[str, Any]]]:

        key = (
            f"{NEWS_KEY_PREFIX}"
            f"{symbol.upper()}"
        )

        result = self.get_json(key)

        if result is None:
            return None

        if not isinstance(
            result,
            list,
        ):
            return None

        return result

    # ========================================================
    # PROCESSED NEWS / DEDUPLICATION
    # ========================================================

    @staticmethod
    def make_article_hash(
        article: Dict[str, Any],
    ) -> str:
        """
        Generate a stable hash for an article.

        Uses title + URL because those together
        identify an article reliably enough for
        our current deduplication layer.
        """

        title = str(
            article.get(
                "title",
                "",
            )
        ).strip()

        url = str(
            article.get(
                "url",
                "",
            )
        ).strip()

        raw = (
            f"{title}|{url}"
        )

        return hashlib.sha256(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()

    def is_news_processed(
        self,
        article: Dict[str, Any],
    ) -> bool:
        """
        Check whether an article has already
        been processed by the worker.
        """

        article_hash = (
            self.make_article_hash(
                article
            )
        )

        key = (
            f"{PROCESSED_NEWS_PREFIX}"
            f"{article_hash}"
        )

        try:

            return bool(
                self.client.exists(key)
            )

        except redis.RedisError as exc:

            logger.error(
                "Redis dedup check failed: %s",
                exc,
            )

            return False

    def mark_news_processed(
        self,
        article: Dict[str, Any],
        ttl: int = 86400,
    ) -> bool:
        """
        Mark an article as processed.

        Default retention = 24 hours.
        """

        article_hash = (
            self.make_article_hash(
                article
            )
        )

        key = (
            f"{PROCESSED_NEWS_PREFIX}"
            f"{article_hash}"
        )

        try:

            self.client.set(
                key,
                "1",
                ex=ttl,
            )

            return True

        except redis.RedisError as exc:

            logger.error(
                "Redis dedup write failed: %s",
                exc,
            )

            return False

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self) -> None:

        try:
            self.client.close()
        except Exception:
            pass


# ============================================================
# SINGLETON
# ============================================================

_redis_client: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    """
    Return the shared Redis client.
    """

    global _redis_client

    if _redis_client is None:

        _redis_client = RedisClient()

    return _redis_client


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    print()
    print("=" * 70)
    print("MARKETPULSE REDIS TEST")
    print("=" * 70)

    redis_client = get_redis()

    # --------------------------------------------------------
    # 1. Connection
    # --------------------------------------------------------

    print(
        "\n1. Redis connection..."
    )

    if not redis_client.ping():

        print(
            "❌ Redis connection failed."
        )

        raise SystemExit(1)

    print(
        "✅ Redis connection successful."
    )

    # --------------------------------------------------------
    # 2. Signal storage
    # --------------------------------------------------------

    print(
        "\n2. Testing signal storage..."
    )

    test_signal = {
        "symbol": "INFY",
        "signal": "BUY",
        "signal_score": 0.618,
        "news_sentiment": 0.46,
        "price_momentum": 0.804,
    }

    redis_client.set_signal(
        "INFY",
        test_signal,
    )

    retrieved_signal = (
        redis_client.get_signal(
            "INFY"
        )
    )

    print(
        "Stored signal:"
    )

    print(
        json.dumps(
            retrieved_signal,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 3. News storage
    # --------------------------------------------------------

    print(
        "\n3. Testing news storage..."
    )

    test_news = [
        {
            "id": "test-article-1",
            "title": (
                "Infosys reports strong "
                "quarterly growth"
            ),
            "source": "Test",
        },
        {
            "id": "test-article-2",
            "title": (
                "Infosys shares move higher"
            ),
            "source": "Test",
        },
    ]

    redis_client.set_news(
        "INFY",
        test_news,
    )

    retrieved_news = (
        redis_client.get_news(
            "INFY"
        )
    )

    print(
        json.dumps(
            retrieved_news,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 4. Deduplication
    # --------------------------------------------------------

    print(
        "\n4. Testing news deduplication..."
    )

    test_article = {
        "title": (
            "Infosys reports strong "
            "quarterly growth"
        ),
        "url": (
            "https://example.com/"
            "infosys-test"
        ),
    }

    before = (
        redis_client.is_news_processed(
            test_article
        )
    )

    print(
        f"Before marking processed: "
        f"{before}"
    )

    redis_client.mark_news_processed(
        test_article
    )

    after = (
        redis_client.is_news_processed(
            test_article
        )
    )

    print(
        f"After marking processed:  "
        f"{after}"
    )

    # --------------------------------------------------------
    # 5. Cleanup test keys
    # --------------------------------------------------------

    print(
        "\n5. Cleaning up test keys..."
    )

    redis_client.delete(
        "signal:INFY"
    )

    redis_client.delete(
        "news:INFY"
    )

    # The processed-news key is intentionally
    # not manually deleted here because its
    # normal behavior is TTL-based expiration.

    print(
        "✅ Test keys cleaned up."
    )

    print()
    print("=" * 70)
    print("REDIS TEST COMPLETE")
    print("=" * 70)