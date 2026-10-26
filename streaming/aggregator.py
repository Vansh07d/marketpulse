"""
Aggregator: news.scored -> durable SQLite + the Redis state the dashboard already reads.

This is what connects the streaming pipeline to signals. For every symbol it:
  1. stores scored headlines in SQLite (idempotent, so redelivery is harmless),
  2. publishes the latest articles to Redis (same shape the legacy worker writes: set_news),
  3. combines news + price + benchmark with signal_service.generate_signal and publishes it
     (set_signal, so signal history and the dashboard keep working), and stores it in SQLite.

Delivery: at-least-once. A message is acked only after its row is committed to SQLite. The consumer
name is fixed (AGGREGATOR_NAME, default "aggregator-1"), so after a restart it re-reads its own
unacked messages first. Run ONE aggregator instance (signals are per-symbol state).

Events with origin in AGGREGATOR_IGNORE_ORIGINS (default "replay,test") are acked and skipped, so
load tests never put fake headlines on the dashboard or in the research store.

Run:  python -m streaming.aggregator
"""
from __future__ import annotations

import logging
import os
import signal
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from signal_service import generate_signal
from streaming.schema import now_ms
from streaming.settings import Settings, load_settings
from streaming.store import Store

logger = logging.getLogger("marketpulse.aggregator")
Entry = Tuple[str, Dict[str, str]]
MAX_ARTICLES = 20


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class Aggregator:
    def __init__(self, r: Any, store: Store, rc: Any,
                 market_fn: Callable[[str], Optional[Dict[str, Any]]],
                 benchmark_fn: Callable[[str], Optional[Dict[str, Any]]] = lambda symbol: None,
                 cfg: Optional[Settings] = None, name: Optional[str] = None,
                 ignore_origins: Optional[set] = None):
        self.r, self.store, self.rc = r, store, rc
        self.market_fn, self.benchmark_fn = market_fn, benchmark_fn
        self.cfg = cfg or load_settings()
        self.name = name or os.getenv("AGGREGATOR_NAME", "aggregator-1")
        self.ignore = ignore_origins if ignore_origins is not None else {
            o.strip() for o in os.getenv("AGGREGATOR_IGNORE_ORIGINS", "replay,test").split(",") if o.strip()}
        self.stop = threading.Event()
        self._pending_done = False
        self._last_refresh: Dict[str, float] = {}
        self._last_periodic = 0.0
        self.ensure_group()

    def ensure_group(self) -> None:
        import redis
        try:
            self.r.xgroup_create(self.cfg.scored, self.cfg.aggregator_group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------ rows / articles
    @staticmethod
    def _row(fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            published, enqueued = _int(fields.get("published_ms")), _int(fields.get("enqueued_ms"))
            return {
                "event_id": fields["id"], "symbol": fields["symbol"].upper(), "title": fields["title"],
                "url": fields.get("url", ""), "source": fields.get("source", ""),
                "published_ms": published, "enqueued_ms": enqueued, "scored_ms": _int(fields.get("scored_ms")),
                "event_ms": published or enqueued or now_ms(),     # unknown publish time -> enqueue time
                "label": fields["label"], "score": float(fields["score"]),
                "negative": float(fields["negative"]), "neutral": float(fields["neutral"]),
                "positive": float(fields["positive"]), "model": fields.get("model", ""),
            }
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _article(row: Dict[str, Any]) -> Dict[str, Any]:
        """Same shape as the legacy worker's stored article, which app2.py renders."""
        return {
            "id": row["event_id"], "title": row["title"], "url": row["url"], "source": row["source"],
            "published_at": _iso(row["event_ms"]), "scored_at": _iso(row["scored_ms"] or row["event_ms"]),
            "sentiment": {"label": row["label"], "score": row["score"], "negative": row["negative"],
                          "neutral": row["neutral"], "positive": row["positive"]},
        }

    # ------------------------------------------------------------------ signal
    def refresh(self, symbol: str, force: bool = False) -> Optional[Dict[str, Any]]:
        """Recompute and publish one symbol's news state + signal. Never raises."""
        mono = time.monotonic()
        if not force and mono - self._last_refresh.get(symbol, -1e9) < self.cfg.signal_min_interval_s:
            return None
        self._last_refresh[symbol] = mono
        try:
            since = now_ms() - self.cfg.max_article_age_hours * 3_600_000
            rows = self.store.recent(symbol, since, MAX_ARTICLES)
            if not rows:
                return None
            articles = [self._article(r) for r in rows]
            self.rc.set_news(symbol, articles)
            market = self.market_fn(symbol)
            if not market:
                logger.warning("%s | no market data; news state updated, signal skipped", symbol)
                return None
            sentiments = [{**a["sentiment"], "published_at": a["published_at"]} for a in articles]
            try:
                benchmark = self.benchmark_fn(symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s | benchmark failed (%s); continuing without it", symbol, exc)
                benchmark = None
            result = generate_signal(symbol=symbol, sentiment_results=sentiments,
                                     market_data=market, benchmark_data=benchmark)
            result["pipeline"] = "stream"
            self.rc.set_signal(symbol, result)
            self.store.add_signal(result, len(rows), now_ms())
            return result
        except Exception:  # noqa: BLE001
            logger.exception("%s | refresh failed", symbol)
            return None

    def refresh_active(self) -> None:
        since = now_ms() - self.cfg.max_article_age_hours * 3_600_000
        for symbol in self.store.active_symbols(since):
            self.refresh(symbol, force=True)

    # ------------------------------------------------------------------ processing
    def handle(self, entries: List[Entry]) -> Dict[str, int]:
        """Store, publish, ack. If the SQLite write fails nothing is acked (redelivery is safe)."""
        stats = {"stored": 0, "skipped": 0, "bad": 0}
        rows: List[Dict[str, Any]] = []
        for _id, fields in entries:
            if not fields:
                stats["bad"] += 1
            elif fields.get("origin") in self.ignore:
                stats["skipped"] += 1
            else:
                row = self._row(fields)
                if row is None:
                    stats["bad"] += 1
                else:
                    rows.append(row)
        if rows:
            stats["stored"] = self.store.add_scored(rows)
        if entries:
            self.r.xack(self.cfg.scored, self.cfg.aggregator_group, *[e[0] for e in entries])
        for symbol in sorted({r["symbol"] for r in rows}):
            self.refresh(symbol)
        return stats

    def _read(self, block_ms: int) -> List[Entry]:
        start = "0" if not self._pending_done else ">"
        resp = self.r.xreadgroup(self.cfg.aggregator_group, self.name, {self.cfg.scored: start},
                                 count=100, block=max(1, block_ms) if start == ">" else None)
        entries = list(resp[0][1]) if resp else []
        if start == "0" and not entries:
            self._pending_done = True        # own backlog drained; switch to new messages
        return entries

    def step(self, block_ms: int = 1000) -> Dict[str, int]:
        entries = self._read(block_ms)
        stats = self.handle(entries) if entries else {"stored": 0, "skipped": 0, "bad": 0}
        if time.monotonic() - self._last_periodic >= self.cfg.signal_refresh_s:
            self._last_periodic = time.monotonic()
            self.refresh_active()
        return stats

    def run(self) -> None:
        import redis
        logger.info("Aggregator %s | %s -> sqlite(%s) + redis signals", self.name, self.cfg.scored,
                    self.cfg.db_path)
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self.step()
                backoff = 1.0
            except (redis.ConnectionError, redis.TimeoutError, redis.ResponseError) as exc:
                logger.error("Redis problem (%s); retrying in %.0fs", exc.__class__.__name__, backoff)
                self.stop.wait(backoff)
                backoff = min(30.0, backoff * 2)
            except sqlite3.Error:
                logger.exception("SQLite write failed; messages stay unacked and will be re-read")
                self.stop.wait(5)
        logger.info("Aggregator %s stopped.", self.name)


def main() -> None:
    from market_service import INDIAN_SYMBOLS, fetch_market_data, fetch_nifty_benchmark
    from redis_client import get_redis
    from streaming.connection import connect
    from streaming.benchmark import fetch_us_benchmark
    from streaming.preflight import require

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    require("aggregator")
    cfg = load_settings()

    def benchmark(symbol: str) -> Optional[Dict[str, Any]]:
        return fetch_nifty_benchmark() if symbol in INDIAN_SYMBOLS else fetch_us_benchmark()

    agg = Aggregator(connect("CONSUMER"), Store(cfg.db_path), get_redis(), fetch_market_data, benchmark, cfg)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: agg.stop.set())
    agg.run()


if __name__ == "__main__":
    main()