"""
Ingestor: provider news -> validate/sanitise -> atomic de-dup + XADD to news.raw.

De-dup and enqueue happen in ONE Lua script (SET NX + XADD), so a duplicate can never slip in
between the check and the write, and a failed XADD never leaves a dedup key behind that would
silently swallow the story. News fetching is the only slow part; symbols are fetched in a small
thread pool and everything is written to Redis in one pipeline.

Run:  python -m streaming.ingestor            (loop)
      python -m streaming.ingestor --once     (single poll, useful on free API tiers)
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional

from streaming.schema import EventError, build_event, check_published, now_ms
from streaming.settings import Settings, load_settings

logger = logging.getLogger("marketpulse.ingestor")

_PUBLISH_LUA = """
if redis.call('SET', KEYS[1], '1', 'NX', 'EX', tonumber(ARGV[1])) then
    return redis.call('XADD', KEYS[2], 'MAXLEN', '~', tonumber(ARGV[2]), '*', unpack(ARGV, 3))
end
return false
"""

_stop = threading.Event()


def _flatten(fields: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for key, value in fields.items():
        out += [key, value]
    return out


def publish(r: Any, event: Dict[str, str], cfg: Settings, dedup: bool = True) -> Any:
    """Enqueue one event. Returns the stream id, or None if it was a duplicate. (Works on a pipeline too.)"""
    event = {**event, "enqueued_ms": str(now_ms())}
    if not dedup:
        return r.xadd(cfg.raw, event, maxlen=cfg.raw_maxlen, approximate=True)
    script = r.register_script(_PUBLISH_LUA)
    return script(keys=[cfg.dedup_key(event["id"]), cfg.raw],
                  args=[cfg.dedup_ttl_s, cfg.raw_maxlen, *_flatten(event)])


def publish_many(r: Any, events: Iterable[Dict[str, str]], cfg: Settings, dedup: bool = True) -> List[Optional[str]]:
    """One network round trip for the whole batch."""
    pipe = r.pipeline(transaction=False)
    if dedup:
        script = r.register_script(_PUBLISH_LUA)
        for event in events:
            event = {**event, "enqueued_ms": str(now_ms())}
            script(keys=[cfg.dedup_key(event["id"]), cfg.raw],
                   args=[cfg.dedup_ttl_s, cfg.raw_maxlen, *_flatten(event)], client=pipe)
    else:
        for event in events:
            pipe.xadd(cfg.raw, {**event, "enqueued_ms": str(now_ms())}, maxlen=cfg.raw_maxlen, approximate=True)
    return pipe.execute()


def poll_once(r: Any, symbols: List[str], fetch: Callable[[str], List[Dict[str, Any]]],
              cfg: Settings, origin: str = "news_service") -> Dict[str, int]:
    """Fetch every symbol, enqueue new headlines. Returns counts for logging/metrics."""
    counts = {"fetched": 0, "new": 0, "duplicate": 0, "invalid": 0, "fetch_errors": 0,
              "stale": 0, "bad_timestamp": 0, "no_timestamp": 0}

    def _fetch(symbol: str):
        try:
            return symbol, fetch(symbol), now_ms()
        except Exception:
            logger.exception("%s | fetch failed", symbol)
            return symbol, None, now_ms()

    events: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(3, len(symbols)))) as pool:
        for symbol, articles, fetched_ms in pool.map(_fetch, symbols):
            if articles is None:
                counts["fetch_errors"] += 1
                continue
            counts["fetched"] += len(articles)
            for article in articles:
                try:
                    event = build_event(article, symbol, origin, fetched_ms)
                except EventError as exc:
                    counts["invalid"] += 1
                    logger.warning("%s | dropped invalid article: %s", symbol, exc)
                    continue
                verdict = check_published(int(event["published_ms"]), now_ms(),
                                          cfg.max_article_age_hours, cfg.future_skew_s)
                if verdict == "old":
                    counts["stale"] += 1          # old news is not a signal
                    continue
                if verdict == "future":
                    counts["bad_timestamp"] += 1  # provider clock/format bug; do not trust it
                    continue
                if verdict == "missing":
                    counts["no_timestamp"] += 1   # kept, but the aggregator treats it as published at enqueue time
                events.append(event)

    if events:
        results = publish_many(r, events, cfg)
        counts["new"] = sum(1 for res in results if res)
        counts["duplicate"] = len(results) - counts["new"]
    return counts


def _symbols() -> List[str]:
    raw = os.getenv("WATCHLIST") or os.getenv("DEFAULT_SYMBOL", "INFY")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def main() -> None:
    from news_service import fetch_company_news
    from streaming.connection import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from streaming.preflight import require
    require("ingestor")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="poll one time and exit")
    args = parser.parse_args()

    cfg = load_settings()
    r = connect("INGESTOR")
    symbols = _symbols()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: _stop.set())
    logger.info("Ingestor | symbols=%s | interval=%ss | stream=%s", symbols, cfg.ingest_interval_s, cfg.raw)

    while not _stop.is_set():
        counts = poll_once(r, symbols, fetch_company_news, cfg)
        r.set(cfg.heartbeat_key("ingestor"), now_ms(), ex=cfg.ingest_interval_s * 3)
        logger.info("poll done: %s", counts)
        if args.once:
            break
        _stop.wait(cfg.ingest_interval_s * random.uniform(0.9, 1.1))   # jitter avoids sync with provider windows
    logger.info("Ingestor stopped.")


if __name__ == "__main__":
    main()