"""
MarketPulse - Latency tracking

Every new article is stamped as it moves through the pipeline:

    published_at -> ingested_at -> scored_at -> signalled_at

One record per article is kept in Redis (latest 500 per symbol) and summarised
as p50 / p95. Note: end_to_end includes the provider's own delay and the poll
interval, so it measures what a user actually waits, not just our code.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("marketpulse.latency")

KEY_PREFIX = "latency:"
MAX_RECORDS = 500
TTL_SECONDS = 60 * 60 * 24 * 30
METRICS = ("ingest_lag_s", "inference_s", "signal_s", "end_to_end_s")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _seconds(start: Any, end: Any) -> Optional[float]:
    a, b = parse_ts(start), parse_ts(end)
    return None if a is None or b is None else max(0.0, (b - a).total_seconds())


def build_record(article: Dict[str, Any], signalled_at: str) -> Dict[str, Any]:
    published, ingested = article.get("published_at"), article.get("ingested_at")
    scored = article.get("scored_at")
    return {
        "id": article.get("id"),
        "title": str(article.get("title", ""))[:80],
        "published_at": published,
        "ingested_at": ingested,
        "scored_at": scored,
        "signalled_at": signalled_at,
        "ingest_lag_s": _seconds(published, ingested),    # provider delay + wait for our next poll
        "inference_s": _seconds(ingested, scored),        # FinBERT
        "signal_s": _seconds(scored, signalled_at),       # market data + signal + Redis
        "end_to_end_s": _seconds(published, signalled_at),
    }


def record_latencies(redis_client: Any, symbol: str, articles: List[Dict[str, Any]],
                     signalled_at: str) -> List[Dict[str, Any]]:
    records = [build_record(a, signalled_at) for a in articles]
    if not records:
        return []
    key = f"{KEY_PREFIX}{symbol.upper()}"
    try:
        pipe = redis_client.client.pipeline()
        for record in records:
            pipe.lpush(key, json.dumps(record))
        pipe.ltrim(key, 0, MAX_RECORDS - 1)
        pipe.expire(key, TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("Could not store latency records for %s: %s", symbol, exc)
    return records


def get_latencies(redis_client: Any, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Newest first."""
    try:
        rows = redis_client.client.lrange(f"{KEY_PREFIX}{symbol.upper()}", 0, max(0, limit - 1))
    except Exception as exc:
        logger.warning("Could not read latency records for %s: %s", symbol, exc)
        return []
    out = []
    for row in rows:
        try:
            item = json.loads(row)
        except (TypeError, ValueError):
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for metric in METRICS:
        values = [r[metric] for r in records if isinstance(r.get(metric), (int, float))]
        summary[metric] = {"count": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}
    return summary


def format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value < 90:
        return f"{value:.1f}s"
    if value < 5400:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.1f} h"