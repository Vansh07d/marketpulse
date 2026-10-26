"""Streaming settings, read from the environment and clamped to safe ranges."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Settings:
    prefix: str = "mp:"                 # namespaces every key; tests use a random prefix
    group: str = "scorers"
    batch_max: int = 32                 # max headlines per forward pass
    batch_wait_ms: int = 50             # how long to wait for a batch to fill
    block_ms: int = 1000                # idle wait for the first message (never 0 = forever)
    max_age_seconds: int = 600          # older events are dropped as stale (0 disables)
    max_deliveries: int = 3             # then dead-letter
    reclaim_idle_ms: int = 30_000       # must exceed the worst-case batch time by a wide margin
    reclaim_every_s: int = 15
    raw_maxlen: int = 50_000
    scored_maxlen: int = 20_000
    dead_maxlen: int = 5_000
    dedup_ttl_s: int = 86_400
    ingest_interval_s: int = 300
    model_tag: str = "finbert"

    @property
    def raw(self) -> str:
        return f"{self.prefix}news.raw"

    @property
    def scored(self) -> str:
        return f"{self.prefix}news.scored"

    @property
    def dead(self) -> str:
        return f"{self.prefix}news.dead"

    def dedup_key(self, event_id: str) -> str:
        return f"{self.prefix}dedup:{event_id}"

    def heartbeat_key(self, name: str) -> str:
        return f"{self.prefix}hb:{name}"


def load_settings(**overrides) -> Settings:
    base = dict(
        prefix=os.getenv("STREAM_PREFIX", "mp:"),
        group=os.getenv("STREAM_GROUP", "scorers"),
        batch_max=_int("STREAM_BATCH_MAX", 32, 1, 256),
        batch_wait_ms=_int("STREAM_BATCH_WAIT_MS", 50, 1, 2000),
        block_ms=_int("STREAM_BLOCK_MS", 1000, 1, 10_000),
        max_age_seconds=_int("STREAM_MAX_AGE_SECONDS", 600, 0, 86_400),
        max_deliveries=_int("STREAM_MAX_DELIVERIES", 3, 1, 20),
        reclaim_idle_ms=_int("STREAM_RECLAIM_IDLE_MS", 30_000, 1, 3_600_000),
        reclaim_every_s=_int("STREAM_RECLAIM_EVERY_S", 15, 1, 600),
        raw_maxlen=_int("STREAM_RAW_MAXLEN", 50_000, 100, 5_000_000),
        scored_maxlen=_int("STREAM_SCORED_MAXLEN", 20_000, 100, 5_000_000),
        dead_maxlen=_int("STREAM_DEAD_MAXLEN", 5_000, 10, 1_000_000),
        dedup_ttl_s=_int("STREAM_DEDUP_TTL_S", 86_400, 60, 30 * 86_400),
        ingest_interval_s=_int("INGEST_INTERVAL_SECONDS", 300, 10, 86_400),
        model_tag=os.getenv("STREAM_MODEL_TAG", "finbert")[:64],
    )
    base.update(overrides)
    return Settings(**base)