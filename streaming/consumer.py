"""
Inference consumer: news.raw -> micro-batch -> model -> news.scored (+ news.dead), at-least-once.

Latency design
  * One blocking read for the first message, then wait at most `batch_wait_ms` for the batch to
    fill (up to `batch_max`). Light load  -> a lone headline waits <= batch_wait_ms.
    Heavy load -> full batches, so throughput scales without raising per-item latency much.
  * Results and acks go out in ONE MULTI/EXEC round trip (also makes "write result + ack" atomic,
    so a crash can never produce a duplicate scored event or an unscored acked one).
  * Redis BLOCK 0 means "forever", so block values are always >= 1 ms.

Failure design
  * Invalid / oversized / unsupported events -> dead-letter, ack (cannot be fixed by retrying).
  * Stale events (older than max_age) -> dead-letter "stale" (backpressure: old news is worth less).
  * Batch fails -> score items one by one. If SOME succeed the failing ones are poison -> dead-letter.
    If ALL fail the problem is systemic (model/GPU) -> nothing is acked, delivery count grows, and
    after `max_deliveries` the messages are dead-lettered so a hot loop can never form.
  * Crashed consumers: pending messages idle longer than `reclaim_idle_ms` are claimed by a live one.

Run:  python -m streaming.consumer
"""
from __future__ import annotations

import logging
import math
import os
import signal
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import redis

from streaming.schema import Event, EventError, now_ms, parse_event
from streaming.settings import Settings, load_settings

logger = logging.getLogger("marketpulse.consumer")
Entry = Tuple[str, Dict[str, str]]


class Scorer(Protocol):
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]: ...


class Consumer:
    def __init__(self, r: Any, scorer: Scorer, cfg: Optional[Settings] = None, name: Optional[str] = None):
        self.r = r
        self.scorer = scorer
        self.cfg = cfg or load_settings()
        self.name = name or f"{socket.gethostname()}-{os.getpid()}"
        self.stop = threading.Event()
        self._last_reclaim = 0.0
        self._last_beat = 0.0
        self.ensure_group()

    # ---------------------------------------------------------------- setup
    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(self.cfg.raw, self.cfg.group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ---------------------------------------------------------------- reading
    def _read(self, count: int, block_ms: int) -> List[Entry]:
        resp = self.r.xreadgroup(self.cfg.group, self.name, {self.cfg.raw: ">"},
                                 count=count, block=max(1, block_ms))
        return list(resp[0][1]) if resp else []

    def read_batch(self, first_block_ms: Optional[int] = None) -> List[Entry]:
        """Micro-batching: block for the first message, then fill up to batch_max within batch_wait_ms."""
        batch = self._read(self.cfg.batch_max, first_block_ms or self.cfg.block_ms)
        if not batch:
            return batch
        deadline = time.monotonic() + self.cfg.batch_wait_ms / 1000
        while len(batch) < self.cfg.batch_max:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms < 1:
                break
            batch += self._read(self.cfg.batch_max - len(batch), remaining_ms)
        return batch

    def reclaim(self) -> List[Entry]:
        """Take over messages a dead/slow consumer left pending; dead-letter ones delivered too often."""
        try:
            pending = self.r.xpending_range(self.cfg.raw, self.cfg.group, min="-", max="+", count=100,
                                            idle=self.cfg.reclaim_idle_ms)
        except redis.ResponseError:
            return []
        claim_ids: List[str] = []
        dead: List[Entry] = []
        for item in pending:
            if item["times_delivered"] >= self.cfg.max_deliveries:
                found = self.r.xrange(self.cfg.raw, item["message_id"], item["message_id"])
                dead += found or [(item["message_id"], {})]
            else:
                claim_ids.append(item["message_id"])
        if dead:
            self._commit([], [(e, "max_deliveries") for e in dead], now_ms())
        if not claim_ids:
            return []
        return list(self.r.xclaim(self.cfg.raw, self.cfg.group, self.name,
                                  min_idle_time=self.cfg.reclaim_idle_ms, message_ids=claim_ids))

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _valid_result(res: Dict[str, Any]) -> bool:
        try:
            return (res.get("label") in ("negative", "neutral", "positive")
                    and all(math.isfinite(float(res[k])) for k in ("score", "negative", "neutral", "positive")))
        except (KeyError, TypeError, ValueError):
            return False

    def _score(self, events: List[Event]) -> Tuple[List[Optional[Dict[str, Any]]], bool]:
        """Returns (results aligned with events, systemic_failure). None marks a poison event."""
        try:
            results = self.scorer.predict_batch([e.title for e in events])
            if len(results) == len(events):
                return [r if self._valid_result(r) else None for r in results], False
            logger.error("model returned %d results for %d inputs", len(results), len(events))
        except Exception:
            logger.exception("batch scoring failed; isolating per item")
        singles: List[Optional[Dict[str, Any]]] = []
        for event in events:
            try:
                res = self.scorer.predict_batch([event.title])[0]
                singles.append(res if self._valid_result(res) else None)
            except Exception:
                singles.append(None)
        return singles, all(s is None for s in singles)

    # ---------------------------------------------------------------- processing
    def process(self, entries: List[Entry], consumed_ms: Optional[int] = None) -> Dict[str, int]:
        consumed = consumed_ms or now_ms()
        stats = {"scored": 0, "dead": 0, "left_pending": 0}
        good: List[Tuple[Entry, Event]] = []
        dead: List[Tuple[Entry, str]] = []
        for entry in entries:
            try:
                event = parse_event(entry[1])
            except EventError as exc:
                dead.append((entry, f"invalid: {exc}"))
                continue
            age_ms = consumed - event.enqueued_ms
            if self.cfg.max_age_seconds and age_ms > self.cfg.max_age_seconds * 1000:
                dead.append((entry, "stale"))
                continue
            good.append((entry, event))

        scored: List[Tuple[Entry, Event, Dict[str, Any]]] = []
        if good:
            infer_start = now_ms()
            results, systemic = self._score([ev for _, ev in good])
            scored_ms = now_ms()
            if systemic:
                stats["left_pending"] = len(good)       # no ack: retried via reclaim, capped by max_deliveries
                logger.error("systemic scoring failure; %d messages left pending", len(good))
                good = []
            for (entry, event), res in zip(good, results):
                if res is None:
                    dead.append((entry, "model_error"))
                else:
                    scored.append((entry, event, res))
            self._commit(scored, dead, consumed, infer_start, scored_ms)
        else:
            self._commit([], dead, consumed)
        stats["scored"], stats["dead"] = len(scored), len(dead)
        return stats

    def _commit(self, scored: List[Tuple[Entry, Event, Dict[str, Any]]], dead: List[Tuple[Entry, str]],
                consumed_ms: int, infer_start_ms: int = 0, scored_ms: int = 0) -> None:
        """Write results + dead letters + acks atomically in one round trip."""
        if not scored and not dead:
            return
        cfg = self.cfg
        pipe = self.r.pipeline(transaction=True)
        for (entry_id, fields), _event, res in scored:
            out = {**fields,
                   "consumed_ms": str(consumed_ms), "infer_start_ms": str(infer_start_ms),
                   "scored_ms": str(scored_ms), "batch_size": str(len(scored)),
                   "label": res["label"], "score": f"{float(res['score']):.6f}",
                   "negative": f"{float(res['negative']):.6f}", "neutral": f"{float(res['neutral']):.6f}",
                   "positive": f"{float(res['positive']):.6f}", "model": cfg.model_tag,
                   "raw_id": entry_id}
            pipe.xadd(cfg.scored, out, maxlen=cfg.scored_maxlen, approximate=True)
            pipe.xack(cfg.raw, cfg.group, entry_id)
        for (entry_id, fields), reason in dead:
            pipe.xadd(cfg.dead, {**{k: str(v)[:512] for k, v in fields.items()}, "reason": reason[:200],
                                 "raw_id": entry_id, "dead_ms": str(now_ms())},
                      maxlen=cfg.dead_maxlen, approximate=True)
            pipe.xack(cfg.raw, cfg.group, entry_id)
        pipe.execute()

    # ---------------------------------------------------------------- loop
    def step(self, first_block_ms: Optional[int] = None) -> Dict[str, int]:
        """One iteration: maybe reclaim, read a micro-batch, process it. Returns summed counts."""
        total = {"scored": 0, "dead": 0, "left_pending": 0}

        def add(stats: Dict[str, int]) -> None:
            for key, value in stats.items():
                total[key] += value

        if time.monotonic() - self._last_reclaim >= self.cfg.reclaim_every_s:
            self._last_reclaim = time.monotonic()
            claimed = self.reclaim()
            if claimed:
                logger.warning("reclaimed %d stale messages", len(claimed))
                add(self.process(claimed))
        entries = self.read_batch(first_block_ms)
        if entries:
            add(self.process(entries))
        return total

    def run(self) -> None:
        logger.info("Consumer %s | batch<=%d wait<=%dms | %s -> %s", self.name, self.cfg.batch_max,
                    self.cfg.batch_wait_ms, self.cfg.raw, self.cfg.scored)
        backoff = 1.0
        while not self.stop.is_set():
            try:
                self.step()
                backoff = 1.0
                if time.monotonic() - self._last_beat > 5:
                    self._last_beat = time.monotonic()
                    self.r.set(self.cfg.heartbeat_key(f"consumer:{self.name}"), now_ms(), ex=30)
            except (redis.ConnectionError, redis.TimeoutError, redis.ResponseError) as exc:
                logger.error("Redis problem (%s); retrying in %.0fs", exc.__class__.__name__, backoff)
                self.stop.wait(backoff)
                backoff = min(30.0, backoff * 2)
        logger.info("Consumer %s stopped (unacked messages stay pending and will be reclaimed).", self.name)


def main() -> None:
    from streaming.connection import connect
    from sentiment_service import get_finbert

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    scorer = get_finbert()
    scorer.predict_batch(["warm-up headline for the model"])      # first call pays one-off setup costs
    consumer = Consumer(connect("CONSUMER"), scorer, load_settings())
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: consumer.stop.set())
    consumer.run()


if __name__ == "__main__":
    main()