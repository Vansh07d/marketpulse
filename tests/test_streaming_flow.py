"""Integration tests against a real Redis (skipped if none is reachable). Each test uses its own key prefix."""
import threading
import time
import uuid

import pytest
import redis

from streaming.consumer import Consumer
from streaming.ingestor import poll_once, publish_many
from streaming.replay import replay
from streaming.report import build_report, hops
from streaming.schema import build_event, now_ms
from streaming.settings import load_settings


@pytest.fixture
def r():
    client = redis.Redis(host="localhost", port=6391, decode_responses=True, socket_connect_timeout=1)
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("Redis not available on localhost:6391")
    yield client


@pytest.fixture
def cfg(r):
    prefix = f"test{uuid.uuid4().hex[:8]}:"
    cfg = load_settings(prefix=prefix, batch_max=8, batch_wait_ms=30, block_ms=50, reclaim_idle_ms=100,
                        reclaim_every_s=1, max_deliveries=3)
    yield cfg
    for key in r.scan_iter(f"{prefix}*"):
        r.delete(key)


class Stub:
    """Fake model: deterministic, records batch sizes, can be told to fail."""
    def __init__(self):
        self.batches, self.fail_on, self.fail_all = [], set(), False

    def predict_batch(self, texts):
        self.batches.append(len(texts))
        if self.fail_all or any(t in self.fail_on for t in texts):
            raise RuntimeError("boom")
        return [{"label": "positive", "score": 0.5, "negative": 0.1, "neutral": 0.3, "positive": 0.6} for _ in texts]


def events(n, symbol="INFY", tag="headline"):
    return [build_event({"title": f"{tag} number {i} about earnings"}, symbol, "test") for i in range(n)]


def drain(consumer, rounds=10):
    totals = {"scored": 0, "dead": 0}
    for _ in range(rounds):
        out = consumer.step(first_block_ms=20)
        totals["scored"] += out["scored"]
        totals["dead"] += out["dead"]
    return totals


def pending(r, cfg):
    return r.xpending(cfg.raw, cfg.group)["pending"]


def test_ingest_dedups_atomically_even_with_concurrent_pollers(r, cfg):
    art = [{"title": "Infosys wins a big deal", "source": "x"}, {"title": "INFOSYS wins a  big deal!", "source": "y"}]
    results = []
    threads = [threading.Thread(target=lambda: results.append(poll_once(r, ["INFY"], lambda s: art, cfg)))
               for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert r.xlen(cfg.raw) == 1
    assert sum(c["new"] for c in results) == 1


def test_end_to_end_scores_everything_and_acks(r, cfg):
    publish_many(r, events(20), cfg)
    stub = Stub()
    c = Consumer(r, stub, cfg, name="c1")
    assert drain(c)["scored"] == 20
    assert r.xlen(cfg.scored) == 20 and pending(r, cfg) == 0
    assert max(stub.batches) <= cfg.batch_max
    row = r.xrevrange(cfg.scored, count=1)[0][1]
    assert int(row["enqueued_ms"]) <= int(row["consumed_ms"]) <= int(row["infer_start_ms"]) <= int(row["scored_ms"])
    assert row["model"] == "finbert" and row["label"] == "positive"


def test_microbatch_fills_batches_under_load(r, cfg):
    publish_many(r, events(24), cfg)
    stub = Stub()
    drain(Consumer(r, stub, cfg, name="c1"))
    assert sum(stub.batches) == 24 and stub.batches[0] == cfg.batch_max


def test_invalid_event_goes_to_dead_letter_and_does_not_block(r, cfg):
    r.xadd(cfg.raw, {"v": "1", "title": "x"})                               # malformed
    r.xadd(cfg.raw, {**events(1)[0], "symbol": "bad symbol", "enqueued_ms": str(now_ms())})
    publish_many(r, events(3), cfg)
    out = drain(Consumer(r, Stub(), cfg, name="c1"))
    assert out == {"scored": 3, "dead": 2} and pending(r, cfg) == 0
    assert all(f["reason"].startswith("invalid") for _, f in r.xrange(cfg.dead))


def test_stale_events_are_dropped_not_scored(r, cfg):
    old = {**events(1)[0], "enqueued_ms": str(now_ms() - 3_600_000)}
    r.xadd(cfg.raw, old)
    out = drain(Consumer(r, Stub(), cfg, name="c1"))
    assert out == {"scored": 0, "dead": 1}
    assert r.xrange(cfg.dead)[0][1]["reason"] == "stale"


def test_poison_item_is_isolated_and_others_still_scored(r, cfg):
    evs = events(6)
    stub = Stub()
    stub.fail_on = {evs[2]["title"]}
    publish_many(r, evs, cfg)
    out = drain(Consumer(r, stub, cfg, name="c1"))
    assert out == {"scored": 5, "dead": 1} and pending(r, cfg) == 0
    assert r.xrange(cfg.dead)[0][1]["reason"] == "model_error"


def test_systemic_failure_acks_nothing_then_recovers_after_reclaim(r, cfg):
    publish_many(r, events(5), cfg)
    stub = Stub()
    stub.fail_all = True
    c = Consumer(r, stub, cfg, name="c1")
    assert drain(c, rounds=2)["scored"] == 0
    assert pending(r, cfg) == 5 and r.xlen(cfg.scored) == 0
    stub.fail_all = False
    time.sleep(1.1)                                                           # exceed reclaim_idle_ms AND reclaim_every_s
    assert drain(c, rounds=4)["scored"] == 5 and pending(r, cfg) == 0


def test_systemic_failure_is_capped_by_max_deliveries(r, cfg):
    publish_many(r, events(2), cfg)
    stub = Stub()
    stub.fail_all = True
    c = Consumer(r, stub, cfg, name="c1")
    for _ in range(6):
        c.step(first_block_ms=10)
        time.sleep(1.1)                                                       # lets reclaim fire each round
    assert r.xlen(cfg.dead) == 2 and pending(r, cfg) == 0
    assert all(f["reason"] == "max_deliveries" for _, f in r.xrange(cfg.dead))


def test_crashed_consumer_work_is_taken_over_exactly_once(r, cfg):
    publish_many(r, events(6), cfg)
    dead_consumer = Consumer(r, Stub(), cfg, name="dies")
    assert len(dead_consumer.read_batch(first_block_ms=20)) == 6             # reads, then "crashes" without acking
    time.sleep(0.25)
    survivor = Consumer(r, Stub(), cfg, name="lives")
    assert drain(survivor, rounds=4)["scored"] == 6
    assert r.xlen(cfg.scored) == 6 and pending(r, cfg) == 0
    ids = [f["id"] for _, f in r.xrange(cfg.scored)]
    assert len(set(ids)) == 6                                                 # no duplicates


def test_two_consumers_share_work_without_overlap(r, cfg):
    publish_many(r, events(40), cfg)
    a, b = Consumer(r, Stub(), cfg, name="a"), Consumer(r, Stub(), cfg, name="b")
    for _ in range(6):
        a.step(first_block_ms=10)
        b.step(first_block_ms=10)
    ids = [f["id"] for _, f in r.xrange(cfg.scored)]
    assert len(ids) == 40 and len(set(ids)) == 40


def test_replay_rate_and_report(r, cfg):
    info = replay(r, ["Infosys beats estimates", "Profit falls sharply", "Shares flat after results"],
                  "INFY", rate=100, count=50, cfg=cfg, run="r1")
    assert info["sent"] == 50 and 0.35 < info["seconds"] < 1.5
    drain(Consumer(r, Stub(), cfg, name="c1"))
    rep = build_report(r, cfg, run="r1")
    assert rep["latency"]["events"] == 50
    assert rep["latency"]["e2e_ms"]["p95"] is not None and rep["health"]["pending"] == 0
    assert hops(r.xrevrange(cfg.scored, count=1)[0][1])["e2e_ms"] >= 0