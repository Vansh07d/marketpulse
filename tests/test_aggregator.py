"""Aggregator + durable store: no Redis or model needed (fake Redis, fake news/signal sinks)."""
import pytest

from streaming.aggregator import Aggregator
from streaming.schema import check_published, now_ms
from streaming.settings import load_settings
from streaming.store import Store


class FakeRedis:
    def __init__(self):
        self.acked = []

    def xgroup_create(self, *a, **k):
        return True

    def xack(self, stream, group, *ids):
        self.acked += list(ids)


class FakeRC:
    """Stands in for redis_client.RedisClient (the object app2.py reads)."""
    def __init__(self):
        self.news, self.signals = {}, {}

    def set_news(self, symbol, articles, ttl=1800):
        self.news[symbol] = articles

    def set_signal(self, symbol, signal, ttl=3600):
        self.signals[symbol] = signal


def fields(i=1, symbol="INFY", label="positive", score=0.8, origin="news_service", **extra):
    ts = now_ms()
    base = {"v": "1", "id": f"{i:032x}", "symbol": symbol, "title": f"Headline number {i}", "url": "https://x.test/a",
            "source": "wire", "origin": origin, "published_ms": str(ts - 60_000), "enqueued_ms": str(ts - 1000),
            "scored_ms": str(ts), "label": label, "score": str(score), "negative": "0.05",
            "neutral": "0.10", "positive": "0.85", "model": "finbert"}
    base.update(extra)
    return base


def make(market=True):
    r, rc, store = FakeRedis(), FakeRC(), Store(":memory:")
    cfg = load_settings(signal_min_interval_s=0)
    mkt = (lambda s: {"price": 100.0, "change_percent": 2.0, "volume": 10}) if market else (lambda s: None)
    return Aggregator(r, store, rc, mkt, cfg=cfg, name="t", ignore_origins={"replay", "test"}), r, rc, store


def test_stores_publishes_and_acks():
    agg, r, rc, store = make()
    stats = agg.handle([("1-0", fields(1)), ("2-0", fields(2))])
    assert stats["stored"] == 2 and r.acked == ["1-0", "2-0"]
    assert store.counts() == {"scored": 2, "signals": 1}
    art = rc.news["INFY"][0]
    assert art["sentiment"]["label"] == "positive" and "published_at" in art and art["title"]
    sig = rc.signals["INFY"]
    assert sig["signal"] in {"BUY", "HOLD", "SELL"} and sig["pipeline"] == "stream" and "market_data" in sig


def test_redelivery_is_idempotent():
    agg, r, rc, store = make()
    agg.handle([("1-0", fields(1))])
    agg.handle([("1-0", fields(1))])
    assert store.counts()["scored"] == 1


def test_replay_and_test_origins_are_skipped_but_acked():
    agg, r, rc, store = make()
    stats = agg.handle([("1-0", fields(1, origin="replay"))])
    assert stats["skipped"] == 1 and r.acked == ["1-0"]
    assert store.counts()["scored"] == 0 and rc.news == {} and rc.signals == {}


def test_malformed_entries_are_acked_not_looped():
    agg, r, rc, store = make()
    bad = fields(1)
    del bad["label"]
    stats = agg.handle([("1-0", bad), ("2-0", {})])
    assert stats["bad"] == 2 and r.acked == ["1-0", "2-0"] and store.counts()["scored"] == 0


def test_no_market_data_updates_news_but_not_signal():
    agg, r, rc, store = make(market=False)
    agg.handle([("1-0", fields(1))])
    assert "INFY" in rc.news and rc.signals == {} and store.counts()["signals"] == 0


def test_sqlite_failure_leaves_messages_unacked():
    agg, r, rc, store = make()
    store.close()
    with pytest.raises(Exception):
        agg.handle([("1-0", fields(1))])
    assert r.acked == []


def test_missing_publish_time_falls_back_to_enqueue_time():
    row = Aggregator._row(fields(1, published_ms="0"))
    assert row["published_ms"] == 0 and row["event_ms"] == row["enqueued_ms"]


def test_signal_dedupe_in_store():
    store = Store(":memory:")
    sig = {"symbol": "INFY", "signal": "BUY", "signal_score": 0.5}
    assert store.add_signal(sig, 3, 1_000_000) is True
    assert store.add_signal(sig, 3, 1_000_500) is False           # unchanged and recent
    assert store.add_signal({**sig, "signal_score": 0.6}, 3, 1_001_000) is True


def test_check_published_verdicts():
    now = 1_700_000_000_000
    assert check_published(0, now, 24) == "missing"
    assert check_published(now + 3_600_000, now, 24) == "future"
    assert check_published(now - 48 * 3_600_000, now, 24) == "old"
    assert check_published(now - 3_600_000, now, 24) == "ok"


def test_us_benchmark_change_cache_and_failure():
    from streaming import benchmark as b
    assert b.pct_change([100.0, 101.0, 102.0]) == pytest.approx((102 / 101 - 1) * 100)
    assert b.pct_change([float("nan"), 5.0]) is None and b.pct_change([]) is None
    b._cache.update(time=0.0, value=None)
    calls = []
    ok = lambda t: calls.append(t) or [100.0, 102.0]
    assert b.fetch_us_benchmark(ok, now=lambda: 1000.0)["change_percent"] == pytest.approx(2.0)
    b.fetch_us_benchmark(ok, now=lambda: 1010.0)
    assert len(calls) == 1                                        # served from cache
    boom = lambda t: (_ for _ in ()).throw(RuntimeError("yahoo down"))
    assert b.fetch_us_benchmark(boom, now=lambda: 9999.0)["change_percent"] == pytest.approx(2.0)  # last good