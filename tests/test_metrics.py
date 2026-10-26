from streaming.metrics import collect, render_prometheus
from streaming.schema import now_ms
from streaming.settings import load_settings


class FakeR:
    def __init__(self, cfg):
        self.cfg = cfg
        base = {"enqueued_ms": "1000", "consumed_ms": "1100", "infer_start_ms": "1150", "scored_ms": "1400",
                "batch_size": "2", "run": "x"}
        self.streams = {cfg.scored: [("1-0", dict(base)), ("2-0", dict(base))], cfg.raw: [1, 2, 3],
                        cfg.dead: [("1-0", {"reason": 'invalid: bad "quote"'})]}
        self.kv = {cfg.heartbeat_key("consumer:a"): str(now_ms() - 2000)}

    def xrevrange(self, stream, count=0):
        return self.streams[stream][:count]

    def xlen(self, stream):
        return len(self.streams[stream])

    def xinfo_groups(self, stream):
        if stream == self.cfg.raw:
            return [{"name": self.cfg.group, "pending": 0, "lag": 3, "consumers": 1}]
        return [{"name": self.cfg.aggregator_group, "pending": 2, "lag": 0, "consumers": 1}]

    def scan_iter(self, pattern):
        return iter([k for k in self.kv if k.startswith(pattern.rstrip("*"))])

    def get(self, key):
        return self.kv[key]


def test_metrics_text():
    cfg = load_settings(prefix="t:")
    text = render_prometheus(collect(FakeR(cfg), cfg), cfg)
    assert 'mp_stream_length{stream="scored"} 2' in text and 'mp_stream_length{stream="raw"} 3' in text
    assert 'mp_latency_ms{hop="e2e_ms",quantile="0.95"} 400' in text      # 1400 - 1000
    assert 'mp_group_pending{group="aggregators"} 2' in text and 'mp_group_lag{group="scorers"} 3' in text
    assert 'mp_heartbeat_age_seconds{component="consumer:a"}' in text
    assert 'reason="invalid"' in text                                     # reason keeps only the part before ':'
    assert "None" not in text and "nan" not in text.lower()
    assert text.endswith("\n") and all(line.startswith(("#", "mp_")) for line in text.splitlines())


def test_empty_streams_render_without_crashing():
    cfg = load_settings(prefix="t:")
    r = FakeR(cfg)
    r.streams[cfg.scored], r.streams[cfg.dead], r.kv = [], [], {}
    text = render_prometheus(collect(r, cfg), cfg)
    assert "mp_up 1" in text and "mp_latency_ms" not in text