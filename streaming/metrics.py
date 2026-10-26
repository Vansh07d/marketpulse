"""
Prometheus-style metrics + health endpoint for the streaming pipeline (stdlib only, no extra deps).

    python -m streaming.metrics            # serves on :9108  (METRICS_PORT / METRICS_HOST)
    GET /metrics   gauges: stream lengths, group pending/lag/consumers, latency percentiles,
                   dead-letter reasons, heartbeat ages
    GET /healthz   200 if Redis answers, 503 otherwise

Latency comes from the last METRICS_WINDOW scored events (default 500), so it is a rolling view.
Results are cached for a few seconds, so scraping cannot load Redis.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from streaming.report import build_report
from streaming.schema import now_ms
from streaming.settings import Settings, load_settings

logger = logging.getLogger("marketpulse.metrics")
CACHE_SECONDS = 5.0


def collect(r: Any, cfg: Settings, last: int = 500) -> Dict[str, Any]:
    rep = build_report(r, cfg, last)
    agg: Dict[str, Any] = {}
    try:
        for g in r.xinfo_groups(cfg.scored):
            if g["name"] == cfg.aggregator_group:
                agg = {"pending": g.get("pending"), "lag": g.get("lag"), "consumers": g.get("consumers")}
    except Exception:  # noqa: BLE001 - group may not exist yet
        pass
    beats: Dict[str, float] = {}
    prefix, now = cfg.heartbeat_key(""), now_ms()
    for key in r.scan_iter(f"{prefix}*"):
        try:
            beats[key[len(prefix):]] = max(0.0, (now - int(r.get(key))) / 1000)
        except (TypeError, ValueError):
            continue
    return {"report": rep, "aggregator": agg, "heartbeats": beats}


def _esc(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render_prometheus(data: Dict[str, Any], cfg: Settings) -> str:
    rep, lat, health = data["report"], data["report"]["latency"], data["report"]["health"]
    out: List[str] = []

    def gauge(name: str, help_: str, samples: List[Tuple[Dict[str, Any], Optional[float]]]) -> None:
        rows = [(lbl, v) for lbl, v in samples if v is not None]
        if not rows:
            return
        out.append(f"# HELP {name} {help_}")
        out.append(f"# TYPE {name} gauge")
        for lbl, v in rows:
            labels = ",".join(f'{k}="{_esc(x)}"' for k, x in lbl.items())
            out.append(f"{name}{{{labels}}} {float(v):g}" if labels else f"{name} {float(v):g}")

    gauge("mp_up", "1 if the metrics exporter could read Redis", [({}, 1)])
    gauge("mp_stream_length", "Entries currently in each stream",
          [({"stream": "raw"}, health.get("raw_len")), ({"stream": "scored"}, health.get("scored_len")),
           ({"stream": "dead"}, health.get("dead_len"))])
    agg = data.get("aggregator") or {}
    gauge("mp_group_pending", "Delivered but unacked messages per consumer group",
          [({"group": cfg.group}, health.get("pending")), ({"group": cfg.aggregator_group}, agg.get("pending"))])
    gauge("mp_group_lag", "Messages not yet delivered to the group",
          [({"group": cfg.group}, health.get("lag")), ({"group": cfg.aggregator_group}, agg.get("lag"))])
    gauge("mp_group_consumers", "Consumers registered in the group",
          [({"group": cfg.group}, health.get("consumers")), ({"group": cfg.aggregator_group}, agg.get("consumers"))])
    samples: List[Tuple[Dict[str, Any], Optional[float]]] = []
    for hop in ("queue_ms", "infer_ms", "e2e_ms"):
        for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
            samples.append(({"hop": hop, "quantile": q}, lat[hop][key]))
    gauge("mp_latency_ms", "Rolling latency per hop over the recent scored window", samples)
    gauge("mp_latency_window_events", "Scored events in the latency window", [({}, lat["events"])])
    gauge("mp_batch_size_mean", "Mean inference batch size in the window", [({}, lat.get("batch_size_mean"))])
    gauge("mp_throughput_per_second", "Scored events per second over the window", [({}, lat.get("throughput_per_s"))])
    gauge("mp_dead_letters_by_reason", "Recent dead letters by reason",
          [({"reason": k}, v) for k, v in (health.get("dead_reasons") or {}).items()])
    gauge("mp_heartbeat_age_seconds", "Seconds since each component last reported",
          [({"component": k}, v) for k, v in sorted((data.get("heartbeats") or {}).items())])
    return "\n".join(out) + "\n"


class _Cache:
    def __init__(self, r: Any, cfg: Settings, window: int):
        self.r, self.cfg, self.window = r, cfg, window
        self.lock, self.at, self.text = threading.Lock(), 0.0, ""

    def get(self) -> str:
        with self.lock:
            if time.monotonic() - self.at > CACHE_SECONDS or not self.text:
                self.text = render_prometheus(collect(self.r, self.cfg, self.window), self.cfg)
                self.at = time.monotonic()
            return self.text


def make_handler(cache: _Cache):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8") -> None:
            raw = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path.split("?")[0] == "/metrics":
                    self._send(200, cache.get(), "text/plain; version=0.0.4; charset=utf-8")
                elif self.path.split("?")[0] == "/healthz":
                    cache.r.ping()
                    self._send(200, "ok\n")
                else:
                    self._send(404, "not found\n")
            except Exception as exc:  # noqa: BLE001 - never crash the server on a Redis hiccup
                logger.warning("request failed: %s", exc.__class__.__name__)
                self._send(503, "redis unavailable\n")

        def log_message(self, *args: Any) -> None:     # keep the container log quiet
            return

    return Handler


def main() -> None:
    from streaming.connection import connect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_settings()
    cache = _Cache(connect("REPORT"), cfg, int(os.getenv("METRICS_WINDOW", "500")))
    host, port = os.getenv("METRICS_HOST", "0.0.0.0"), int(os.getenv("METRICS_PORT", "9108"))
    logger.info("metrics on http://%s:%d/metrics", host, port)
    ThreadingHTTPServer((host, port), make_handler(cache)).serve_forever()


if __name__ == "__main__":
    main()