"""
Latency and health report for the streaming pipeline.

    python -m streaming.report                  # last 2000 scored events
    python -m streaming.report --run ab12cd34   # only one replay run
    python -m streaming.report --json

Hops (all milliseconds, measured from epoch-ms stamps written by each stage):
    queue_ms   enqueued -> consumed   time waiting in the stream (backlog / consumer speed)
    infer_ms   infer_start -> scored  model time of the batch the headline was in
    e2e_ms     enqueued -> scored     what our pipeline adds (the number to quote)
    source_ms  published -> fetched   provider delay + our poll wait (0 for replay; not under our control)
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from latency import percentile
from streaming.settings import Settings, load_settings


def _num(fields: Dict[str, str], key: str) -> Optional[int]:
    try:
        return int(fields[key])
    except (KeyError, ValueError):
        return None


def hops(fields: Dict[str, str]) -> Dict[str, Optional[float]]:
    enq, con = _num(fields, "enqueued_ms"), _num(fields, "consumed_ms")
    ist, sc = _num(fields, "infer_start_ms"), _num(fields, "scored_ms")
    pub, fet = _num(fields, "published_ms"), _num(fields, "fetched_ms")
    diff = lambda a, b: None if a is None or b is None else float(max(0, b - a))  # noqa: E731
    return {
        "queue_ms": diff(enq, con), "infer_ms": diff(ist, sc), "e2e_ms": diff(enq, sc),
        "source_ms": diff(pub, fet) if pub else None,
    }


def build_report(r: Any, cfg: Settings, last: int = 2000, run: str = "") -> Dict[str, Any]:
    entries = r.xrevrange(cfg.scored, count=last)
    rows = [f for _, f in entries if not run or f.get("run") == run]
    stats: Dict[str, Any] = {"events": len(rows)}
    for metric in ("queue_ms", "infer_ms", "e2e_ms", "source_ms"):
        values = [h[metric] for h in (hops(f) for f in rows) if h[metric] is not None]
        stats[metric] = {"p50": percentile(values, 50), "p95": percentile(values, 95),
                         "p99": percentile(values, 99), "max": max(values) if values else None}
    scored_ms = [int(f["scored_ms"]) for f in rows if "scored_ms" in f]
    enq_ms = [int(f["enqueued_ms"]) for f in rows if "enqueued_ms" in f]
    if len(rows) > 1 and scored_ms and enq_ms:
        span = max(scored_ms) - min(enq_ms)
        stats["throughput_per_s"] = round(len(rows) / (span / 1000), 1) if span > 0 else None
    sizes = [int(f["batch_size"]) for f in rows if "batch_size" in f]
    stats["batch_size_mean"] = round(sum(sizes) / len(sizes), 1) if sizes else None

    health: Dict[str, Any] = {"raw_len": r.xlen(cfg.raw), "scored_len": r.xlen(cfg.scored),
                              "dead_len": r.xlen(cfg.dead)}
    try:
        for group in r.xinfo_groups(cfg.raw):
            if group["name"] == cfg.group:
                health.update(pending=group.get("pending"), lag=group.get("lag"), consumers=group.get("consumers"))
    except Exception:
        pass
    reasons: Dict[str, int] = {}
    for _, f in r.xrevrange(cfg.dead, count=500):
        key = str(f.get("reason", "?")).split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    health["dead_reasons"] = reasons
    return {"latency": stats, "health": health}


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:,.0f}"


def render(rep: Dict[str, Any]) -> str:
    lat, h = rep["latency"], rep["health"]
    lines = [f"events: {lat['events']}   throughput: {lat.get('throughput_per_s', 'n/a')}/s   "
             f"mean batch: {lat.get('batch_size_mean', 'n/a')}", "",
             f"{'hop (ms)':<12}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"]
    for m in ("queue_ms", "infer_ms", "e2e_ms", "source_ms"):
        s = lat[m]
        lines.append(f"{m:<12}{_fmt(s['p50']):>9}{_fmt(s['p95']):>9}{_fmt(s['p99']):>9}{_fmt(s['max']):>9}")
    lines += ["", f"streams: raw={h['raw_len']} scored={h['scored_len']} dead={h['dead_len']}   "
              f"pending={h.get('pending')} lag={h.get('lag')} consumers={h.get('consumers')}",
              f"dead-letter reasons: {h['dead_reasons'] or 'none'}"]
    return "\n".join(lines)


def main() -> None:
    from streaming.connection import connect

    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=2000)
    parser.add_argument("--run", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rep = build_report(connect("REPORT"), load_settings(), args.last, args.run)
    print(json.dumps(rep, indent=2) if args.json else render(rep))


if __name__ == "__main__":
    main()