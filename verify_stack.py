"""
One-command health check for the streaming pipeline.

    python verify_stack.py                 # 100 events at 20/s, waits up to 90 s
    python verify_stack.py --count 300 --rate 50 --timeout 120

Needs: Redis reachable (REDIS_PORT, default 6391) and a consumer running
(`docker compose --profile stream up -d inference` or `python -m streaming.consumer`).
It replays headlines through the real pipeline and checks every stage. Exit code 0 = all passed.
"""
import argparse
import sys
import time

from streaming.connection import connect
from streaming.replay import load_headlines, replay
from streaming.report import build_report
from streaming.settings import load_settings

results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--rate", type=float, default=20)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--file", default="all-data.csv")
    p.add_argument("--wait-consumer", type=int, default=60, help="seconds to wait for the consumer to attach")
    p.add_argument("--p95-limit-ms", type=float, default=2000, help="fail if e2e p95 is above this")
    a = p.parse_args()

    cfg, r = load_settings(), connect("REPLAY")
    try:
        check("redis reachable", bool(r.ping()))
    except Exception as exc:
        check("redis reachable", False, exc.__class__.__name__)
        return 1

    def live_consumers() -> int:
        """Heartbeat keys (refreshed by every running consumer); XINFO misses idle consumers on older Redis."""
        try:
            beats = len(list(r.scan_iter(f"{cfg.prefix}hb:consumer:*", count=100)))
            groups = [g for g in r.xinfo_groups(cfg.raw) if g["name"] == cfg.group]
            return max(beats, groups[0]["consumers"] if groups else 0)
        except Exception:
            return 0

    consumers, wait_until = live_consumers(), time.time() + a.wait_consumer
    while consumers < 1 and time.time() < wait_until:     # the model can take a while to load after `up`
        time.sleep(1)
        consumers = live_consumers()
    check("a consumer is attached to the stream", consumers >= 1,
          f"{consumers} consumer(s)" if consumers else "none after waiting; start: docker compose --profile stream up -d inference")

    info = replay(r, load_headlines(a.file), "INFY", a.rate, a.count, cfg)
    run = info["run"]
    print(f"      replayed {info['sent']} events in {info['seconds']}s (run {run}); waiting for the consumer...")

    def done():
        scored = sum(1 for _, f in r.xrevrange(cfg.scored, count=a.count * 3 + 200) if f.get("run") == run)
        dead = sum(1 for _, f in r.xrevrange(cfg.dead, count=a.count + 200) if f.get("run") == run)
        return scored, dead

    deadline, scored, dead = time.time() + a.timeout, 0, 0
    while time.time() < deadline:
        scored, dead = done()
        if scored + dead >= a.count:
            break
        time.sleep(0.5)

    check("all events processed", scored + dead == a.count, f"scored={scored} dead={dead} of {a.count}")
    check("nothing dead-lettered", dead == 0, "see: redis-cli XRANGE mp:news.dead - + COUNT 3" if dead else "")
    ids = [f["id"] for _, f in r.xrevrange(cfg.scored, count=a.count * 3 + 200) if f.get("run") == run]
    check("no duplicate results", len(ids) == len(set(ids)), f"{len(ids)} results")

    rep = build_report(r, cfg, last=a.count * 3 + 200, run=run)
    e2e = rep["latency"]["e2e_ms"]
    check("latency recorded", e2e["p95"] is not None)
    if e2e["p95"] is not None:
        check(f"e2e p95 under {a.p95_limit_ms:.0f} ms", e2e["p95"] <= a.p95_limit_ms,
              f"p50={e2e['p50']:.0f} p95={e2e['p95']:.0f} p99={e2e['p99']:.0f} ms, mean batch {rep['latency']['batch_size_mean']}")
    check("no stuck pending messages", (rep["health"].get("pending") or 0) == 0, f"pending={rep['health'].get('pending')}")

    ok = all(results)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())