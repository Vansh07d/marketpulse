"""
Failure drills against the RUNNING stack. Proves (and records) what happens when parts die.

    docker compose --profile stream up -d --build redis ingestor inference aggregator
    python tools/drill.py                    # all three drills, about 6 minutes
    python tools/drill.py --only inference   # inference | aggregator | redis

  inference   kill the model consumer mid-replay; another start must reclaim the stuck messages
  aggregator  kill the aggregator mid-batch; every row must land in SQLite exactly once, nothing left pending
  redis       restart Redis; the stream must survive (AOF) and the pipeline must work again afterwards

Replay and drill events never reach the dashboard (origin=replay) or the research store (the
aggregator drill uses its own fake symbol ZZDRILL and removes it afterwards).
Results are printed and written to drill_results.md (paste them into the README).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DC = ["docker", "compose", "--profile", "stream"]
EXPECTED = {"redis", "ingestor", "inference", "aggregator"}
RESULTS = []


def run(*cmd, timeout=180):
    return subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def running_services():
    return set(run(*DC, "ps", "--status", "running", "--services").stdout.split())


def verify_proc(count, rate, timeout, p95_limit):
    return subprocess.Popen([sys.executable, "verify_stack.py", "--count", str(count), "--rate", str(rate),
                             "--timeout", str(timeout), "--p95-limit-ms", str(p95_limit), "--wait-consumer", "120"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def p95_of(output):
    m = re.search(r"e2e p95 under \d+ ms\s+\((.*?)\)", output)
    return m.group(1) if m else "n/a"


def record(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def drill_inference():
    p = verify_proc(300, 10, 240, 600_000)      # latency is expected to spike during the outage; not judged here
    time.sleep(6)
    t0 = time.time()
    run(*DC, "kill", "inference")
    time.sleep(1)
    run(*DC, "start", "inference")
    out, _ = p.communicate(timeout=300)
    ok = p.returncode == 0
    record("inference killed mid-replay (300 events at 10/s)", ok,
           f"all checks {'passed' if ok else 'FAILED'}; zero loss/duplicates/dead; finished {time.time() - t0:.0f}s after the kill; "
           f"{p95_of(out)}" + ("" if ok else "\n" + out[-800:]))


def drill_aggregator():
    from streaming.connection import connect
    from streaming.settings import load_settings
    cfg, r, run_id = load_settings(), connect("DRILL"), uuid.uuid4().hex[:8]
    n, ids = 50, []
    try:
        ts = int(time.time() * 1000)
        for i in range(n):
            ids.append(r.xadd(cfg.scored, {
                "v": "1", "id": f"drill{run_id}{i:03d}", "symbol": "ZZDRILL", "title": f"Drill headline {run_id} {i}",
                "url": "", "source": "drill", "origin": "drill", "run": run_id, "published_ms": str(ts),
                "enqueued_ms": str(ts), "scored_ms": str(ts), "label": "neutral", "score": "0.0",
                "negative": "0.1", "neutral": "0.8", "positive": "0.1", "model": "drill"}))
        time.sleep(0.3)
        run(*DC, "kill", "aggregator")
        time.sleep(2)
        run(*DC, "start", "aggregator")
        query = ("import sqlite3;print(sqlite3.connect('/app/data/marketpulse.db').execute("
                 "\"select count(*),count(distinct event_id) from scored where symbol='ZZDRILL'\").fetchone())")
        count = (0, 0)
        deadline = time.time() + 120
        while time.time() < deadline:
            res = run(*DC, "exec", "-T", "aggregator", "python", "-c", query)
            m = re.search(r"\((\d+), (\d+)\)", res.stdout)
            if m:
                count = (int(m.group(1)), int(m.group(2)))
                if count[0] >= n:
                    break
            time.sleep(3)
        time.sleep(3)
        pending = r.xpending(cfg.scored, cfg.aggregator_group)["pending"]
        ok = count == (n, n) and pending == 0
        record("aggregator killed mid-batch (50 events)", ok, f"rows={count[0]} distinct={count[1]} (want {n}); pending={pending} (want 0)")
    finally:
        try:
            if ids:
                r.xdel(cfg.scored, *ids)
            run(*DC, "exec", "-T", "aggregator", "python", "-c",
                "import sqlite3;c=sqlite3.connect('/app/data/marketpulse.db');"
                "c.execute(\"delete from scored where symbol='ZZDRILL'\");"
                "c.execute(\"delete from signals where symbol='ZZDRILL'\");c.commit();"
                "from redis_client import get_redis,NEWS_KEY_PREFIX;get_redis().client.delete(NEWS_KEY_PREFIX+'ZZDRILL')")
        except Exception as exc:  # noqa: BLE001
            print("cleanup warning:", exc)


def drill_redis():
    from streaming.connection import connect
    from streaming.settings import load_settings
    cfg = load_settings()
    before = connect("DRILL").xlen(cfg.scored)
    time.sleep(2)                                   # let the AOF fsync
    run(*DC, "restart", "redis")
    deadline = time.time() + 90
    while time.time() < deadline:
        if "PONG" in run(*DC, "exec", "-T", "redis", "redis-cli", "ping").stdout:
            break
        time.sleep(2)
    after = connect("DRILL").xlen(cfg.scored)
    time.sleep(20)                                  # give clients time to back off and reconnect
    alive = running_services()
    p = verify_proc(50, 5, 120, 5000)
    out, _ = p.communicate(timeout=240)
    ok = after >= before and p.returncode == 0 and EXPECTED <= alive
    record("redis restarted", ok,
           f"scored stream {before} -> {after} entries (AOF kept data); services running: {', '.join(sorted(alive))}; "
           f"post-restart replay {'passed' if p.returncode == 0 else 'FAILED'} ({p95_of(out)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["inference", "aggregator", "redis"])
    a = ap.parse_args()
    missing = EXPECTED - {"ingestor"} - running_services()
    if missing:
        sys.exit(f"Not running: {', '.join(sorted(missing))}. Start the stack first (see top of this file).")
    for name, fn in (("inference", drill_inference), ("aggregator", drill_aggregator), ("redis", drill_redis)):
        if not a.only or a.only == name:
            print(f"--- drill: {name}")
            fn()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Failure drills ({stamp})", "", "| Drill | Result | Detail |", "|---|---|---|"]
    lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d.splitlines()[0]} |" for n, ok, d in RESULTS]
    with open(os.path.join(ROOT, "drill_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote drill_results.md")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()