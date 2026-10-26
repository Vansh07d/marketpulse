"""
Replay: stream headlines from a file into news.raw at a fixed rate, for reproducible demos,
load tests and latency measurement.

    python -m streaming.replay --file all-data.csv --rate 50 --count 500
    python -m streaming.report --run <printed run id>

Accepts .txt (one headline per line), .jsonl ({"title": ...}) or .csv (a title/text column, or
headerless with the headline in the last column, like Financial PhraseBank). Replayed events skip
de-duplication (every emission is unique) and carry `origin=replay` plus the run id.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, List

from streaming.ingestor import publish_many
from streaming.schema import EventError, build_event, now_ms
from streaming.settings import Settings, load_settings


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def load_headlines(path: str) -> List[str]:
    p = Path(path)
    text = _read_text(p)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = (json.loads(line) for line in text.splitlines() if line.strip())
        return [str(r.get("title") or r.get("text") or "") for r in rows]
    if suffix == ".csv":
        rows = list(csv.reader(text.splitlines()))
        if not rows:
            return []
        header = [c.strip().lower() for c in rows[0]]
        for name in ("title", "headline", "text"):
            if name in header:
                idx = header.index(name)
                return [r[idx] for r in rows[1:] if len(r) > idx]
        return [r[-1] for r in rows if r]
    return [line for line in text.splitlines() if line.strip()]


def replay(r: Any, headlines: List[str], symbol: str, rate: float, count: int, cfg: Settings,
           seed: int = 42, run: str = "") -> dict:
    """Emit `count` events at `rate` per second using an absolute schedule (no cumulative drift)."""
    if rate <= 0 or count <= 0:
        raise ValueError("rate and count must be positive")
    rng = random.Random(seed)
    pool = [h for h in headlines if len(h.strip()) >= 5]
    if not pool:
        raise ValueError("no usable headlines in input")
    run = run or uuid.uuid4().hex[:8]
    order: Iterator[str] = itertools.chain.from_iterable(rng.sample(pool, len(pool)) for _ in itertools.count())

    start = time.perf_counter()
    sent = 0
    while sent < count:
        due = min(count, int((time.perf_counter() - start) * rate) + 1)      # events whose slot has arrived
        if due > sent:
            batch = []
            for i in range(sent, due):
                try:
                    ev = build_event({"title": next(order), "source": "replay"}, symbol, "replay", run=run)
                except EventError:
                    continue
                ev["id"] = uuid.uuid4().hex[:32]                         # unique per emission
                ev["published_ms"] = ev["fetched_ms"] = str(now_ms())
                batch.append(ev)
            publish_many(r, batch, cfg, dedup=False)
            sent = due
        else:
            time.sleep(max(0.0, min(0.005, (sent / rate) - (time.perf_counter() - start))))
    elapsed = time.perf_counter() - start
    return {"run": run, "sent": sent, "seconds": round(elapsed, 3), "achieved_rate": round(sent / elapsed, 1)}


def main() -> None:
    from streaming.connection import connect

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default="all-data.csv")
    parser.add_argument("--symbol", default="INFY")
    parser.add_argument("--rate", type=float, default=20.0, help="events per second")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    info = replay(connect("REPLAY"), load_headlines(args.file), args.symbol, args.rate, args.count,
                  load_settings(), args.seed)
    print(json.dumps(info))
    print(f"Now run:  python -m streaming.report --run {info['run']}")


if __name__ == "__main__":
    main()