"""
Durable research store (SQLite). Redis keeps only the live, bounded state; this keeps the history
needed for audit, labeling and backtesting.

    scored   one row per scored headline (idempotent on event id)
    signals  one row per emitted BUY/HOLD/SELL signal (repeats of an unchanged score are skipped)

Path: MARKETPULSE_DB (default data/marketpulse.db). ":memory:" works for tests.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Iterable, List

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scored (
    event_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, title TEXT NOT NULL, url TEXT, source TEXT,
    published_ms INTEGER, enqueued_ms INTEGER, scored_ms INTEGER, event_ms INTEGER NOT NULL,
    label TEXT, score REAL, negative REAL, neutral REAL, positive REAL, model TEXT
);
CREATE INDEX IF NOT EXISTS ix_scored_symbol_time ON scored(symbol, event_ms);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, ts_ms INTEGER NOT NULL,
    signal TEXT NOT NULL, signal_score REAL, news_sentiment REAL, price_momentum REAL,
    market_momentum REAL, price REAL, change_percent REAL, n_articles INTEGER
);
CREATE INDEX IF NOT EXISTS ix_signals_symbol_time ON signals(symbol, ts_ms);
"""


def _num(value: Any):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class Store:
    def __init__(self, path: str = "data/marketpulse.db"):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ scored
    def add_scored(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Insert rows, ignoring ids already stored. Returns how many were new."""
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                """INSERT OR IGNORE INTO scored (event_id, symbol, title, url, source, published_ms,
                   enqueued_ms, scored_ms, event_ms, label, score, negative, neutral, positive, model)
                   VALUES (:event_id, :symbol, :title, :url, :source, :published_ms, :enqueued_ms,
                           :scored_ms, :event_ms, :label, :score, :negative, :neutral, :positive, :model)""",
                list(rows))
        return self.db.total_changes - before

    def recent(self, symbol: str, since_ms: int, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.db.execute(
            "SELECT * FROM scored WHERE symbol=? AND event_ms>=? ORDER BY event_ms DESC LIMIT ?",
            (symbol, since_ms, limit))
        return [dict(r) for r in cur]

    def active_symbols(self, since_ms: int) -> List[str]:
        cur = self.db.execute("SELECT DISTINCT symbol FROM scored WHERE event_ms>=?", (since_ms,))
        return [r[0] for r in cur]

    def headlines(self, limit: int = 200, symbol: str = "") -> List[Dict[str, Any]]:
        """Random sample of stored headlines (for hand-labeling an evaluation set)."""
        sql = "SELECT symbol, title, label, score FROM scored"
        args: List[Any] = []
        if symbol:
            sql += " WHERE symbol=?"
            args.append(symbol)
        cur = self.db.execute(sql + " ORDER BY RANDOM() LIMIT ?", (*args, limit))
        return [dict(r) for r in cur]

    # ------------------------------------------------------------------ signals
    def add_signal(self, sig: Dict[str, Any], n_articles: int, ts_ms: int, dedupe_ms: int = 600_000) -> bool:
        """Store a signal unless the previous one for the symbol is identical and recent."""
        last = self.db.execute(
            "SELECT ts_ms, signal_score, signal FROM signals WHERE symbol=? ORDER BY ts_ms DESC LIMIT 1",
            (sig["symbol"],)).fetchone()
        if last and last["signal"] == sig["signal"] and last["signal_score"] == sig.get("signal_score") \
                and ts_ms - last["ts_ms"] < dedupe_ms:
            return False
        with self.db:
            self.db.execute(
                """INSERT INTO signals (symbol, ts_ms, signal, signal_score, news_sentiment, price_momentum,
                   market_momentum, price, change_percent, n_articles) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (sig["symbol"], ts_ms, sig["signal"], _num(sig.get("signal_score")),
                 _num(sig.get("news_sentiment")), _num(sig.get("price_momentum")),
                 _num(sig.get("market_momentum")), _num(sig.get("price")),
                 _num(sig.get("change_percent")), n_articles))
        return True

    def signals(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM signals ORDER BY ts_ms")]

    def counts(self) -> Dict[str, int]:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("scored", "signals")}