"""
Startup checks: fail fast with a clear message instead of crashing later with a stack trace.

    from streaming.preflight import require
    require("consumer")      # roles: ingestor | consumer | aggregator
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

API_KEYS = ("ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", "STOCKDATA_API_KEY", "EODHD_API_KEY")


def problems(role: str) -> List[str]:
    found: List[str] = []
    try:
        from streaming.connection import connect
        connect(role.upper()).ping()
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot use Redis"
        found.append(f"Redis is not reachable ({exc.__class__.__name__}). Check REDIS_HOST/REDIS_PORT/REDIS_URL.")
    if role == "ingestor" and not any(os.getenv(k, "").strip() for k in API_KEYS):
        found.append("No news/market API key set. Add at least one of: " + ", ".join(API_KEYS))
    if role == "consumer":
        model = Path(os.getenv("MODEL_PATH", "models/finbert_model"))
        if not model.exists() and not os.getenv("MODEL_ID", "").strip():
            found.append(f"Model not found at {model}. Mount ./models or set MODEL_PATH / MODEL_ID.")
    if role in ("aggregator", "consumer"):
        for var in ("STREAM_BATCH_MAX", "SIGNAL_REFRESH_S", "NEWS_LOOKBACK_HOURS"):
            raw = os.getenv(var)
            if raw is not None and not raw.strip().lstrip("-").isdigit():
                found.append(f"{var}={raw!r} is not an integer.")
    return found


def require(role: str) -> None:
    issues = problems(role)
    if issues:
        raise SystemExit(f"[preflight:{role}] cannot start:\n  - " + "\n  - ".join(issues))