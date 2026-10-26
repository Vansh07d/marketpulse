"""
Event schema, sanitisation and time helpers.

Every field in a Redis stream entry is a string. Headlines come from the internet, so they are
treated as untrusted: control and bidi characters are stripped, lengths are capped, URLs must be
http(s), and symbols must match a strict pattern. Key names are never built from event content.

Timestamps are epoch milliseconds (ints), kept separate:
    published_ms  when the source says it was published (event time; 0 if unknown)
    fetched_ms    when we downloaded it
    enqueued_ms   when it entered news.raw
    consumed_ms   when the inference consumer read it
    infer_start_ms / scored_ms   inference window of its batch
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1"
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
MAX_TITLE_CHARS = 300
MAX_URL_CHARS = 1000
MAX_SOURCE_CHARS = 64
MAX_FIELD_BYTES = 4096                      # hard cap on any single field read from Redis
_WS = re.compile(r"\s+")
_TRACKING = re.compile(r"[?&](utm_[a-z]+|fbclid|gclid)=[^&#]*", re.I)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class EventError(ValueError):
    """The event is malformed or unsafe and must not be scored."""


def clean_text(value: Any, limit: int) -> str:
    """Strip control/format (incl. bidi) characters, collapse whitespace, cap length."""
    text = _WS.sub(" ", unicodedata.normalize("NFKC", str(value or "")))   # newlines/tabs become spaces
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn"))
    return _WS.sub(" ", text).strip()[:limit]


def clean_url(value: Any) -> str:
    url = str(value or "").strip()
    if len(url) > MAX_URL_CHARS or not url.lower().startswith(("http://", "https://")):
        return ""
    if any(ord(ch) < 33 or ord(ch) == 127 for ch in url):
        return ""
    return _TRACKING.sub("", url)


def normalize_title(title: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", title).casefold()).strip(" .!?\"'")


def event_id(symbol: str, title: str) -> str:
    """Stable id used for de-duplication: same symbol + same normalised headline."""
    return hashlib.sha256(f"{symbol}|{normalize_title(title)}".encode("utf-8")).hexdigest()[:32]


def to_ms(value: Any) -> int:
    if not value:
        return 0
    if isinstance(value, (int, float)):
        return int(value if value > 1e11 else value * 1000)
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0
    ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    return int(ts.timestamp() * 1000)


def check_published(published_ms: int, now: int, max_age_hours: int, future_skew_s: int = 300) -> str:
    """Classify a source timestamp: 'ok', 'missing' (0/unknown), 'future' (clock/bad data) or 'old'."""
    if not published_ms:
        return "missing"
    if published_ms > now + future_skew_s * 1000:
        return "future"
    if max_age_hours and now - published_ms > max_age_hours * 3_600_000:
        return "old"
    return "ok"


def build_event(article: Dict[str, Any], symbol: str, origin: str, fetched_ms: Optional[int] = None,
                run: str = "") -> Dict[str, str]:
    """Turn a provider article into a validated, sanitised stream entry (flat str -> str)."""
    symbol = str(symbol or "").strip().upper()
    if not SYMBOL_RE.match(symbol):
        raise EventError("invalid symbol")
    title = clean_text(article.get("title"), MAX_TITLE_CHARS)
    if len(title) < 5:
        raise EventError("empty or too short title")
    fetched = fetched_ms or now_ms()
    return {
        "v": SCHEMA_VERSION,
        "id": event_id(symbol, title),
        "symbol": symbol,
        "title": title,
        "url": clean_url(article.get("url")),
        "source": clean_text(article.get("source"), MAX_SOURCE_CHARS),
        "origin": clean_text(origin, MAX_SOURCE_CHARS),
        "run": clean_text(run, 32),
        "published_ms": str(to_ms(article.get("published_at")) or 0),
        "fetched_ms": str(fetched),
    }


@dataclass(frozen=True)
class Event:
    id: str
    symbol: str
    title: str
    url: str
    source: str
    origin: str
    run: str
    published_ms: int
    fetched_ms: int
    enqueued_ms: int


def parse_event(fields: Dict[str, str]) -> Event:
    """Validate an entry read from Redis. Never trusts what is on the wire."""
    try:
        for key, value in fields.items():
            if len(str(value).encode("utf-8", "ignore")) > MAX_FIELD_BYTES:
                raise EventError(f"field {key!r} too large")
        if fields.get("v") != SCHEMA_VERSION:
            raise EventError("unsupported schema version")
        symbol = str(fields.get("symbol", "")).upper()
        if not SYMBOL_RE.match(symbol):
            raise EventError("invalid symbol")
        title = clean_text(fields.get("title"), MAX_TITLE_CHARS)
        if len(title) < 5:
            raise EventError("empty or too short title")
        ident = str(fields.get("id", ""))
        if not re.fullmatch(r"[0-9a-f]{32}", ident):
            raise EventError("invalid id")
        return Event(
            id=ident, symbol=symbol, title=title, url=clean_url(fields.get("url")),
            source=clean_text(fields.get("source"), MAX_SOURCE_CHARS),
            origin=clean_text(fields.get("origin"), MAX_SOURCE_CHARS),
            run=clean_text(fields.get("run"), 32),
            published_ms=int(fields.get("published_ms") or 0),
            fetched_ms=int(fields.get("fetched_ms") or 0),
            enqueued_ms=int(fields["enqueued_ms"]),
        )
    except EventError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EventError(f"malformed event: {exc.__class__.__name__}") from exc