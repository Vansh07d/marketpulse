import pytest

from streaming.schema import (EventError, build_event, clean_text, clean_url, event_id, parse_event, to_ms)


def article(**kw):
    return {"title": "Infosys wins a large cloud contract", "url": "https://example.com/a?utm_source=x&id=7",
            "source": "Reuters", "published_at": "2026-10-24T08:00:00+00:00", **kw}


def test_build_event_sanitises_and_stamps():
    ev = build_event(article(), "infy", "finnhub", fetched_ms=1_000)
    assert ev["symbol"] == "INFY" and ev["v"] == "1" and ev["fetched_ms"] == "1000"
    assert "utm_source" not in ev["url"] and "id=7" in ev["url"]
    assert int(ev["published_ms"]) > 0


def test_control_and_bidi_characters_are_removed():
    dirty = "Infosys\x00 up\u202e 5%\n\x1b[31mred\x07 today"
    cleaned = clean_text(dirty, 300)
    assert "\x00" not in cleaned and "\u202e" not in cleaned and "\x1b" not in cleaned and "\n" not in cleaned
    assert cleaned.startswith("Infosys up 5%")


def test_title_is_length_capped():
    assert len(build_event(article(title="x" * 5000), "INFY", "t")["title"]) == 300


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///etc/passwd", "ftp://x.y/z", "https://a b.com", ""])
def test_unsafe_urls_are_dropped(url):
    assert clean_url(url) == ""


@pytest.mark.parametrize("symbol", ["", "in fy", "INFY;DROP", "../etc", "A" * 40, "inf\ny"])
def test_bad_symbols_rejected(symbol):
    with pytest.raises(EventError):
        build_event(article(), symbol, "t")


def test_short_or_empty_titles_rejected():
    for title in ("", "  ", "ab", None):
        with pytest.raises(EventError):
            build_event(article(title=title), "INFY", "t")


def test_event_id_ignores_case_whitespace_punctuation_but_not_symbol():
    a = event_id("INFY", "Infosys   wins deal!")
    assert a == event_id("INFY", "infosys wins deal")
    assert a != event_id("TCS", "infosys wins deal")


def test_parse_event_roundtrip_and_validation():
    ev = build_event(article(), "INFY", "t")
    ev["enqueued_ms"] = "5"
    parsed = parse_event(ev)
    assert parsed.symbol == "INFY" and parsed.enqueued_ms == 5
    for bad in ({**ev, "v": "9"}, {**ev, "id": "nothex"}, {**ev, "symbol": "x y"},
                {**ev, "title": ""}, {**ev, "enqueued_ms": "abc"}, {**ev, "title": "a" * 9000}):
        with pytest.raises(EventError):
            parse_event(bad)
    missing = dict(ev)
    del missing["enqueued_ms"]
    with pytest.raises(EventError):
        parse_event(missing)


def test_to_ms_handles_formats():
    assert to_ms("2026-10-24T08:00:00Z") == to_ms("2026-10-24T08:00:00+00:00") > 0
    assert to_ms(1_700_000_000) == 1_700_000_000_000
    assert to_ms("garbage") == 0 and to_ms(None) == 0