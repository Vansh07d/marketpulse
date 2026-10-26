import pandas as pd
import pytest

from backtest import backtest, build_trades

DAY = 86_400_000
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)


def prices(closes, start="2024-01-01"):
    days = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": days, "symbol": "A", "close": closes})


def test_entry_is_next_close_not_signal_day_close():
    sig = pd.DataFrame({"symbol": ["A"], "ts_ms": [T0 + 3_600_000], "signal": ["BUY"]})
    # signal day close=100, next day=110, day after=121 -> must earn 10% (110->121), not 21%
    t = build_trades(sig, prices([100, 110, 121]), horizon=1, cost_bps=0)
    assert len(t) == 1 and t["fwd_return"].iloc[0] == pytest.approx(0.10)


def test_costs_and_sell_direction():
    sig = pd.DataFrame({"symbol": ["A", "A"], "ts_ms": [T0, T0 + DAY], "signal": ["SELL", "HOLD"]})
    t = build_trades(sig, prices([100, 100, 90, 90, 90]), horizon=1, cost_bps=20)
    sell = t[t["signal"] == "SELL"].iloc[0]
    assert sell["net_return"] == pytest.approx(0.10 - 0.002)
    assert t[t["signal"] == "HOLD"].iloc[0]["net_return"] == 0.0     # no cost on HOLD


def test_one_signal_per_day_keeps_the_last():
    sig = pd.DataFrame({"symbol": ["A", "A"], "ts_ms": [T0, T0 + 1000], "signal": ["BUY", "SELL"]})
    t = build_trades(sig, prices([100, 100, 100]), horizon=1)
    assert len(t) == 1 and t["signal"].iloc[0] == "SELL"


def test_no_future_prices_does_not_crash():
    sig = pd.DataFrame({"symbol": ["A"], "ts_ms": [T0], "signal": ["BUY"]})
    assert backtest(sig, prices([100]))["test"] == {"n": 0}


def test_time_split_is_chronological():
    ts = [T0 + i * DAY for i in range(10)]
    sig = pd.DataFrame({"symbol": "A", "ts_ms": ts, "signal": ["BUY"] * 10})
    out = backtest(sig, prices([100 + i for i in range(20)]), test_frac=0.3)
    assert out["tune"]["n"] + out["test"]["n"] > 0 and out["split_date"] > "2024-01-01"