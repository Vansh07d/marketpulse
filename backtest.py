"""
Backtest stored BUY/HOLD/SELL signals against forward returns.

Rules that keep it honest
  * Entry is the CLOSE OF THE NEXT TRADING DAY after the signal date (the signal uses same-day price
    change, so entering on the signal day's close would be look-ahead). Exit is `horizon` days later.
  * One signal per symbol per day (the last one), so overlapping duplicates do not inflate n.
  * Costs: `--cost-bps` is the round-trip fee + slippage charged on every non-HOLD trade.
  * Time-based split: the first (1 - test_frac) of dates is "tune", the last test_frac is "test".
    Look at "test" once, after you stop adjusting weights/thresholds on "tune".
  * Baseline: average forward return of ALL signals (what you get by always being long).

Usage
  python backtest.py --db data/marketpulse.db --yf --suffix .NS --horizon 1 --cost-bps 20
  python backtest.py --db data/marketpulse.db --prices prices.csv      # columns: date,symbol,close
"""
from __future__ import annotations

import argparse
import sqlite3
from typing import Any, Dict

import numpy as np
import pandas as pd

POSITION = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


def build_trades(signals: pd.DataFrame, prices: pd.DataFrame, horizon: int = 1,
                 cost_bps: float = 20.0, long_only: bool = False) -> pd.DataFrame:
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["ts_ms"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    sig = sig.sort_values("ts_ms").groupby(["symbol", "date"], as_index=False).last()
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.normalize()
    rows = []
    for symbol, grp in sig.groupby("symbol"):
        series = px[px["symbol"] == symbol].sort_values("date").drop_duplicates("date")
        dates, closes = series["date"].to_numpy(), series["close"].to_numpy(dtype=float)
        for _, s in grp.iterrows():
            entry = int(np.searchsorted(dates, np.datetime64(s["date"]), side="right"))   # first date > signal date
            exit_ = entry + horizon
            if exit_ >= len(closes):
                continue
            fwd = closes[exit_] / closes[entry] - 1.0
            pos = POSITION.get(s["signal"], 0.0)
            if long_only and pos < 0:
                pos = 0.0
            rows.append({"symbol": symbol, "date": s["date"], "signal": s["signal"], "position": pos,
                         "fwd_return": fwd, "net_return": pos * fwd - abs(pos) * cost_bps / 1e4})
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {"n": 0}
    active = trades[trades["position"] != 0]
    by_signal = {k: round(float(v) * 1e4, 1) for k, v in trades.groupby("signal")["fwd_return"].mean().items()}
    return {
        "n": int(len(trades)), "trades": int(len(active)),
        "hit_rate": round(float((active["position"] * active["fwd_return"] > 0).mean()), 3) if len(active) else None,
        "mean_net_bps_per_trade": round(float(active["net_return"].mean()) * 1e4, 1) if len(active) else None,
        "total_net_return_pct": round(float(active["net_return"].sum()) * 100, 2),
        "baseline_always_long_bps": round(float(trades["fwd_return"].mean()) * 1e4, 1),
        "avg_forward_bps_by_signal": by_signal,
        "warning": "n < 30: not statistically meaningful" if len(active) < 30 else "",
    }


def backtest(signals: pd.DataFrame, prices: pd.DataFrame, horizon: int = 1, cost_bps: float = 20.0,
             test_frac: float = 0.3, long_only: bool = False) -> Dict[str, Any]:
    trades = build_trades(signals, prices, horizon, cost_bps, long_only) if len(signals) else pd.DataFrame()
    if trades.empty:        # no signal had a usable future price yet (e.g. data collected today)
        return {"tune": {"n": 0}, "test": {"n": 0}}
    trades = trades.sort_values("date")
    cut_date = trades["date"].iloc[int(len(trades) * (1 - test_frac))] if len(trades) > 1 else trades["date"].iloc[0]
    tune, test = trades[trades["date"] < cut_date], trades[trades["date"] >= cut_date]
    return {"tune": summarize(tune), "test": summarize(test), "split_date": str(cut_date.date())}


def _load_signals(db: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query("SELECT symbol, ts_ms, signal FROM signals", conn)


def _load_prices(args: argparse.Namespace, signals: pd.DataFrame) -> pd.DataFrame:
    if args.prices:
        return pd.read_csv(args.prices)
    import yfinance as yf
    start = pd.to_datetime(signals["ts_ms"].min(), unit="ms") - pd.Timedelta(days=5)
    frames = []
    for symbol in signals["symbol"].unique():
        hist = yf.Ticker(symbol + args.suffix).history(start=start.date().isoformat(), auto_adjust=True)
        if not hist.empty:
            frames.append(pd.DataFrame({"date": hist.index.tz_localize(None), "symbol": symbol, "close": hist["Close"].values}))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["date", "symbol", "close"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/marketpulse.db")
    p.add_argument("--prices", help="CSV with date,symbol,close (skips yfinance)")
    p.add_argument("--yf", action="store_true", help="download prices with yfinance")
    p.add_argument("--suffix", default="", help="yfinance ticker suffix, e.g. .NS for NSE")
    p.add_argument("--horizon", type=int, default=1, help="holding period in trading days")
    p.add_argument("--cost-bps", type=float, default=20.0, help="round-trip fees + slippage per trade")
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--long-only", action="store_true")
    args = p.parse_args()
    if not (args.prices or args.yf):
        p.error("give --prices CSV or --yf")
    signals = _load_signals(args.db)
    if signals.empty:
        raise SystemExit("No signals stored yet. Run the stack for a few days first.")
    result = backtest(signals, _load_prices(args, signals), args.horizon, args.cost_bps, args.test_frac, args.long_only)
    for part in ("tune", "test"):
        print(f"[{part}]", result[part])
    print("split date:", result.get("split_date"))


if __name__ == "__main__":
    main()