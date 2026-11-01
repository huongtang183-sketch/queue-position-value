#!/usr/bin/env python3
"""Pair screen: share of time the best bid-ask spread is exactly one tick.

Binance USD-M futures bookTicker, daily files from data.binance.vision.

    python src/screen.py
    python src/screen.py --symbols BTCUSDT 1000PEPEUSDT --days 2023-09-12 2023-12-12

Each zip is downloaded, processed in chunks and deleted. Output: a table on
screen and results/screen_results.csv.

Method: the quote stays valid until the next update, so each spread value is
weighted by the time until the next update. The tick is inferred as the
smallest non-zero step of the best bid or ask in the file (not stated in the
file). Rows whose time goes backwards are dropped and counted (known issue,
binance-public-data #305).
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import requests

URL = ("https://data.binance.vision/data/futures/um/daily/bookTicker/"
       "{s}/{s}-bookTicker-{d}.zip")

# Candidate symbols; symbols with no file on a day are skipped and reported.
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "1000PEPEUSDT",
    "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
]
# Sample days inside the bookTicker window (2023-05-16 to 2024-03-30).
DEFAULT_DAYS = ["2023-09-12", "2023-12-12"]
LOW, HIGH = 0.40, 0.70  # pass band for the share of time at exactly 1 tick


def download(url, dest):
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        return False
    with open(dest, "wb") as f:
        for part in r.iter_content(1 << 20):
            f.write(part)
    return True


def screen_file(path, chunksize=2_000_000):
    """Return (tick, {spread_in_ticks: seconds}, n_rows, n_backwards)."""
    spread_time = {}
    min_step = np.inf
    n_rows = 0
    n_back = 0
    prev = None
    cols = ["best_bid_price", "best_ask_price", "transaction_time"]
    for df in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        t = df["transaction_time"].to_numpy(dtype="int64")
        b = df["best_bid_price"].to_numpy(dtype="float64")
        a = df["best_ask_price"].to_numpy(dtype="float64")
        n_rows += len(df)
        if prev is not None:  # carry the last row across chunks
            t = np.r_[prev[0], t]
            b = np.r_[prev[1], b]
            a = np.r_[prev[2], a]
        prev = (t[-1], b[-1], a[-1])
        if len(t) < 2:
            continue
        dt = np.diff(t)
        ok = dt >= 0
        n_back += int((~ok).sum())
        for p in (b, a):
            d = np.abs(np.diff(p))
            d = d[d > 0]
            if d.size:
                min_step = min(min_step, d.min())
        sp = np.round(a[:-1] - b[:-1], 12)
        s = pd.Series(dt[ok] / 1000.0).groupby(sp[ok]).sum()
        for k, v in s.items():
            spread_time[k] = spread_time.get(k, 0.0) + float(v)
    if not np.isfinite(min_step):
        return None, {}, n_rows, n_back
    tick = float("%.6g" % min_step)
    by_ticks = {}
    for sp, secs in spread_time.items():
        k = int(round(sp / tick))
        if k >= 1:  # drop locked or crossed quotes
            by_ticks[k] = by_ticks.get(k, 0.0) + secs
    return tick, by_ticks, n_rows, n_back


def summarize(tick, by_ticks):
    total = sum(by_ticks.values())
    if total <= 0:
        return None
    one = by_ticks.get(1, 0.0) / total
    two = by_ticks.get(2, 0.0) / total
    return one, two, 1.0 - one - two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", nargs="+", default=DEFAULT_DAYS)
    ap.add_argument("--out", default="results/screen_results.csv")
    args = ap.parse_args()

    rows, missing = [], []
    for sym in args.symbols:
        for day in args.days:
            url = URL.format(s=sym, d=day)
            with tempfile.TemporaryDirectory() as tmp:
                z = os.path.join(tmp, f"{sym}-{day}.zip")
                print(f"{sym} {day} ...", end=" ", flush=True)
                if not download(url, z):
                    print("no file")
                    missing.append((sym, day))
                    continue
                tick, by_ticks, n, back = screen_file(z)
            res = summarize(tick, by_ticks) if tick else None
            if res is None:
                print("empty")
                continue
            one, two, more = res
            print(f"tick {tick:g}  1-tick {one:.1%}  rows {n:,}  backwards {back}")
            rows.append(dict(symbol=sym, day=day, tick=tick, share_1tick=one,
                             share_2tick=two, share_3plus=more, rows=n,
                             backwards=back))
    if not rows:
        print("No data processed.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    g = df.groupby("symbol").agg(days=("day", "count"),
                                 tick=("tick", "median"),
                                 share_1tick=("share_1tick", "mean"),
                                 share_2tick=("share_2tick", "mean"),
                                 share_3plus=("share_3plus", "mean"))
    g["pass"] = (g.share_1tick >= LOW) & (g.share_1tick <= HIGH)
    print()
    print(g.sort_values("share_1tick").to_string(
        float_format=lambda x: f"{x:.3f}"))
    if missing:
        print(f"\nNo file for {len(missing)} symbol-days: {missing[:10]}")
    print(f"\nSaved {args.out}. Pass = {LOW:.0%} to {HIGH:.0%} of time at 1 tick.")


if __name__ == "__main__":
    main()
