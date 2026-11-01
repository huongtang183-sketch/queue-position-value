#!/usr/bin/env python3
"""Build the fill table: one row per aggregated trade, for the chosen pairs.

Uses screen.py. For each pair and day it downloads the quote file and the
trade file, writes data/<PAIR>/<PAIR>-<DAY>.csv.gz and deletes the raw zips.
Re-running skips days already built.

    python src/build_fills.py --start 2023-09-01 --end 2023-10-30

The run log is written to results/build_log.csv.

Definitions (s = +1 if the aggressor buys, -1 if it sells):
    m0   mid of the last quote strictly before the trade
    mD   mid of the last quote at or before trade time + D (D = 1 s, 10 s)
    ES   = 2 s (p - m0)        effective spread
    RS   = 2 s (p - mD)        realized spread, passive side
    PI   = ES - RS = 2 s (mD - m0)
"""
import argparse
import os
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

import screen

QURL = ("https://data.binance.vision/data/futures/um/daily/bookTicker/"
        "{s}/{s}-bookTicker-{d}.zip")
TURL = ("https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "{s}/{s}-aggTrades-{d}.zip")
PAIRS = ["WOOUSDT", "RNDRUSDT", "MTLUSDT", "ASTRUSDT", "KEYUSDT"]
QCOLS = ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty",
         "transaction_time"]
TCOLS = ["price", "quantity", "transact_time", "is_buyer_maker"]


def load_quotes(path):
    q = pd.read_csv(path, usecols=QCOLS)
    t = q["transaction_time"].to_numpy("int64")
    keep = t >= np.maximum.accumulate(t)          # drop rows going backwards
    n_back = int((~keep).sum())
    q = q[keep]
    return (q["transaction_time"].to_numpy("int64"),
            q["best_bid_price"].to_numpy("float64"),
            q["best_bid_qty"].to_numpy("float64"),
            q["best_ask_price"].to_numpy("float64"),
            q["best_ask_qty"].to_numpy("float64"), n_back)


def infer_tick(b, a):
    step = np.inf
    for x in (b, a):
        d = np.abs(np.diff(x))
        d = d[d > 0]
        if d.size:
            step = min(step, d.min())
    return float("%.6g" % step)


def build(quotes, trades):
    qt, b, bq, a, aq, n_back = quotes
    tick = infer_tick(b, a)
    mid = (b + a) / 2.0
    t = trades["transact_time"].to_numpy("int64")
    p = trades["price"].to_numpy("float64")
    qty = trades["quantity"].to_numpy("float64")
    maker = trades["is_buyer_maker"].astype(str).str.lower().eq("true").to_numpy()
    s = np.where(maker, -1, 1)                    # buyer is maker -> seller aggressor

    i0 = np.searchsorted(qt, t, side="left") - 1  # last quote strictly before t
    ok = i0 >= 0
    t, p, qty, s, i0 = t[ok], p[ok], qty[ok], s[ok], i0[ok]
    last = qt[-1]

    out = dict(t_ms=t, hour=(t // 3600000) % 24, price=p, qty=qty, s=s,
               spread_ticks=np.rint((a[i0] - b[i0]) / tick).astype(int),
               bid_qty=bq[i0], ask_qty=aq[i0],
               depth_touch=np.where(s == 1, aq[i0], bq[i0]),
               m0=mid[i0], tick=tick)
    out["T"] = (out["spread_ticks"] == 1).astype(int)
    out["ES"] = 2 * s * (p - mid[i0])
    for name, d in (("1", 1000), ("10", 10000)):
        iD = np.searchsorted(qt, t + d, side="right") - 1
        mD = np.where(t + d <= last, mid[iD], np.nan)
        out["m" + name] = mD
        out["RS" + name] = 2 * s * (p - mD)
        out["PI" + name] = out["ES"] - out["RS" + name]

    # trailing 60 s realised volatility on a 1 s grid, no look-ahead
    g0 = -(-qt[0] // 1000) * 1000
    g = np.arange(g0, last + 1, 1000)
    mg = mid[np.searchsorted(qt, g, side="right") - 1]
    r = np.diff(np.log(mg))
    cs = np.r_[0.0, np.cumsum(r ** 2)]           # cs[k] = sum of r_1..r_k squared
    k = (t - g0) // 1000
    rv = np.full(len(t), np.nan)
    good = (k >= 60) & (k < len(g))
    rv[good] = np.sqrt(cs[k[good]] - cs[k[good] - 60])
    out["rv60"] = rv

    stats = dict(trades=len(trades), no_quote_before=int((~ok).sum()),
                 no_future_10s=int(np.isnan(out["m10"]).sum()),
                 quote_rows=len(qt), quote_backwards=n_back, tick=tick)
    return pd.DataFrame(out), stats


def daterange(start, end):
    d = start
    while d <= end:
        yield d.isoformat()
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default="2023-10-30")
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--out", default="data", help="folder for the fill files")
    ap.add_argument("--log", default="results/build_log.csv")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log = []
    for sym in args.pairs:
        os.makedirs(os.path.join(args.out, sym), exist_ok=True)
        for day in daterange(start, end):
            dest = os.path.join(args.out, sym, f"{sym}-{day}.csv.gz")
            if os.path.exists(dest):
                print(sym, day, "already done")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                qz, tz = os.path.join(tmp, "q.zip"), os.path.join(tmp, "t.zip")
                print(sym, day, "...", end=" ", flush=True)
                if not (screen.download(QURL.format(s=sym, d=day), qz)
                        and screen.download(TURL.format(s=sym, d=day), tz)):
                    print("missing file")
                    log.append(dict(symbol=sym, day=day, status="missing"))
                    continue
                fills, st = build(load_quotes(qz),
                                  pd.read_csv(tz, usecols=TCOLS))
            fills.to_csv(dest, index=False, compression="gzip")
            print(f"{len(fills):,} fills, tick {st['tick']:g}")
            log.append(dict(symbol=sym, day=day, status="ok", **st))
            pd.DataFrame(log).to_csv(args.log, index=False)
    pd.DataFrame(log).to_csv(args.log, index=False)
    done = sum(1 for r in log if r["status"] == "ok")
    print(f"\nDone: {done} pair-days written, "
          f"{len(log) - done} missing. See {args.log}.")


if __name__ == "__main__":
    main()
