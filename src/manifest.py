#!/usr/bin/env python3
"""Data manifest: the fill files on disk, per pair.

    python src/manifest.py --data data --start 2023-09-01 --end 2023-10-30

Prints and saves results/manifest.txt. Reads the fill files built by
build_fills.py and, if present, its run log results/build_log.csv.
"""
import argparse
import glob
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

OUT = []


def say(t=""):
    print(t)
    OUT.append(str(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with the fill files")
    ap.add_argument("--start", default="2023-09-01")
    ap.add_argument("--end", default="2023-10-30")
    ap.add_argument("--log", default="results/build_log.csv")
    ap.add_argument("--out", default="results", help="output folder")
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    want = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    rows, cols = [], None
    for sd in sorted(glob.glob(os.path.join(a.data, "*"))):
        sym = os.path.basename(sd)
        files = sorted(glob.glob(os.path.join(sd, "*.csv.gz")))
        days = [os.path.basename(f)[len(sym) + 1:-7] for f in files]
        n = nt = 0
        ticks = set()
        for f in files:
            d = pd.read_csv(f)
            cols = list(d.columns)
            ok = d["spread_ticks"] >= 1
            n += int(ok.sum())
            t = ok & ((d["ES"] / d["tick"] - d["spread_ticks"]).abs() < 0.01)
            nt += int(t.sum())
            ticks.add(float(d["tick"].iloc[0]) if len(d) else np.nan)
            print(sym, os.path.basename(f)[-17:-7], end="\r", flush=True)
        miss = [x for x in want if x not in set(days)]
        rows.append(dict(symbol=sym, days=len(days), first=min(days, default=""),
                         last=max(days, default=""), missing=len(miss),
                         tick=",".join("%g" % x for x in sorted(ticks)),
                         fills_valid_quote=n, fills_at_touch=nt))
    say("DATA MANIFEST (fill files on disk)")
    say(f"calendar range {a.start} to {a.end}: {len(want)} days")
    say(pd.DataFrame(rows).to_string(index=False))
    say("\nfill-file columns: " + ", ".join(cols or []))
    if os.path.exists(a.log):
        lg = pd.read_csv(a.log)
        say(f"\nBuild log ({a.log}): {len(lg)} rows; status counts:")
        say(lg["status"].value_counts().to_string())
        if "trades" in lg:
            g = lg[lg["status"] == "ok"].groupby("symbol")[
                ["trades", "quote_rows", "no_quote_before", "no_future_10s",
                 "quote_backwards"]].sum()
            say("\nraw rows and drops, per pair (from the last build run only):")
            say(g.to_string())
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "manifest.txt")
    with open(path, "w") as f:
        f.write("\n".join(OUT))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
