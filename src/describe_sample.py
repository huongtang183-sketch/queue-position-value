#!/usr/bin/env python3
"""Sample description: fills at the best quote by pair and spread state.

    python src/describe_sample.py --data data

Per pair: tick size, number of fills in total and at each spread of one to six
ticks, the number of days without fills in each of the first four states, and
the share of fills at one tick (overall and per day: minimum, median, maximum).

Output: results/sample_by_spread.csv
"""
import argparse
import os

import pandas as pd

from fills import load


def describe(df, names):
    n_days = df["day"].nunique()
    rows = []
    for c, name in sorted(names.items()):
        d = df[df["pair"] == c]
        counts = d.groupby("k").size()
        by_day = d.groupby(["day", "k"]).size().unstack(fill_value=0)
        share1 = by_day.get(1, 0) / by_day.sum(axis=1)
        r = dict(pair=name, tick=d["tick"].iloc[0], fills=len(d), days=d["day"].nunique())
        for j in range(1, 7):
            r[f"fills_k{j}"] = int(counts.get(j, 0))
        for j in range(1, 5):
            empty = int((by_day.get(j, pd.Series(0, index=by_day.index)) == 0).sum())
            r[f"days_without_fills_k{j}"] = empty + (n_days - d["day"].nunique())
        r["share_one_tick_min"] = float(share1.min())
        r["share_one_tick_median"] = float(share1.median())
        r["share_one_tick_max"] = float(share1.max())
        r["share_one_tick_all"] = float(counts.get(1, 0) / counts.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data", help="folder with the fill files")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    df, names = load(args.data)
    out = describe(df, names)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "sample_by_spread.csv")
    out.to_csv(path, index=False)
    print(f"Fills at the best quote: {len(df):,}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
