#!/usr/bin/env python3
"""Find USD-M futures pairs whose spread varies between 1 tick and wider.

Uses screen.py. Steps:
 1. list every symbol in the bookTicker folder (no download);
 2. read the zip size of the quote file and the trade file for one day
    (a size check only; size is a proxy for activity);
 3. screen the N thinnest symbols that still have enough trades.

    python src/find_pairs.py --day 2023-09-12 --n 30

Output: results/symbol_sizes.csv and results/find_pairs_results.csv.
"""
import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import screen

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PREFIX = "data/futures/um/daily/bookTicker/"
QURL = ("https://data.binance.vision/data/futures/um/daily/bookTicker/"
        "{s}/{s}-bookTicker-{d}.zip")
TURL = ("https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "{s}/{s}-aggTrades-{d}.zip")


def list_symbols():
    syms, marker = [], ""
    while True:
        r = requests.get(S3, params={"delimiter": "/", "prefix": PREFIX,
                                     "marker": marker}, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        prefixes = root.findall("s:CommonPrefixes/s:Prefix", NS)
        syms += [p.text.rstrip("/").split("/")[-1] for p in prefixes]
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        marker = (root.findtext("s:NextMarker", namespaces=NS)
                  or prefixes[-1].text)
    return sorted(set(syms))


def size_mb(url):
    try:
        r = requests.head(url, timeout=30)
        if r.status_code == 200:
            return int(r.headers.get("Content-Length", 0)) / 1e6
    except requests.RequestException:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2023-09-12")
    ap.add_argument("--n", type=int, default=30, help="symbols to screen")
    ap.add_argument("--min-quote-mb", type=float, default=1.0)
    ap.add_argument("--min-trades-mb", type=float, default=1.0)
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    syms = list_symbols()
    print(f"{len(syms)} symbols listed. Reading file sizes for {args.day} ...")
    with ThreadPoolExecutor(8) as ex:
        q = list(ex.map(lambda s: size_mb(QURL.format(s=s, d=args.day)), syms))
        t = list(ex.map(lambda s: size_mb(TURL.format(s=s, d=args.day)), syms))
    df = pd.DataFrame(dict(symbol=syms, quote_mb=q, trades_mb=t))
    have = df.dropna()
    have = have[(have.quote_mb >= args.min_quote_mb)
                & (have.trades_mb >= args.min_trades_mb)]
    have = have.sort_values("quote_mb").reset_index(drop=True)
    have.to_csv(os.path.join(args.out, "symbol_sizes.csv"), index=False)
    print(f"{len(have)} symbols have both files on {args.day}. "
          f"Thinnest 15 by quote size:")
    print(have.head(15).to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    rows = []
    for sym in have.symbol.head(args.n):
        with tempfile.TemporaryDirectory() as tmp:
            z = os.path.join(tmp, f"{sym}.zip")
            print(f"{sym} ...", end=" ", flush=True)
            if not screen.download(QURL.format(s=sym, d=args.day), z):
                print("no file")
                continue
            tick, by_ticks, n, back = screen.screen_file(z)
        res = screen.summarize(tick, by_ticks) if tick else None
        if res is None:
            print("empty")
            continue
        one, two, more = res
        print(f"tick {tick:g}  1-tick {one:.1%}  rows {n:,}")
        rows.append(dict(symbol=sym, tick=tick, share_1tick=one,
                         share_2tick=two, share_3plus=more, rows=n))
    if not rows:
        print("Nothing processed.")
        return
    out = pd.DataFrame(rows).sort_values("share_1tick")
    out["pass"] = out.share_1tick.between(screen.LOW, screen.HIGH)
    out.to_csv(os.path.join(args.out, "find_pairs_results.csv"), index=False)
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nPass = {screen.LOW:.0%} to {screen.HIGH:.0%} of time at 1 tick.")


if __name__ == "__main__":
    main()
