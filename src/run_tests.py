#!/usr/bin/env python3
"""Tests of the one-tick gap G (Q1 and Q2) with their robustness checks.

    python src/run_tests.py --data data

Equation (2) is estimated on fills at the best quote with a spread of one to
six ticks. G = (1/5) sum_p (theta_p3 - 2 theta_p2) is passive realized spread at
one tick minus the straight line through its two- and three-tick values,
averaged over pairs. Q1: pair-by-day effects only. Q2: adds volatility, depth
and fill-size controls. Each test is run with R measured 10 s and 1 s after the
fill; G is also reported for each pair.

Output: results/test_results.csv (95% intervals from t(days - 1); p-values
two-sided, standard errors clustered by day).
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from fills import fit, load

MAX_SPREAD = 6


def gap(df, names, y, with_controls):
    """G pooled over pairs and for each pair, with confidence intervals and p-values."""
    k = df["k"].to_numpy()
    pair = df["pair"].to_numpy()
    use = (k <= MAX_SPREAD) & np.isfinite(y)
    if with_controls:
        use &= df["rv60"].notna().to_numpy()
    dummies, col = [], {}
    for c in sorted(names):
        for j in range(2, MAX_SPREAD + 1):
            d = ((pair == c) & (k == j)).astype(np.int8)
            if d[use].sum() > 0:
                col[(c, j)] = len(dummies)
                dummies.append(d)
    beta, cov, n, n_days = fit(df, np.flatnonzero(use), y, dummies, with_controls)

    def weights(c):
        v = np.zeros(len(dummies))
        v[col[(c, 3)]] += 1.0
        v[col[(c, 2)]] -= 2.0
        return v

    scopes = [("pooled", np.mean([weights(c) for c in sorted(names)], axis=0))]
    scopes += [(names[c], weights(c)) for c in sorted(names)]
    rows = []
    for scope, w in scopes:
        est = float(w @ beta)
        se = float(np.sqrt(w @ cov @ w))
        half = float(stats.t.ppf(0.975, n_days - 1) * se)
        rows.append(dict(scope=scope, fills=n, G=est, ci_low=est - half, ci_high=est + half,
                         p_value=float(2 * stats.t.sf(abs(est / se), n_days - 1))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data", help="folder with the fill files")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    df, names = load(args.data)
    tests = [("Q1", False, "R10", 10), ("Q1", False, "R1", 1),
             ("Q2", True, "R10", 10), ("Q2", True, "R1", 1)]
    parts = []
    for test, with_controls, column, horizon in tests:
        r = gap(df, names, df[column].to_numpy(), with_controls)
        r.insert(0, "horizon_s", horizon)
        r.insert(0, "test", test)
        parts.append(r)
        print("done", test, f"{horizon} s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "test_results.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
