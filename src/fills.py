"""Shared routines for the analysis of the fill files.

load()   reads the fill files written by build_fills.py, keeps the fills that
         executed at the best quote and builds the control bins.
fit()    least squares for equation (2) with standard errors clustered by day.
"""
import glob
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse

CHUNK = 500_000
COLS = ["qty", "spread_ticks", "depth_touch", "ES", "RS1", "RS10", "rv60", "tick"]
CONTROLS = ["rvbin", "dbin", "rbin"]
SIZE_EDGES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.999]


def load(folder):
    """Return the fills at the best quote (one row per fill) and a pair-code map.

    Control bins: trailing volatility and depth at the best quote in deciles
    within pair (rvbin, dbin); fill size relative to displayed depth in seven
    bins (rbin), the last holding the fills that clear the displayed level.
    """
    parts, pairs = [], {}
    for path in sorted(glob.glob(os.path.join(folder, "*", "*.csv.gz"))):
        pair = os.path.basename(os.path.dirname(path))
        day = date.fromisoformat(os.path.basename(path).split(pair + "-")[1][:10]).toordinal()
        code = pairs.setdefault(pair, len(pairs))
        d = pd.read_csv(path, usecols=COLS)
        at_quote = (d["spread_ticks"] >= 1).to_numpy() & \
            ((d["ES"] / d["tick"] - d["spread_ticks"]).abs() < 0.01).to_numpy()
        d = d[at_quote]
        parts.append(pd.DataFrame({
            "pair": np.int8(code), "day": np.int32(day),
            "qty": d["qty"].astype("float64"),
            "depth": d["depth_touch"].astype("float32"),
            "k": d["spread_ticks"].clip(upper=100).astype("int8"),
            "tick": d["tick"].astype("float64"),
            "R10": (d["RS10"] / (2 * d["tick"])).astype("float32"),
            "R1": (d["RS1"] / (2 * d["tick"])).astype("float32"),
            "rv60": d["rv60"].astype("float32")}))
        print(pair, day, end="\r", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["pairday"] = df["pair"].astype("int64") * 100000 + df["day"]
    for col, new in (("rv60", "rvbin"), ("depth", "dbin")):
        bins = np.full(len(df), -1, dtype="int8")
        for code in range(len(pairs)):
            m = (df["pair"] == code).to_numpy()
            q = pd.qcut(df.loc[m, col], 10, labels=False, duplicates="drop")
            bins[m] = q.fillna(-1).to_numpy().astype("int8")
        df[new] = bins
    size = (df["qty"] / df["depth"].astype(float)).to_numpy()
    df["rbin"] = np.digitize(size, SIZE_EDGES).astype("int8")
    return df, {v: k for k, v in pairs.items()}


def _codes(values, drop_first):
    """Integer codes of a categorical effect; the first level is dropped when asked."""
    levels, inv = np.unique(values, return_inverse=True)
    return inv.astype(np.int32), len(levels) - (1 if drop_first else 0), drop_first


def fit(df, idx, y, regressors, with_controls):
    """Least squares of y on the regressors, pair-by-day effects and, if asked, the controls.

    Standard errors are clustered by calendar day (CR1 correction). Returns the
    coefficients and covariance of the regressors, the number of fills and the
    number of day-clusters.
    """
    n = len(idx)
    arrs = [np.asarray(a)[idx] for a in regressors]
    day_code = np.unique(df["day"].to_numpy()[idx], return_inverse=True)[1].astype(np.int32)
    n_days = int(day_code.max()) + 1
    blocks = [_codes(df["pairday"].to_numpy()[idx], False)]
    if with_controls:
        blocks += [_codes(df[f].to_numpy()[idx], True) for f in CONTROLS]
    offsets, width = [], len(arrs)
    for _, n_cols, _ in blocks:
        offsets.append(width)
        width += n_cols

    def design(a, b):
        m = b - a
        rows, cols, vals = [], [], []
        for j, arr in enumerate(arrs):
            part = arr[a:b]
            nz = np.flatnonzero(part)
            rows.append(nz)
            cols.append(np.full(len(nz), j, dtype=np.int64))
            vals.append(part[nz].astype(float))
        for (codes, _, drop), off in zip(blocks, offsets):
            cc = codes[a:b]
            use = cc >= 1 if drop else np.ones(m, bool)
            rows.append(np.arange(m)[use])
            cols.append(off + cc[use] - (1 if drop else 0))
            vals.append(np.ones(int(use.sum())))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(m, width))

    xtx, xty = np.zeros((width, width)), np.zeros(width)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        x = design(a, b)
        xtx += (x.T @ x).toarray()
        xty += x.T @ y[idx[a:b]].astype(float)
    beta = np.linalg.solve(xtx, xty)
    scores = np.zeros((n_days, width))
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        x = design(a, b)
        resid = y[idx[a:b]].astype(float) - x @ beta
        xe = sparse.csr_matrix(x.multiply(resid[:, None]))
        by_day = sparse.csr_matrix((np.ones(b - a), (day_code[a:b], np.arange(b - a))),
                                   shape=(n_days, b - a))
        scores += np.asarray((by_day @ xe).todense())
    inv = np.linalg.inv(xtx)
    cov = inv @ (scores.T @ scores) @ inv * (n_days / (n_days - 1) * (n - 1) / (n - width))
    m = len(arrs)
    return beta[:m], cov[:m, :m], n, n_days
