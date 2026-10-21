"""Paired t-test (profit-aware minus MSE) per market and combined.

Reads the benchmark CSVs written by benchmark.py (one row per symbol x loss),
pairs profit-aware with mse per symbol, and runs a two-sided paired t-test.
Writes a CSV table and a chart next to the inputs.

Usage:
    .venv/Scripts/python.exe significance.py                 # latest top-level us/ph CSVs
    .venv/Scripts/python.exe significance.py --us X.csv --ph Y.csv --out results/significance
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = {
    "cum_profit_geo": "Geometric return",
    "sharpe_like": "Sharpe",
    "directional_acc": "Directional accuracy",
}


def latest(market: str) -> str:
    files = sorted(glob.glob(f"results/benchmark_summary_{market}_*.csv"))
    if not files:
        raise SystemExit(f"no results/benchmark_summary_{market}_*.csv found")
    return files[-1]


def paired_diffs(path: str, market: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["symbol"] = market + ":" + d["symbol"].astype(str)
    mse = d[d.loss_type == "mse"].set_index("symbol")
    pa = d[d.loss_type == "profit-aware"].set_index("symbol")
    return (pa[list(METRICS)] - mse[list(METRICS)]).dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", default=None)
    ap.add_argument("--ph", default=None)
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    us = paired_diffs(args.us or latest("us"), "US")
    ph = paired_diffs(args.ph or latest("ph"), "PH")
    groups = {"US": us, "PH": ph, "Combined": pd.concat([us, ph])}

    rows = []
    for g, diff in groups.items():
        for col, label in METRICS.items():
            x = diff[col]
            t = stats.ttest_1samp(x, 0)
            half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({
                "group": g, "metric": label, "n": len(x),
                "mean_diff": x.mean(), "ci95_lo": x.mean() - half, "ci95_hi": x.mean() + half,
                "t": t.statistic, "p_two_sided": t.pvalue,
            })
    table = pd.DataFrame(rows)

    print("Paired t-test, profit-aware minus MSE (two-sided)\n")
    for label in METRICS.values():
        print(label)
        sub = table[table.metric == label]
        for r in sub.itertuples():
            star = " *" if r.p_two_sided < 0.05 else ""
            print(f"  {r.group:9s} n={r.n:2d}  mean diff {r.mean_diff:+.4f}  "
                  f"95% CI [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]  p={r.p_two_sided:.3f}{star}")
        print()
    print("* = p < 0.05")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out + ".csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, label in zip(axes, METRICS.values()):
        sub = table[table.metric == label].reset_index(drop=True)
        err = [sub.mean_diff - sub.ci95_lo, sub.ci95_hi - sub.mean_diff]
        ax.bar(sub.group, sub.mean_diff, yerr=err, capsize=4,
               color=["#4c78a8", "#f58518", "#54a24b"])
        ax.axhline(0, color="black", lw=0.8)
        for i, r in sub.iterrows():
            ax.text(i, ax.get_ylim()[1], f"p={r.p_two_sided:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label, pad=18)
        ax.set_ylabel("profit-aware - MSE")
    fig.suptitle("Mean paired difference with 95% CI (two-sided paired t-test)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
