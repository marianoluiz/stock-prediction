"""Paired t-test (profit-aware minus MSE) over the PSEi benchmark, overall and by tier.

Reads a benchmark CSV written by benchmark.py (one row per symbol x loss, or
per symbol x loss x fold in walk-forward mode -- folds are averaged per symbol
first), pairs profit-aware with MSE per symbol, and runs a two-sided paired
t-test for each metric, over all symbols and within each performance tier.
Writes a CSV table and a dot-and-whisker chart of the mean paired difference
with its 95% confidence interval.

Usage:
    .venv/Scripts/python.exe scripts/significance.py                  # latest results/benchmark_summary_ph_*.csv
    .venv/Scripts/python.exe scripts/significance.py --csv X.csv --out results/significance
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.plot_loss_comparison import TIER_LABELS, latest_csv
from utils.chart_style import INK, INK_MUTED, SURFACE, apply_style, footnote, percent_axis, save, zero_line

METRICS = {
    "cum_profit_geo": "Geometric return",
    "sharpe_like": "Sharpe-like ratio",
    "directional_acc": "Directional accuracy",
}
PERCENT_METRICS = {"cum_profit_geo", "directional_acc"}
GROUPS = ("all", "strong", "mid", "weak")  # the lone PSEi index row is too small to test alone


def paired_diffs(path: Path) -> pd.DataFrame:
    """Per-symbol (profit-aware minus MSE) for each metric, with the symbol's tier."""
    d = pd.read_csv(path)
    per_symbol = d.groupby(["symbol", "tier", "loss_type"], as_index=False)[list(METRICS)].mean()
    mse = per_symbol[per_symbol.loss_type == "mse"].set_index("symbol")
    pa = per_symbol[per_symbol.loss_type == "profit-aware"].set_index("symbol")
    diff = (pa[list(METRICS)] - mse[list(METRICS)]).dropna()
    diff["tier"] = mse.loc[diff.index, "tier"]
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired t-test, profit-aware minus MSE, over the PSEi benchmark")
    ap.add_argument("--csv", default=None, help="benchmark_summary CSV (default: latest results/benchmark_summary_ph_*.csv)")
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    print(f"Input: {csv_path}\n")
    diffs = paired_diffs(csv_path)

    rows = []
    for g in GROUPS:
        sub = diffs if g == "all" else diffs[diffs.tier == g]
        if len(sub) < 2:
            continue
        for col, label in METRICS.items():
            x = sub[col]
            t = stats.ttest_1samp(x, 0)
            half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({
                "group": TIER_LABELS[g], "metric": label, "n": len(x),
                "mean_diff": x.mean(), "ci95_lo": x.mean() - half, "ci95_hi": x.mean() + half,
                "t": t.statistic, "p_two_sided": t.pvalue,
            })
    table = pd.DataFrame(rows)

    print("Paired t-test, profit-aware minus MSE (two-sided)\n")
    for label in METRICS.values():
        print(label)
        for r in table[table.metric == label].itertuples():
            star = " *" if r.p_two_sided < 0.05 else ""
            print(f"  {r.group:7s} n={r.n:2d}  mean diff {r.mean_diff:+.4f}  "
                  f"95% CI [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]  p={r.p_two_sided:.3f}{star}")
        print()
    print("* = p < 0.05")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)

    apply_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 3.6), sharey=True)
    for ax, (col, label) in zip(axes, METRICS.items()):
        sub = table[table.metric == label].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax.hlines(y, sub.ci95_lo, sub.ci95_hi, color=INK, linewidth=2, zorder=3)
        ax.scatter(sub.mean_diff, y, s=48, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
        for yi, r in zip(y, sub.itertuples()):
            star = "*" if r.p_two_sided < 0.05 else ""
            ax.annotate(f"p = {r.p_two_sided:.3f}{star}", (r.ci95_hi, yi), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_MUTED)
        zero_line(ax, axis="x")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        if col in PERCENT_METRICS:
            percent_axis(ax, axis="x")
        ax.margins(x=0.3, y=0.15)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.group} (n = {r.n})" for r in sub.itertuples()])
    fig.suptitle("Profit-aware minus MSE: mean paired difference with 95% confidence interval",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    footnote(fig, f"Two-sided paired t-test across symbols; * p < 0.05. Source: {csv_path.name}")
    save(fig, out.with_suffix(".png"))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
