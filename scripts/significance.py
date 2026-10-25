"""Paired t-test (profit-aware minus MSE) over the PSEi constituents.

Reads a benchmark CSV written by benchmark.py (one row per symbol x loss, or
per symbol x loss x fold in walk-forward mode -- folds are averaged per symbol
first), pairs profit-aware with MSE per stock, and runs a two-sided paired
t-test for each metric. The PSEi index row is a market reference, not a
stock, so it is left out of the test. Writes a CSV table and a dot-and-whisker
chart of the mean paired difference with its 95% confidence interval.

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

from scripts.plot_loss_comparison import latest_csv, stocks_only
from utils.chart_style import INK, INK_MUTED, SURFACE, apply_style, footnote, percent_axis, save, zero_line

METRICS = {
    "cum_profit_geo": "Geometric return",
    "sharpe_like": "Sharpe-like ratio",
    "directional_acc": "Directional accuracy",
}
PERCENT_METRICS = {"cum_profit_geo", "directional_acc"}


def paired_diffs(path: Path) -> pd.DataFrame:
    """Per-stock (profit-aware minus MSE) for each metric."""
    d = stocks_only(pd.read_csv(path))
    per_symbol = d.groupby(["symbol", "loss_type"], as_index=False)[list(METRICS)].mean()
    mse = per_symbol[per_symbol.loss_type == "mse"].set_index("symbol")
    pa = per_symbol[per_symbol.loss_type == "profit-aware"].set_index("symbol")
    return (pa[list(METRICS)] - mse[list(METRICS)]).dropna()


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired t-test, profit-aware minus MSE, over the PSEi constituents")
    ap.add_argument("--csv", default=None, help="benchmark_summary CSV (default: latest results/benchmark_summary_ph_*.csv)")
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    print(f"Input: {csv_path}\n")
    diffs = paired_diffs(csv_path)
    n = len(diffs)

    rows = []
    for col, label in METRICS.items():
        x = diffs[col]
        t = stats.ttest_1samp(x, 0)
        half = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
        rows.append({
            "metric": label, "n": n,
            "mean_diff": x.mean(), "ci95_lo": x.mean() - half, "ci95_hi": x.mean() + half,
            "t": t.statistic, "p_two_sided": t.pvalue,
        })
    table = pd.DataFrame(rows)

    print(f"Paired t-test, profit-aware minus MSE (two-sided), n = {n} stocks\n")
    for r in table.itertuples():
        star = " *" if r.p_two_sided < 0.05 else ""
        print(f"  {r.metric:<22} mean diff {r.mean_diff:+.4f}  95% CI [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]  "
              f"t={r.t:+.3f}  p={r.p_two_sided:.3f}{star}")
    print("\n* = p < 0.05")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)

    apply_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 2.4))
    for ax, (col, r) in zip(axes, zip(METRICS, table.itertuples())):
        ax.hlines(0, r.ci95_lo, r.ci95_hi, color=INK, linewidth=2, zorder=3)
        ax.scatter([r.mean_diff], [0], s=48, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
        star = "*" if r.p_two_sided < 0.05 else ""
        ax.annotate(f"p = {r.p_two_sided:.3f}{star}", (r.ci95_hi, 0), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_MUTED)
        zero_line(ax, axis="x")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_yticks([])
        if col in PERCENT_METRICS:
            percent_axis(ax, axis="x")
        ax.margins(x=0.35)
        ax.set_ylim(-1, 1)
        ax.set_title(r.metric, loc="left", fontsize=10)
    fig.suptitle(f"Profit-aware minus MSE across {n} PSEi stocks: mean paired difference with 95% CI",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    footnote(fig, f"Two-sided paired t-test across stocks (PSEi index excluded); * p < 0.05. Source: {csv_path.name}")
    save(fig, out.with_suffix(".png"))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
