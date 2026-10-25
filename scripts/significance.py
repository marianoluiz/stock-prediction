"""Significance tests for profit-aware minus MSE on a benchmark run.

Reads a benchmark CSV written by benchmark.py. Which test runs depends on
what was benchmarked:

* **Constituents in the run** (``benchmark.py --stocks``): a two-sided paired
  t-test across the stocks (folds averaged per stock first) for geometric
  return, Sharpe-like ratio and directional accuracy. The PSEi index row is a
  market reference, not a stock, so it is left out.

* **PSEi index only** (the default run): there is one symbol, so stocks can't
  be the sample. Two tests instead:

  1. **Daily paired test** on the run's ``_daily.csv``: for every out-of-sample
     test day, the difference in net (after-cost) return, and in directional
     hit, between profit-aware and MSE. The mean difference is tested against
     0 with a Newey-West (HAC) standard error, because daily trading returns
     are autocorrelated (positions persist across days). Each model's hit
     rate is also tested against 50% (two-sided binomial test).
  2. **Fold-level count**: on how many walk-forward folds profit-aware beats
     MSE, with a two-sided sign test. With only 4-6 folds this has very low
     power -- it is reported for transparency, not as the main evidence.

Writes ``<out>.csv`` (every test as a row) and ``<out>.png`` (mean paired
difference with its 95% confidence interval).

Usage:
    .venv/Scripts/python.exe scripts/significance.py                  # newest results/benchmark_summary_*.csv
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

from scripts.plot_loss_comparison import is_index_only, latest_csv, stocks_only
from utils.chart_style import INK, INK_MUTED, SURFACE, apply_style, footnote, percent_axis, save, zero_line

METRICS = {
    "cum_profit_geo": "Geometric return",
    "sharpe_like": "Sharpe-like ratio",
    "directional_acc": "Directional accuracy",
}
PERCENT_METRICS = {"cum_profit_geo", "directional_acc", "daily_net_return", "daily_hit"}
TRADING_DAYS = 252


# --- Stock mode -------------------------------------------------------------------

def paired_diffs(path: Path) -> pd.DataFrame:
    """Per-stock (profit-aware minus MSE) for each metric."""
    d = stocks_only(pd.read_csv(path))
    per_symbol = d.groupby(["symbol", "loss_type"], as_index=False)[list(METRICS)].mean()
    mse = per_symbol[per_symbol.loss_type == "mse"].set_index("symbol")
    pa = per_symbol[per_symbol.loss_type == "profit-aware"].set_index("symbol")
    return (pa[list(METRICS)] - mse[list(METRICS)]).dropna()


def stock_tests(csv_path: Path) -> tuple[pd.DataFrame, str]:
    diffs = paired_diffs(csv_path)
    n = len(diffs)
    rows = []
    for col, label in METRICS.items():
        x = diffs[col]
        t = stats.ttest_1samp(x, 0)
        half = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
        rows.append({
            "test": "paired t-test across stocks", "key": col, "metric": label, "n": n,
            "mean_diff": x.mean(), "ci95_lo": x.mean() - half, "ci95_hi": x.mean() + half,
            "t": t.statistic, "p_two_sided": t.pvalue,
        })
    return pd.DataFrame(rows), f"across {n} PSEi stocks"


# --- Index mode -------------------------------------------------------------------

def newey_west_test(x: np.ndarray, lags: int | None = None) -> tuple[float, float, float, float]:
    """Mean of ``x`` tested against 0 with a Newey-West (Bartlett kernel) HAC standard error.

    Returns ``(mean, se, t, p_two_sided)``. ``lags`` defaults to the usual
    ``floor(4 * (n / 100) ** (2 / 9))`` rule of thumb.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(var, 0.0) / n))
    t = float(x.mean() / se) if se > 0 else 0.0
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return float(x.mean()), se, t, p


def index_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per test day: each model's net return and directional hit side by side."""
    d = daily[daily["model"].isin(["mse", "profit-aware"])]
    wide = d.pivot_table(index=["symbol", "fold", "date"], columns="model",
                         values=["net_return", "position", "actual_return"]).dropna()
    out = pd.DataFrame(index=wide.index)
    actual = wide[("actual_return", "mse")]
    for model in ("mse", "profit-aware"):
        out[f"net_{model}"] = wide[("net_return", model)]
        # Same rule as utils.metrics.directional_accuracy: sign(position) == sign(actual).
        # position = tanh(alpha * pred) has the sign of pred.
        out[f"hit_{model}"] = (np.sign(wide[("position", model)]) == np.sign(actual)).astype(float)
    return out.reset_index()


def index_tests(csv_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    rows: list[dict] = []
    n_days = 0

    daily_path = csv_path.with_name(csv_path.stem + "_daily.csv")
    if daily_path.exists():
        days = index_daily_frame(pd.read_csv(daily_path))
        n_days = len(days)
        for key, label, a, b in (
            ("daily_net_return", "Daily net return (annualised)", "net_profit-aware", "net_mse"),
            ("daily_hit", "Directional hit rate", "hit_profit-aware", "hit_mse"),
        ):
            mean, se, t, p = newey_west_test(days[a].to_numpy() - days[b].to_numpy())
            scale = TRADING_DAYS if key == "daily_net_return" else 1.0
            half = stats.t.ppf(0.975, n_days - 1) * se
            rows.append({
                "test": "daily paired, Newey-West HAC", "key": key, "metric": label, "n": n_days,
                "mean_diff": mean * scale, "ci95_lo": (mean - half) * scale, "ci95_hi": (mean + half) * scale,
                "t": t, "p_two_sided": p,
            })
        for model in ("mse", "profit-aware"):
            hits = int(days[f"hit_{model}"].sum())
            res = stats.binomtest(hits, n_days, 0.5)
            ci = res.proportion_ci(0.95)
            rows.append({
                "test": "hit rate vs 50%, binomial", "key": f"hit_{model}", "metric": f"Hit rate, {model}",
                "n": n_days, "mean_diff": hits / n_days - 0.5, "ci95_lo": ci.low - 0.5, "ci95_hi": ci.high - 0.5,
                "t": np.nan, "p_two_sided": res.pvalue,
            })
    else:
        print(f"Note: {daily_path.name} not found (older benchmark run); only the fold-level test is available.")

    piv = df.pivot_table(index="fold", columns="loss_type", values=list(METRICS), aggfunc="mean")
    n_folds = len(piv)
    for col, label in METRICS.items():
        diff = (piv[(col, "profit-aware")] - piv[(col, "mse")]).dropna()
        wins = int((diff > 0).sum())
        nonzero = int((diff != 0).sum())
        p = stats.binomtest(wins, nonzero, 0.5).pvalue if nonzero else 1.0
        rows.append({
            "test": "fold sign test", "key": f"fold_{col}", "metric": f"{label}, folds won",
            "n": n_folds, "mean_diff": diff.mean(), "ci95_lo": np.nan, "ci95_hi": np.nan,
            "t": np.nan, "p_two_sided": p, "wins": wins,
        })
    unit = f"on the PSEi ({n_days} test days, {n_folds} fold{'s' if n_folds != 1 else ''})"
    return pd.DataFrame(rows), unit


# --- Output ----------------------------------------------------------------------

def plot(table: pd.DataFrame, unit: str, csv_name: str, out: Path) -> None:
    """Dot-and-whisker of every test that has a confidence interval."""
    plotted = table.dropna(subset=["ci95_lo"]).reset_index(drop=True)
    apply_style()
    fig, axes = plt.subplots(1, len(plotted), figsize=(3.7 * len(plotted), 2.4), squeeze=False)
    for ax, r in zip(axes[0], plotted.itertuples()):
        ax.hlines(0, r.ci95_lo, r.ci95_hi, color=INK, linewidth=2, zorder=3)
        ax.scatter([r.mean_diff], [0], s=48, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
        star = "*" if r.p_two_sided < 0.05 else ""
        ax.annotate(f"p = {r.p_two_sided:.3f}{star}", (r.ci95_hi, 0), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_MUTED)
        zero_line(ax, axis="x")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_yticks([])
        if r.key in PERCENT_METRICS or r.key.startswith("hit_"):
            percent_axis(ax, axis="x", decimals=1)
        ax.margins(x=0.35)
        ax.set_ylim(-1, 1)
        title = r.metric if not r.key.startswith("hit_") else f"{r.metric} minus 50%"
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(f"Profit-aware minus MSE {unit}: mean difference with 95% CI",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    footnote(fig, f"Two-sided tests; * p < 0.05. Source: {csv_name}")
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Significance tests, profit-aware minus MSE")
    ap.add_argument("--csv", default=None, help="benchmark_summary CSV (default: newest results/benchmark_summary_*.csv)")
    ap.add_argument("--out", default="results/significance")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    print(f"Input: {csv_path}\n")
    df = pd.read_csv(csv_path)
    table, unit = index_tests(csv_path) if is_index_only(df) else stock_tests(csv_path)

    print(f"Profit-aware minus MSE {unit}\n")
    for r in table.itertuples():
        star = " *" if r.p_two_sided < 0.05 else ""
        ci = f"  95% CI [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]" if np.isfinite(r.ci95_lo) else ""
        t = f"  t={r.t:+.3f}" if np.isfinite(r.t) else ""
        wins = f"  wins {int(r.wins)}/{r.n}" if "wins" in table and np.isfinite(getattr(r, "wins", np.nan)) else ""
        print(f"  [{r.test}] {r.metric:<34} n={r.n:<5} mean {r.mean_diff:+.4f}{ci}{t}{wins}  "
              f"p={r.p_two_sided:.3f}{star}")
    print("\n* = p < 0.05")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    plot(table, unit, csv_path.name, out.with_suffix(".png"))
    print(f"\nwrote {out.with_suffix('.csv')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
