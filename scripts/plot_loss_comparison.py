"""Profit-aware vs MSE comparison charts for the PSEi benchmark, by tier.

Reads benchmark.py's per-symbol output (columns: symbol, tier, loss_type,
cum_profit, cum_profit_geo, sharpe_like, rmse, directional_acc, buy_hold, and
the classic-strategy columns) and writes one figure per metric. Every grouped
chart has one group per tier plus an "All" group, with the symbol count under
each label; all numbers are means across symbols computed at plot time --
nothing is hardcoded.

Figures written (into --output-dir):
  profit_comparison.png          additive cumulative return
  profit_geo_comparison.png      geometric (compounding) return
  buy_hold_comparison.png        geometric return vs buy & hold -- buy_hold is
                                 prod(1+r)-1, the same compounding formula as
                                 cum_profit_geo, so the two are comparable
  sharpe_comparison.png          Sharpe-like ratio
  rmse_comparison.png            RMSE of the predicted return
  directional_acc_comparison.png directional accuracy, with the 50% chance line
  strategy_comparison.png        all-universe mean geometric return and Sharpe
                                 for both GRUs, buy & hold and the TA baselines
  per_symbol_geo_diff.png        profit-aware minus MSE geometric return, per symbol

Called standalone on a benchmark CSV, or from benchmark.py via --charts.

Usage:
    python scripts/plot_loss_comparison.py                       # latest results/benchmark_summary_ph_*.csv
    python scripts/plot_loss_comparison.py --csv results/benchmark_summary_ph_20261025_120000.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from utils.chart_style import (
    COLOR_MSE,
    COLOR_PA,
    COLOR_REF,
    INK_MUTED,
    LABEL_MSE,
    LABEL_PA,
    apply_style,
    footnote,
    label_bars,
    legend_above,
    percent_axis,
    save,
    title_block,
    zero_line,
)

TIER_ORDER = ("strong", "mid", "weak", "index")
TIER_LABELS = {"all": "All", "strong": "Strong", "mid": "Mid", "weak": "Weak", "index": "PSEi index"}
STRATEGY_LABELS = {"sma_crossover": "SMA crossover", "rsi": "RSI", "macd": "MACD"}


def latest_csv(results_dir: Path = Path("results")) -> Path:
    matches = sorted(results_dir.glob("benchmark_summary_ph_*.csv"))
    if not matches:
        raise SystemExit(f"No benchmark_summary_ph_*.csv found in {results_dir}")
    return matches[-1]


def tier_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """[("All", every row), ("Strong", ...), ...] for the tiers present in ``df``."""
    groups = [("all", df)]
    groups += [(t, df[df["tier"] == t]) for t in TIER_ORDER if (df["tier"] == t).any()]
    return groups


def group_label(key: str, sub: pd.DataFrame) -> str:
    return f"{TIER_LABELS.get(key, key)}\nn = {sub['symbol'].nunique()}"


def grouped_bars(
    df: pd.DataFrame,
    series: list[tuple[str, str, str | None, str]],
    title: str,
    subtitle: str,
    ylabel: str,
    value_fmt: str,
    percent: bool = False,
    reference_line: tuple[float, str] | None = None,
):
    """series: (label, column, loss_type or None, color). One bar per series per tier group."""
    groups = tier_groups(df)
    n = len(series)
    width = min(0.8 / n, 0.26)
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (label, column, loss_type, color) in enumerate(series):
        vals = []
        for _, sub in groups:
            if loss_type is not None:
                sub = sub[sub["loss_type"] == loss_type]
            elif "loss_type" in sub:
                # reference columns (buy_hold, strategies) repeat per loss_type; count each symbol once
                sub = sub[sub["loss_type"] == "mse"]
            vals.append(float(sub[column].mean()) if len(sub) else np.nan)
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=label, color=color, zorder=3)
        label_bars(ax, bars, value_fmt)

    if reference_line is not None:
        ref_val, ref_label = reference_line
        ax.axhline(ref_val, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2, label=ref_label)
    zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([group_label(k, sub) for k, sub in groups])
    ax.set_ylabel(ylabel)
    if percent:
        percent_axis(ax)
    ax.margins(y=0.15)
    legend_above(ax)
    title_block(ax, title, subtitle)
    fig.tight_layout()
    return fig


def strategy_figure(df: pd.DataFrame, subtitle: str):
    """Horizontal bars: universe-mean geometric return and Sharpe for every strategy."""
    mse = df[df["loss_type"] == "mse"]
    pa = df[df["loss_type"] == "profit-aware"]
    rows = [
        (LABEL_PA, pa["cum_profit_geo"].mean(), pa["sharpe_like"].mean(), COLOR_PA),
        (LABEL_MSE, mse["cum_profit_geo"].mean(), mse["sharpe_like"].mean(), COLOR_MSE),
        ("Buy & hold", mse["buy_hold"].mean(), np.nan, COLOR_REF),
    ]
    for key, label in STRATEGY_LABELS.items():
        if f"{key}_geo_ret" in mse:
            rows.append((label, mse[f"{key}_geo_ret"].mean(), mse[f"{key}_sharpe"].mean(), COLOR_REF))

    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, idx, title, fmt, pct in (
        (axes[0], 1, "Mean geometric return", "{:+.1%}", True),
        (axes[1], 2, "Mean Sharpe-like ratio", "{:+.2f}", False),
    ):
        vals = [r[idx] for r in rows]
        bars = ax.barh(y, vals, height=0.55, color=[r[3] for r in rows], zorder=3)
        label_bars(ax, [b for b, v in zip(bars, vals) if np.isfinite(v)], fmt, horizontal=True)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        zero_line(ax, axis="x")
        if pct:
            percent_axis(ax, axis="x")
        ax.margins(x=0.32)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.suptitle("Profit-aware GRU against the MSE baseline and classic strategies", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.905, subtitle, ha="left", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    footnote(fig, "Buy & hold has no Sharpe-like value in the benchmark output.")
    return fig


def per_symbol_diff_figure(df: pd.DataFrame, subtitle: str):
    """Profit-aware minus MSE geometric return for each symbol, sorted."""
    piv = df.pivot_table(index="symbol", columns="loss_type", values="cum_profit_geo", aggfunc="mean")
    piv = piv.dropna(subset=["mse", "profit-aware"])
    diff = (piv["profit-aware"] - piv["mse"]).sort_values()
    names = [s.split(":")[-1] for s in diff.index]
    colors = [COLOR_PA if v > 0 else COLOR_MSE for v in diff.values]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.24 * len(diff) + 1.4)))
    y = np.arange(len(diff))
    bars = ax.barh(y, diff.values, height=0.62, color=colors, zorder=3)
    label_bars(ax, bars, "{:+.1%}", horizontal=True, fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    zero_line(ax, axis="x")
    percent_axis(ax, axis="x")
    ax.margins(x=0.18, y=0.01)
    ax.set_xlabel("Geometric return difference (profit-aware minus MSE)")
    legend_above(
        ax,
        handles=[Patch(color=COLOR_PA), Patch(color=COLOR_MSE)],
        labels=[f"{LABEL_PA} higher ({int((diff > 0).sum())} symbols)",
                f"{LABEL_MSE} higher ({int((diff <= 0).sum())} symbols)"],
    )
    title_block(ax, "Per-symbol geometric return difference", subtitle)
    fig.tight_layout()
    return fig


def default_subtitle(df: pd.DataFrame) -> str:
    """Sample size plus the run's cost/lambda settings, read from the CSV when it records them."""
    text = f"{df['symbol'].nunique()} PSE symbols, held-out test window"
    if {"buy_cost", "sell_cost", "loss_lambda"} <= set(df.columns):
        r = df.iloc[0]
        text += (f"; costs: buy {r['buy_cost']:.3%}, sell {r['sell_cost']:.3%}; "
                 f"loss lambda = {r['loss_lambda']:g}")
    return text


def plot_comparisons(df: pd.DataFrame, output_dir: Path, subtitle: str | None = None) -> list[Path]:
    """Write every comparison figure for one benchmark results table. Returns the paths written."""
    apply_style()
    output_dir = Path(output_dir)
    sub = subtitle or default_subtitle(df)

    mse_pa = lambda col: [  # noqa: E731
        (LABEL_MSE, col, "mse", COLOR_MSE),
        (LABEL_PA, col, "profit-aware", COLOR_PA),
    ]
    specs = [
        ("profit_comparison.png", mse_pa("cum_profit"),
         "Cumulative return (additive) by tier", "Cumulative return", "{:+.1%}", True, None),
        ("profit_geo_comparison.png", mse_pa("cum_profit_geo"),
         "Geometric return by tier", "Geometric return", "{:+.1%}", True, None),
        ("sharpe_comparison.png", mse_pa("sharpe_like"),
         "Sharpe-like ratio by tier", "Annualised Sharpe-like ratio", "{:+.2f}", False, None),
        ("rmse_comparison.png", mse_pa("rmse"),
         "Prediction error (RMSE) by tier", "RMSE of predicted daily return", "{:.4f}", False, None),
        ("directional_acc_comparison.png", mse_pa("directional_acc"),
         "Directional accuracy by tier", "Directional accuracy", "{:.1%}", True, (0.5, "Chance (50%)")),
    ]
    if "buy_hold" in df.columns:
        specs.append((
            "buy_hold_comparison.png",
            mse_pa("cum_profit_geo") + [("Buy & hold", "buy_hold", None, COLOR_REF)],
            "Geometric return against buy & hold, by tier", "Geometric return", "{:+.1%}", True, None,
        ))

    written: list[Path] = []
    for filename, series, title, ylabel, fmt, pct, ref in specs:
        fig = grouped_bars(df, series, title, f"{sub}; mean across the symbols in each group", ylabel, fmt,
                           percent=pct, reference_line=ref)
        written.append(save(fig, output_dir / filename))

    written.append(save(strategy_figure(df, sub), output_dir / "strategy_comparison.png"))
    written.append(save(per_symbol_diff_figure(df, sub), output_dir / "per_symbol_geo_diff.png"))

    for p in written:
        print(f"Saved {p}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MSE vs profit-aware comparison charts for the PSEi benchmark")
    parser.add_argument("--csv", type=str, default=None, help="benchmark_summary CSV (default: latest results/benchmark_summary_ph_*.csv)")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <csv stem>_charts/ next to the CSV")
    parser.add_argument("--subtitle", type=str, default=None, help="Override the subtitle line on every figure")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    df = pd.read_csv(csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / f"{csv_path.stem}_charts"
    plot_comparisons(df, output_dir, args.subtitle)


if __name__ == "__main__":
    main()
