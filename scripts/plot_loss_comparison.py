"""Profit-aware vs MSE comparison charts for the PSEi benchmark.

Reads benchmark.py's per-symbol output (columns: symbol, loss_type,
cum_profit, cum_profit_geo, sharpe_like, rmse, directional_acc, buy_hold, and
the classic-strategy columns) and writes one figure per metric. Groups on
the grouped charts depend on what was benchmarked:

* with constituents: the PSEi constituents (mean across the stocks), then the
  PSEi index as a separate market reference;
* PSEi index only, walk-forward: one group per fold (its test window), then
  the mean across folds;
* PSEi index only, single split: one group, the index's test window.

All numbers are computed from the CSV at plot time -- nothing is hardcoded.

Figures written (into --output-dir):
  profit_comparison.png          additive cumulative return
  profit_geo_comparison.png      geometric (compounding) return
  buy_hold_comparison.png        geometric return vs buy & hold -- buy_hold is
                                 prod(1+r)-1, the same compounding formula as
                                 cum_profit_geo, so the two are comparable
  sharpe_comparison.png          Sharpe-like ratio
  rmse_comparison.png            RMSE of the predicted return
  directional_acc_comparison.png directional accuracy, with the 50% chance line
  strategy_comparison.png        mean geometric return and Sharpe across the
                                 stocks (or index folds) for both GRUs, buy & hold
                                 and the TA baselines
  per_symbol_geo_diff.png        profit-aware minus MSE geometric return, per stock
                                 (only when constituents were benchmarked)
  per_fold_geo_diff.png          profit-aware minus MSE geometric return, per index
                                 fold (index-only walk-forward runs)
  equity_curve.png               growth of 1 PHP over the test period(s) for every
                                 model and baseline, from the run's _daily.csv
                                 (index-only runs; folds are stitched end to end)

Called standalone on a benchmark CSV, or from benchmark.py via --charts.

Usage:
    python scripts/plot_loss_comparison.py                       # newest results/benchmark_summary_*.csv
    python scripts/plot_loss_comparison.py --csv results/benchmark_summary_psei_20261025_120000.csv
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

from benchmark import INDEX_SYMBOL
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

STRATEGY_LABELS = {"sma_crossover": "SMA crossover", "rsi": "RSI", "macd": "MACD"}


def latest_csv(results_dir: Path = Path("results")) -> Path:
    """Newest benchmark results CSV (index-only ``_psei_``, full ``_ph_`` or ``_custom_``)."""
    matches = [p for p in results_dir.glob("benchmark_summary_*.csv") if not p.stem.endswith("_daily")]
    if not matches:
        raise SystemExit(f"No benchmark_summary_*.csv found in {results_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def stocks_only(df: pd.DataFrame) -> pd.DataFrame:
    """The constituent stocks, without the PSEi index reference row."""
    return df[df["symbol"] != INDEX_SYMBOL]


def is_index_only(df: pd.DataFrame) -> bool:
    return not len(stocks_only(df))


def fold_label(rows: pd.DataFrame) -> str:
    """'Fold k' plus its test window as 'Mon YYYY - Mon YYYY' when the CSV records it."""
    fold = int(rows["fold"].iloc[0])
    if "test_start" in rows and isinstance(rows["test_start"].iloc[0], str):
        fmt = lambda d: pd.Timestamp(d).strftime("%b %Y")  # noqa: E731
        return f"Fold {fold}\n{fmt(rows['test_start'].iloc[0])} - {fmt(rows['test_end'].iloc[0])}"
    return f"Fold {fold}"


def model_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """[(label, rows)] for the grouped bar charts (see the module docstring)."""
    stocks = stocks_only(df)
    index = df[df["symbol"] == INDEX_SYMBOL]
    if not len(stocks):
        if index["fold"].nunique() > 1:
            groups = [(fold_label(rows), rows) for _, rows in index.groupby("fold")]
            return groups + [(f"Mean of\n{index['fold'].nunique()} folds", index)]
        return [("PSEi index\ntest window", index)]
    groups = [(f"PSEi constituents\nmean of {stocks['symbol'].nunique()} stocks", stocks)]
    if len(index):
        groups.append(("PSEi index\nmarket reference", index))
    return groups


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
    """series: (label, column, loss_type or None, color). One bar per series per group."""
    groups = model_groups(df)
    n = len(series)
    width = min(0.8 / n, 0.26)
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(groups) + 1.5), 4.6))
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
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_ylabel(ylabel)
    if percent:
        percent_axis(ax)
    ax.margins(y=0.15)
    legend_above(ax)
    title_block(ax, title, subtitle)
    fig.tight_layout()
    return fig


def strategy_figure(df: pd.DataFrame, subtitle: str):
    """Horizontal bars: mean geometric return and Sharpe across the stocks (or the
    index's folds, for an index-only run) for every strategy."""
    stocks = stocks_only(df)
    if not len(stocks):
        stocks = df
    mse = stocks[stocks["loss_type"] == "mse"]
    pa = stocks[stocks["loss_type"] == "profit-aware"]
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
    unit = "across the stocks" if len(stocks_only(df)) else (
        f"across {df['fold'].nunique()} PSEi folds" if df["fold"].nunique() > 1 else "on the PSEi")
    fig.suptitle(f"Profit-aware GRU against the MSE baseline and classic strategies, {unit}", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.text(0.01, 0.905, subtitle, ha="left", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    footnote(fig, "Buy & hold has no Sharpe-like value in the benchmark output.")
    return fig


def per_symbol_diff_figure(df: pd.DataFrame, subtitle: str):
    """Profit-aware minus MSE geometric return for each stock, sorted."""
    piv = stocks_only(df).pivot_table(index="symbol", columns="loss_type", values="cum_profit_geo", aggfunc="mean")
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
        labels=[f"{LABEL_PA} higher ({int((diff > 0).sum())} stocks)",
                f"{LABEL_MSE} higher ({int((diff <= 0).sum())} stocks)"],
    )
    title_block(ax, "Per-stock geometric return difference", subtitle)
    fig.tight_layout()
    return fig


def _n_folds(n: int) -> str:
    return f"{n} fold" if n == 1 else f"{n} folds"


def per_fold_diff_figure(df: pd.DataFrame, subtitle: str):
    """Profit-aware minus MSE geometric return for each index fold, next to that fold's buy & hold."""
    index = df[df["symbol"] == INDEX_SYMBOL]
    piv = index.pivot_table(index="fold", columns="loss_type", values="cum_profit_geo", aggfunc="mean")
    diff = piv["profit-aware"] - piv["mse"]
    labels = [fold_label(index[index["fold"] == f]).replace("\n", "  ") for f in diff.index]
    bh = index[index["loss_type"] == "mse"].set_index("fold")["buy_hold"].reindex(diff.index)

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(diff) + 2.0))
    y = np.arange(len(diff))[::-1]
    bars = ax.barh(y, diff.values, height=0.55,
                   color=[COLOR_PA if v > 0 else COLOR_MSE for v in diff.values], zorder=3)
    label_bars(ax, bars, "{:+.1%}", horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl}\nbuy & hold {b:+.1%}" for lbl, b in zip(labels, bh.values)], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    zero_line(ax, axis="x")
    percent_axis(ax, axis="x", decimals=1)
    ax.margins(x=0.25)
    ax.set_xlabel("Geometric return difference (profit-aware minus MSE)")
    legend_above(
        ax,
        handles=[Patch(color=COLOR_PA), Patch(color=COLOR_MSE)],
        labels=[f"{LABEL_PA} higher ({_n_folds(int((diff > 0).sum()))})",
                f"{LABEL_MSE} higher ({_n_folds(int((diff <= 0).sum()))})"],
    )
    title_block(ax, "PSEi: geometric return difference per walk-forward fold", subtitle)
    fig.tight_layout()
    return fig


# (daily-CSV model name, legend label, color, line style, line width)
EQUITY_SERIES = [
    ("profit-aware", LABEL_PA, COLOR_PA, "-", 2.2),
    ("mse", LABEL_MSE, COLOR_MSE, "-", 2.2),
    ("buy_hold", "Buy & hold (no cost)", COLOR_REF, "-", 1.6),
    ("sma_crossover", STRATEGY_LABELS["sma_crossover"], COLOR_REF, (0, (4, 2)), 1.0),
    ("rsi", STRATEGY_LABELS["rsi"], COLOR_REF, (0, (1, 1.5)), 1.0),
    ("macd", STRATEGY_LABELS["macd"], COLOR_REF, (0, (6, 2, 1, 2)), 1.0),
]


def equity_curve_figure(daily: pd.DataFrame, subtitle: str):
    """Growth of 1 PHP for every model and baseline over the index's test period.

    Walk-forward folds are contiguous, so the per-fold test windows are
    stitched end to end into one out-of-sample period (each fold's model
    trades only its own window). Fold boundaries are marked.
    """
    d = daily[daily["symbol"] == INDEX_SYMBOL].copy()
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, label, color, ls, lw in EQUITY_SERIES:
        sub = d[d["model"] == model].sort_values("date")
        if not len(sub):
            continue
        curve = np.cumprod(1.0 + sub["net_return"].to_numpy())
        ax.plot(sub["date"], curve, color=color, linestyle=ls, linewidth=lw,
                label=f"{label} ({curve[-1] - 1:+.1%})", zorder=3 if model in ("mse", "profit-aware") else 2)
    starts = d.groupby("fold")["date"].min().sort_values().iloc[1:]
    for x in starts:
        ax.axvline(x, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    ax.axhline(1.0, color=INK_MUTED, linewidth=0.9, zorder=1)
    ax.set_ylabel("Value of 1 PHP")
    ax.margins(x=0.01)
    legend_above(ax, ncol=3)
    title_block(ax, "PSEi: growth of 1 PHP over the out-of-sample test period", subtitle)
    fig.tight_layout()
    if len(starts):
        footnote(fig, "Dotted vertical lines are walk-forward fold boundaries; each fold's model trades only its "
                      "own window. Costs applied to every policy except buy & hold.")
    return fig


def default_subtitle(df: pd.DataFrame) -> str:
    """Sample size plus the run's cost/lambda settings, read from the CSV when it records them."""
    universe = "PSEi index" if is_index_only(df) else "PSEi constituents"
    window = "walk-forward test windows" if df["fold"].max() > 0 else "held-out test window"
    text = f"{universe}, {window}"
    if {"buy_cost", "sell_cost", "loss_lambda"} <= set(df.columns):
        r = df.iloc[0]
        text += (f"; costs: buy {r['buy_cost']:.3%}, sell {r['sell_cost']:.3%}; "
                 f"loss lambda = {r['loss_lambda']:g}")
    return text


def plot_comparisons(
    df: pd.DataFrame, output_dir: Path, subtitle: str | None = None, daily: pd.DataFrame | None = None,
) -> list[Path]:
    """Write every comparison figure for one benchmark results table. Returns the paths written.

    ``daily`` is the run's ``_daily.csv`` (per-day positions/returns); when
    given, an index-only run also gets the equity-curve figure.
    """
    apply_style()
    output_dir = Path(output_dir)
    sub = subtitle or default_subtitle(df)

    mse_pa = lambda col: [  # noqa: E731
        (LABEL_MSE, col, "mse", COLOR_MSE),
        (LABEL_PA, col, "profit-aware", COLOR_PA),
    ]
    specs = [
        ("profit_comparison.png", mse_pa("cum_profit"),
         "Cumulative return (additive)", "Cumulative return", "{:+.1%}", True, None),
        ("profit_geo_comparison.png", mse_pa("cum_profit_geo"),
         "Geometric return", "Geometric return", "{:+.1%}", True, None),
        ("sharpe_comparison.png", mse_pa("sharpe_like"),
         "Sharpe-like ratio", "Annualised Sharpe-like ratio", "{:+.2f}", False, None),
        ("rmse_comparison.png", mse_pa("rmse"),
         "Prediction error (RMSE)", "RMSE of predicted daily return", "{:.4f}", False, None),
        ("directional_acc_comparison.png", mse_pa("directional_acc"),
         "Directional accuracy", "Directional accuracy", "{:.1%}", True, (0.5, "Chance (50%)")),
    ]
    if "buy_hold" in df.columns:
        specs.append((
            "buy_hold_comparison.png",
            mse_pa("cum_profit_geo") + [("Buy & hold", "buy_hold", None, COLOR_REF)],
            "Geometric return against buy & hold", "Geometric return", "{:+.1%}", True, None,
        ))

    written: list[Path] = []
    for filename, series, title, ylabel, fmt, pct, ref in specs:
        fig = grouped_bars(df, series, title, sub, ylabel, fmt,
                           percent=pct, reference_line=ref)
        written.append(save(fig, output_dir / filename))

    written.append(save(strategy_figure(df, sub), output_dir / "strategy_comparison.png"))
    if len(stocks_only(df)):
        written.append(save(per_symbol_diff_figure(df, sub), output_dir / "per_symbol_geo_diff.png"))
    elif df["fold"].nunique() > 1:
        written.append(save(per_fold_diff_figure(df, sub), output_dir / "per_fold_geo_diff.png"))
    if daily is not None and is_index_only(df):
        written.append(save(equity_curve_figure(daily, sub), output_dir / "equity_curve.png"))

    for p in written:
        print(f"Saved {p}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MSE vs profit-aware comparison charts for the PSEi benchmark")
    parser.add_argument("--csv", type=str, default=None, help="benchmark_summary CSV (default: newest results/benchmark_summary_*.csv)")
    parser.add_argument("--output-dir", type=str, default=None, help="Default: <csv stem>_charts/ next to the CSV")
    parser.add_argument("--subtitle", type=str, default=None, help="Override the subtitle line on every figure")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv()
    df = pd.read_csv(csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / f"{csv_path.stem}_charts"
    daily_path = csv_path.with_name(csv_path.stem + "_daily.csv")
    daily = pd.read_csv(daily_path) if daily_path.exists() else None
    plot_comparisons(df, output_dir, args.subtitle, daily=daily)


if __name__ == "__main__":
    main()
