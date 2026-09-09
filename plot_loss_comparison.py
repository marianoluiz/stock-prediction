"""Profit-aware vs MSE comparison charts, PH market vs US market.

Reads benchmark.py's own per-symbol output (columns: symbol, tier, loss_type,
cum_profit, cum_profit_geo, sharpe_like, rmse, buy_hold, ...) and plots grouped
bar charts -- one group of bars per market -- for:
  - cumulative profit, MSE vs Profit-aware (additive and geometric)
  - Sharpe-like ratio, MSE vs Profit-aware
  - RMSE, MSE vs Profit-aware
  - Buy & Hold, its own chart, compared against the models' *geometric*
    return only -- benchmark.py's buy_hold column is prod(1+returns)-1 (the
    same compounding formula as cum_profit_geo), so pairing it with the
    additive cum_profit would silently compare two different things.
All numbers are aggregated (mean across symbols) directly from the data at
plot time; nothing is hardcoded. Called both standalone (pointed at a
--lambda-dir of separate PH/US benchmark_summary CSVs) and from benchmark.py
itself via --charts (on the in-memory results of a single run).

Usage:
    python plot_loss_comparison.py --lambda-dir results/llambda_0.7
    python plot_loss_comparison.py --csv results/benchmark_summary_all_20260910_120000.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def market_of(symbol: str) -> str:
    """PH exchange symbols are prefixed 'PSE:'; everything else is US."""
    return "PH" if ":" in symbol else "US"


def load_market_csv(lambda_dir: Path, market: str) -> pd.DataFrame:
    matches = sorted(lambda_dir.glob(f"benchmark_summary_{market}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No benchmark_summary_{market}_*.csv found in {lambda_dir}")
    df = pd.read_csv(matches[-1])
    df["market"] = market.upper()
    return df


def grouped_bar(
    ax,
    markets: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
    title: str,
    reference_line: tuple[float, str] | None = None,
) -> None:
    """series: list of (label, values-per-market) bars, plotted side by side per market.
    reference_line: optional (y-value, label) for a dashed horizontal baseline, e.g. chance-level accuracy.
    """
    x = range(len(markets))
    n = len(series)
    width = 0.8 / n
    offsets = [(-0.4 + width / 2) + i * width for i in range(n)]
    for offset, (label, vals) in zip(offsets, series):
        ax.bar([i + offset for i in x], vals, width, label=label)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (i + offset, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    if reference_line is not None:
        ref_val, ref_label = reference_line
        ax.axhline(ref_val, color="gray", linewidth=1.2, linestyle="--", label=ref_label)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m} Market" for m in markets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_comparisons(df: pd.DataFrame, output_dir: Path, label: str) -> list[Path]:
    """Build the profit / profit_geo / sharpe / rmse comparison PNGs from a
    combined (multi-market, multi-loss_type) benchmark results DataFrame.
    Returns the list of paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    markets = sorted(df["market"].unique())

    def agg(column: str, loss_type: str | None = None) -> list[float]:
        vals = []
        for m in markets:
            sub = df[df["market"] == m]
            if loss_type is not None:
                sub = sub[sub["loss_type"] == loss_type]
            vals.append(float(sub[column].mean()) if len(sub) else float("nan"))
        return vals

    metrics = [
        ("cum_profit", "Cumulative Profit (additive, fraction of capital)", "profit_comparison.png", None),
        ("cum_profit_geo", "Cumulative Profit (geometric, fraction of capital)", "profit_geo_comparison.png", None),
        ("sharpe_like", "Sharpe-like Ratio", "sharpe_comparison.png", None),
        ("rmse", "RMSE", "rmse_comparison.png", None),
        ("directional_acc", "Directional Accuracy", "directional_acc_comparison.png", (0.5, "Chance (50%)")),
    ]

    written: list[Path] = []
    for column, ylabel, filename, reference_line in metrics:
        series = [
            ("MSE", agg(column, "mse")),
            ("Profit-aware", agg(column, "profit-aware")),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped_bar(
            ax, markets, series,
            ylabel=ylabel,
            title=f"{ylabel}:\nMSE vs Profit-aware ({label})",
            reference_line=reference_line,
        )
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        written.append(out_path)
        print(f"Saved {out_path}  " + ", ".join(f"{lbl}={vals}" for lbl, vals in series))

    if "buy_hold" in df.columns:
        series = [
            ("MSE (geometric)", agg("cum_profit_geo", "mse")),
            ("Profit-aware (geometric)", agg("cum_profit_geo", "profit-aware")),
            ("Buy & Hold", agg("buy_hold")),
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped_bar(
            ax, markets, series,
            ylabel="Geometric Return (fraction of capital)",
            title=f"Geometric Return vs Buy & Hold ({label})",
        )
        fig.tight_layout()
        out_path = output_dir / "buy_hold_comparison.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        written.append(out_path)
        print(f"Saved {out_path}  " + ", ".join(f"{lbl}={vals}" for lbl, vals in series))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MSE vs profit-aware comparison charts for PH and US markets"
    )
    parser.add_argument("--lambda-dir", type=str, default=None,
                         help="Directory with separate benchmark_summary_ph_*.csv / benchmark_summary_us_*.csv")
    parser.add_argument("--csv", type=str, default=None,
                         help="Single combined benchmark_summary CSV (market inferred from symbol prefix)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        df = pd.read_csv(csv_path)
        df["market"] = df["symbol"].map(market_of)
        output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
        label = csv_path.stem
    else:
        lambda_dir = Path(args.lambda_dir or "results/llambda_0.7")
        df = pd.concat([load_market_csv(lambda_dir, m) for m in ("ph", "us")], ignore_index=True)
        output_dir = Path(args.output_dir) if args.output_dir else lambda_dir
        label = lambda_dir.name

    plot_comparisons(df, output_dir, label)


if __name__ == "__main__":
    main()
