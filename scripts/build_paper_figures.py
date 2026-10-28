"""Rebuild all paper figures and result tables from saved PSEi CSVs.

No model is trained. Chapter 1 contains labelled illustrations, Chapter 3
contains the two retained diagrams, and Chapter 4 uses the existing plotting
functions. No Word export or duplicate figure gallery is needed.
See README.md for the figure-to-output mapping.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chapter1_loss_figures import build as build_loss_figures
from scripts.cost_comparison import plot as plot_cost_comparison
from scripts.significance import index_tests, plot as plot_significance
from scripts.sop2_compare import main as compare_lambdas

SAVED_BENCHMARK = ROOT / "results/benchmark_summary_psei_20261025_203842.csv"


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_chapter_1(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.5, r"$\mathrm{MSE}=\frac{1}{N}\sum_{t=1}^{N}(r_t-\hat r_t)^2$",
            ha="center", va="center", fontsize=24, transform=ax.transAxes)
    save(fig, output / "figure_1_1.png")

    # Reproduce the group's original Colab illustration. This synthetic
    # fixed 0.5-unit-per-row cost is not the model's asymmetric cost rule.
    rng = np.random.RandomState(42)
    price = 100 + np.cumsum(rng.normal(0, 1, 200))
    predicted = price + rng.normal(0, 0.5, 200)
    frame = pd.DataFrame({"actual": price, "predicted": predicted})
    position = np.where(frame.predicted.diff() > 0, 1, -1)
    profit = position * frame.actual.diff()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No Transaction Cost", "With Transaction Cost"],
           [float(profit.sum()), float((profit - 0.5).sum())])
    ax.set_title("Trading Friction Impact")
    ax.set_ylabel("Total Profit")
    save(fig, output / "figure_1_4.png")
    build_loss_figures(output, ROOT / "data/PSE-PSEI_2018-01-01_latest.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=SAVED_BENCHMARK,
                        help="PSEi four-fold benchmark CSV (default: committed reference run)")
    source.add_argument("--latest", action="store_true",
                        help="Use the newest PSEi benchmark after a fresh training run")
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="Directory containing cost_comparison.csv and the five lambda runs")
    args = parser.parse_args(argv)
    results = args.results.resolve()
    if args.latest:
        candidates = [path for path in results.glob("benchmark_summary_psei_*.csv")
                      if not path.stem.endswith("_daily")]
        if not candidates:
            parser.error("No PSEi benchmark CSV found")
        summary_path = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        summary_path = args.csv.resolve()
    summary = pd.read_csv(summary_path)
    daily_path = summary_path.with_name(summary_path.stem + "_daily.csv")
    daily = pd.read_csv(daily_path)
    cost = pd.read_csv(results / "cost_comparison.csv")
    for label, table in (("benchmark", summary), ("cost ablation", cost)):
        if set(table.symbol) != {"PSE:PSEI"} or set(table.fold) != {0, 1, 2, 3}:
            raise ValueError(f"Expected four PSEi-only folds in {label}")
    for name in ("figure_3_1.png", "figure_3_2.png"):
        if not (ROOT / "docs" / name).is_file():
            raise FileNotFoundError(f"Missing source diagram: docs/{name}")

    # Reuse main's summary and plotting code, without duplicate chart copies.
    subprocess.run([sys.executable, "benchmark.py", "--from-csv", str(summary_path), "--charts"],
                   cwd=ROOT, check=True)
    compare_lambdas(["--results", str(results)])
    tests, unit = index_tests(summary_path)
    tests.to_csv(results / "significance.csv", index=False)
    plot_significance(tests, unit, summary_path.name, results / "significance.png")
    plot_cost_comparison(cost, 0.00296, 0.00396, results / "cost_comparison.png")
    with plt.rc_context(plt.rcParamsDefault):
        build_chapter_1(results / "chapter1")
    print("Rebuilt benchmark, significance, SOP 2 and Chapter 1 outputs.")
    print("Cost table: results/cost_comparison.csv; source diagrams: docs/figure_3_1.png, figure_3_2.png")


if __name__ == "__main__":
    main()
