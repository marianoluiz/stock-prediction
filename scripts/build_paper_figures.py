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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_chapter1_loss_figures import build_all as build_chapter_1
from scripts.cost_comparison import plot as plot_cost_comparison
from scripts.significance import index_tests, plot as plot_significance
from scripts.sop2_compare import main as compare_lambdas

SAVED_BENCHMARK = ROOT / "results/benchmark_summary_psei_20261025_203842.csv"


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
    build_chapter_1(results / "chapter1", ROOT / "data/PSE-PSEI_2018-01-01_latest.csv")
    print("Rebuilt benchmark, significance, SOP 2 and Chapter 1 outputs.")
    print("Cost table: results/cost_comparison.csv; source diagrams: docs/figure_3_1.png, figure_3_2.png")


if __name__ == "__main__":
    main()
