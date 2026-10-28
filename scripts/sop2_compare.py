"""Compare matched PSEi profit-aware runs across MSE calibration weights.

Run after benchmark.py has produced sop2_lambda_XX.csv and _daily.csv files.
This reports fold means descriptively; it does not tune lambda on test data.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from math import prod, sqrt
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "").zfill(2)


def value(row: dict, key: str) -> float:
    return float(row[key])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.3, 0.7, 1.0, 2.0])
    parser.add_argument("--train", action="store_true",
                        help="Rerun the matched four-fold CPU experiments before comparing")
    args = parser.parse_args(argv)
    if not args.lambdas or any(weight < 0 for weight in args.lambdas):
        parser.error("Calibration weights must be nonnegative")
    if len(set(args.lambdas)) != len(args.lambdas):
        parser.error("Calibration weights must be unique")
    if len({tag(weight) for weight in args.lambdas}) != len(args.lambdas):
        parser.error("Calibration weights must have distinct output file tags")
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.train:
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        for weight in args.lambdas:
            command = [sys.executable, "benchmark.py", "--symbols", "PSE:PSEI",
                       "--start", "2018-01-01", "--walk-forward", "--folds", "4",
                       "--eval-fraction", "0.30", "--epochs", "50", "--seed", "42",
                       "--device", "cpu", "--buy-cost", "0.00296", "--sell-cost", "0.00396",
                       "--loss-lambda", str(weight), "--output",
                       str(args.results / f"sop2_lambda_{tag(weight)}.csv")]
            subprocess.run(command, cwd=ROOT, env=environment, check=True)

    records: list[dict] = []
    reference_windows: dict[int, tuple[str, str, int]] | None = None
    reference_dates: dict[int, list[str]] | None = None
    mse_reference: dict[int, dict] | None = None

    for weight in args.lambdas:
        stem = args.results / f"sop2_lambda_{tag(weight)}"
        summary = read_csv(stem.with_suffix(".csv"))
        daily = read_csv(Path(f"{stem}_daily.csv"))
        pa = {int(row["fold"]): row for row in summary
              if row["symbol"] == "PSE:PSEI" and row["loss_type"] == "profit-aware"}
        mse = {int(row["fold"]): row for row in summary
               if row["symbol"] == "PSE:PSEI" and row["loss_type"] == "mse"}
        if set(pa) != {0, 1, 2, 3} or set(mse) != {0, 1, 2, 3}:
            raise ValueError(f"Expected four PSEi folds for both models: {stem}")

        windows = {fold: (row["test_start"], row["test_end"], int(row["n_test"]))
                   for fold, row in pa.items()}
        by_fold: dict[int, list[dict]] = defaultdict(list)
        for row in daily:
            if row["symbol"] == "PSE:PSEI" and row["model"] == "profit-aware":
                by_fold[int(row["fold"])].append(row)
        dates = {fold: [row["date"] for row in rows]
                 for fold, rows in by_fold.items()}
        if reference_windows is None:
            reference_windows, reference_dates, mse_reference = windows, dates, mse
        if windows != reference_windows or dates != reference_dates:
            raise ValueError(f"Test windows or dates differ for lambda={weight}")
        if set(by_fold) != set(pa):
            raise ValueError(f"Missing daily rows for lambda={weight}")
        # Identical MSE rows confirm the non-PA comparator remained fixed.
        for fold in mse:
            for metric in ("rmse", "cum_profit_geo", "directional_acc"):
                if mse[fold][metric] != mse_reference[fold][metric]:
                    raise ValueError(f"MSE baseline changed at lambda={weight}, fold={fold}")

        for fold in sorted(pa):
            row = pa[fold]
            days = by_fold[fold]
            positions = [value(day, "position") for day in days]
            predictions = [value(day, "predicted_return") for day in days]
            if len(days) != windows[fold][2]:
                raise ValueError(f"Daily row count mismatch: lambda={weight}, fold={fold}")
            actuals = [value(day, "actual_return") for day in days]
            net_returns = [value(day, "net_return") for day in days]
            recalculated = {
                "rmse": sqrt(mean((prediction - actual) ** 2
                                  for prediction, actual in zip(predictions, actuals))),
                "cum_profit": sum(net_returns),
                "cum_profit_geo": prod(1 + daily_net for daily_net in net_returns) - 1,
            }
            for metric, computed in recalculated.items():
                if abs(computed - value(row, metric)) > 2e-5:
                    raise ValueError(f"Daily {metric} mismatch: lambda={weight}, "
                                     f"fold={fold}, {computed} vs {row[metric]}")
            switches = sum((a > 0) != (b > 0)
                           for a, b in zip(positions, positions[1:]))
            records.append({
                "lambda": weight,
                "fold": fold,
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "n_test": row["n_test"],
                "rmse": row["rmse"],
                "additive_return": row["cum_profit"],
                "geometric_return": row["cum_profit_geo"],
                "sharpe_like": row["sharpe_like"],
                "directional_accuracy": row["directional_acc"],
                "mean_abs_predicted_return": mean(abs(x) for x in predictions),
                "mean_predicted_return": mean(predictions),
                "sd_predicted_return": pstdev(predictions),
                "mean_position": row["mean_position"],
                "mean_abs_position": row["mean_abs_position"],
                "turnover_per_day": row["turnover"],
                "short_share": row["short_share"],
                "side_switches": switches,
                "degenerate": row["degenerate"],
            })

    out_csv = args.results / "sop2_lambda_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metrics = ("rmse", "mean_abs_predicted_return", "mean_abs_position",
               "turnover_per_day", "additive_return", "geometric_return",
               "sharpe_like", "directional_accuracy")
    aggregate = []
    for weight in args.lambdas:
        subset = [row for row in records if row["lambda"] == weight]
        aggregate.append({
            "lambda": weight,
            **{key: mean(value(row, key) for row in subset) for key in metrics},
            "one_side_folds": sum(row["degenerate"] == "True" for row in subset),
            "n_folds": len(subset),
        })
    table = pd.DataFrame(aggregate)
    summary_csv = args.results / "sop2_lambda_summary.csv"
    table.to_csv(summary_csv, index=False)
    out_summary = args.results / "sop2_lambda_comparison_summary.txt"
    description = (
        "PSEi MSE-calibration comparison: four test folds, seed 42, 50 epochs.\n"
        "Costs: buy 0.296%, sell 0.396%. Matched CPU runs.\n"
        "Values are arithmetic means of fold metrics, not pooled portfolio returns.\n"
        "Lambda 0 vs 0.7 is the primary contrast; other weights are exploratory.\n"
        "One-side folds count positions keeping the same sign; magnitude may vary.\n"
        "The test-window sweep does not select a generally optimal lambda.\n\n"
    )
    out_summary.write_text(description + table.to_string(index=False) + "\n",
                           encoding="utf-8")

    panels = (
        ("rmse", "Test RMSE", False),
        ("mean_abs_predicted_return", "Mean absolute predicted return", True),
        ("mean_abs_position", "Mean absolute position", False),
        ("geometric_return", "Mean fold geometric return", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (key, title, as_percent) in zip(axes.flat, panels):
        x_values = list(args.lambdas)
        y_values = [mean(value(row, key) for row in records
                         if row["lambda"] == weight) for weight in x_values]
        if as_percent:
            y_values = [100 * y for y in y_values]
        for weight in x_values:
            fold_values = [value(row, key) for row in records
                           if row["lambda"] == weight]
            if as_percent:
                fold_values = [100 * y for y in fold_values]
            ax.scatter([weight] * len(fold_values), fold_values,
                       color="#8db1d9", alpha=0.6, s=18, zorder=2)
        ax.plot(x_values, y_values, marker="o", color="#245b99",
                linewidth=1.8, zorder=3)
        ax.set_xticks(x_values)
        ax.set_xlabel("MSE calibration weight (λ)")
        ax.set_ylabel("Percent" if as_percent else "Value")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("PSEi sensitivity to the MSE calibration weight\n"
                 "Four held-out folds, seed 42; dots = folds, line = mean",
                 fontsize=13)
    out_png = args.results / "sop2_lambda_sensitivity.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(out_csv)
    print(summary_csv)
    print(out_summary)
    print(out_png)


if __name__ == "__main__":
    main()
