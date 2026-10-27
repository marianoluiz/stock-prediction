"""Compare matched PSEi profit-aware runs across MSE calibration weights.

Run after benchmark.py has produced sop2_lambda_XX.csv and _daily.csv files.
This reports fold means descriptively; it does not tune lambda on test data.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import prod, sqrt
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "").zfill(2)


def value(row: dict, key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.3, 0.7, 1.0, 2.0])
    args = parser.parse_args()

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

    keys = ("rmse", "mean_abs_predicted_return", "mean_abs_position",
            "turnover_per_day", "additive_return", "geometric_return",
            "directional_accuracy")
    lines = [
        "# SOP 2: PSEi lambda comparison",
        "",
        "Same cached PSEi data, four test windows, seed 42, 50 epochs, "
        "and asymmetric buy/sell costs (0.296%/0.396%) for every run.",
        "Lambda 0 versus 0.7 is the planned comparison; other weights are "
        "exploratory sensitivity checks. Values below are arithmetic means "
        "of four fold-level results, not a pooled investment return.",
        "",
        "| λ | RMSE | Mean absolute predicted return | Mean absolute position | "
        "Turnover/day | Mean additive return | Mean geometric return | "
        "Directional accuracy | "
        "One-side folds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weight in args.lambdas:
        subset = [row for row in records if row["lambda"] == weight]
        means = [mean(value(row, key) for row in subset) for key in keys]
        one_side = sum(row["degenerate"] == "True" for row in subset)
        lines.append(
            f"| {weight:g} | {means[0]:.5f} | {means[1]:.5f} | "
            f"{means[2]:.3f} | {means[3]:.5f} | {means[4]:+.4%} | "
            f"{means[5]:+.4%} | {means[6]:.2%} | {one_side}/4 |"
        )
    if 0.0 in args.lambdas and 0.7 in args.lambdas:
        plain = {row["fold"]: row for row in records if row["lambda"] == 0.0}
        calibrated = {row["fold"]: row for row in records
                      if row["lambda"] == 0.7}
        difference = lambda key: mean(value(calibrated[f], key) - value(plain[f], key)
                                      for f in plain)
        rmse_wins = sum(value(calibrated[f], "rmse") < value(plain[f], "rmse")
                        for f in plain)
        profit_wins = sum(value(calibrated[f], "geometric_return") >
                          value(plain[f], "geometric_return") for f in plain)
        lines += ["", "## Planned contrast: λ = 0.7 minus λ = 0", "",
                  f"- Mean fold RMSE difference: {difference('rmse'):+.5f}; "
                  f"lower RMSE in {rmse_wins}/4 folds.",
                  f"- Mean absolute predicted-return difference: "
                  f"{difference('mean_abs_predicted_return'):+.5f}.",
                  f"- Mean absolute position difference: "
                  f"{difference('mean_abs_position'):+.3f}.",
                  f"- Mean fold geometric-return difference: "
                  f"{100*difference('geometric_return'):+.2f} percentage "
                  f"points; higher return in {profit_wins}/4 folds."]
    lines += ["", "## Per-fold results", "",
              "| λ | Fold | Dates | RMSE | Geometric return | "
              "Mean predicted return | Mean position | Side switches |",
              "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(
            f"| {row['lambda']:g} | {row['fold']} | {row['test_start']} "
            f"to {row['test_end']} | {value(row, 'rmse'):.5f} | "
            f"{value(row, 'geometric_return'):+.4%} | "
            f"{value(row, 'mean_predicted_return'):+.5f} | "
            f"{value(row, 'mean_position'):+.3f} | {row['side_switches']} |"
        )
    lines += ["", "The one-side flag means the executed position kept the "
              "same sign throughout a fold. It does not mean the position's "
              "magnitude was constant. This single-seed test sweep is "
              "descriptive and should not be used to select a lambda on the "
              "held-out test windows."]
    out_md = args.results / "sop2_lambda_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    print(out_md)
    print(out_png)


if __name__ == "__main__":
    main()
