"""Reproducible, data-based Chapter 1 illustrations for SOP 1 and SOP 2.

Run in Google Colab or locally. The curves use one *observed* PSEi daily return,
the repo's tanh position transformation, and its asymmetric cost formula. Grid
points are hypothetical candidate forecasts, not trained-model predictions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SHA256 = "35bb13112fc537d62697042bb7dd27803816f51254cadad2822ed0e5913d4972"
BUY_COST = 0.00296
SELL_COST = 0.00396
MAIN_LAMBDA = 0.7
FIRST_TEST_DATE = "2024-03-25"


def build(output_dir: Path, csv_path: Path):
    ROOT = output_dir
    ROOT.mkdir(parents=True, exist_ok=True)
    CSV = csv_path
    if hashlib.sha256(CSV.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Illustrative PSEi source CSV checksum mismatch")
    data = pd.read_csv(CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if data["Date"].duplicated().any() or (data["Close"] <= 0).any():
        raise ValueError("PSEi CSV has duplicate dates or nonpositive closes")
    data["daily_return"] = data["Close"].pct_change()
    pretest = data.loc[data["Date"] < pd.Timestamp(FIRST_TEST_DATE)].copy()
    warmup = 252
    # Prespecified illustration selection rule: the first pre-test PSEi up day
    # after 252 prior returns whose move falls between the two cost coefficients.
    # This makes the trade-off visible; it is not a representative-day claim.
    eligible = pretest.iloc[warmup + 1:]
    example = eligible.loc[
        (eligible["daily_return"] > BUY_COST)
        & (eligible["daily_return"] < SELL_COST)
    ].iloc[0]
    prior_returns = data.loc[example.name - warmup:example.name - 1, "daily_return"].to_numpy()
    if not np.isfinite(prior_returns).all() or len(prior_returns) != warmup:
        raise ValueError("Missing returns in the 252-day calibration history")
    alpha = 1.0 / np.std(prior_returns, ddof=0)
    actual_return = float(example["daily_return"])
    example_date = example["Date"].strftime("%Y-%m-%d")
    assert example_date == "2019-01-31"

    # One-step calculation from a flat previous position. This is a loss-surface
    # slice, not a trained model or a backtest.
    candidate_forecasts = np.linspace(-0.01, 0.03, 1601)
    positions = np.tanh(alpha * candidate_forecasts)
    delta = positions  # previous position = 0
    costs = BUY_COST * np.maximum(delta, 0) + SELL_COST * np.maximum(-delta, 0)
    net_return = positions * actual_return - costs
    mse = (candidate_forecasts - actual_return) ** 2
    lambda_values = [0.0, 0.3, 0.7, 1.0, 2.0]
    losses = {lam: -net_return + lam * mse for lam in lambda_values}

    table = pd.DataFrame({
        "candidate_forecast_decimal": candidate_forecasts,
        "position_tanh": positions,
        "simulated_cost_decimal": costs,
        "simulated_net_return_decimal": net_return,
        "mse_loss": mse,
    })
    for lam in lambda_values:
        table[f"profit_aware_loss_lambda_{lam:g}"] = losses[lam]
    table.to_csv(ROOT / "SOP_loss_curve_values.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    x_pct = candidate_forecasts * 100
    scale = 1e4  # display scaling only; both curves remain in their original loss units

    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=False)
    ax.plot(x_pct, mse * scale, color="#174A75", linewidth=2.6, label="MSE-only loss")
    ax.plot(x_pct, losses[MAIN_LAMBDA] * scale, color="#B5551D", linewidth=2.6,
            label="Profit-aware loss (λ = 0.7)")
    for arr, color in [(mse, "#174A75"), (losses[MAIN_LAMBDA], "#B5551D")]:
        i = int(np.argmin(arr))
        ax.scatter([x_pct[i]], [arr[i] * scale], s=78, color=color, edgecolor="white", zorder=4)
    ax.axvline(actual_return * 100, color="#59636E", linewidth=1.2, linestyle="--",
               label="Observed return")
    ax.axhline(0, color="#A5ADB5", linewidth=0.8)
    ax.set_xlim(x_pct[0], x_pct[-1])
    ax.set_xlabel("Hypothetical next-day return forecast (%)")
    ax.set_ylabel("Numerical loss × 10⁴")
    ax.set_title("Figure 1.2  Different loss objectives on one observed PSEi return", pad=22,
                 weight="bold", color="#17324F")
    ax.text(0.5, 1.01, f"{example_date}: observed close-to-close return = {actual_return*100:+.3f}%",
            transform=ax.transAxes, ha="center", color="#46525E")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    fig.text(0.5, 0.025,
             "Alpha uses 252 prior returns. Forecast grid points are not trained GRU predictions or test results.",
             ha="center", fontsize=9, color="#53606B")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.81, bottom=0.19)
    fig.savefig(ROOT / "Figure_1_2_SOP1_data_based_loss_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=False)
    colors = {0.0: "#174A75", 0.3: "#5A849C", 0.7: "#B5551D", 1.0: "#8E6998", 2.0: "#3B7A56"}
    for lam in lambda_values:
        ax.plot(x_pct, losses[lam] * scale, linewidth=2.7 if lam in (0.0, 0.7) else 1.8,
                color=colors[lam], label=f"λ = {lam:g}")
    for lam in (0.0, 0.7):
        i = int(np.argmin(losses[lam]))
        ax.scatter([x_pct[i]], [losses[lam][i] * scale], s=75,
                   color=colors[lam], edgecolor="white", zorder=4)
    ax.axvline(actual_return * 100, color="#59636E", linewidth=1.2, linestyle="--",
               label="Observed return")
    ax.axhline(0, color="#A5ADB5", linewidth=0.8)
    ax.set_xlim(x_pct[0], x_pct[-1])
    ax.set_xlabel("Hypothetical next-day return forecast (%)")
    ax.set_ylabel("Profit-aware loss × 10⁴")
    ax.set_title("Figure 1.3  Effect of λ on the one-step profit-aware loss", pad=22,
                 weight="bold", color="#17324F")
    ax.text(0.5, 1.01, f"Same {example_date} PSEi return, tanh scale, and asymmetric cost rule",
            transform=ax.transAxes, ha="center", color="#46525E")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.text(0.5, 0.025,
             "Alpha uses 252 prior returns. These loss curves do not show trained-model outcomes.",
             ha="center", fontsize=9, color="#53606B")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.81, bottom=0.19)
    fig.savefig(ROOT / "Figure_1_3_SOP2_data_based_lambda_curves.png", dpi=300)
    plt.close(fig)

    print("Example date:", example_date, "observed return:", actual_return)
    print("Illustrative alpha:", alpha)
    return {"example_date": example_date, "observed_return": actual_return,
            "alpha": float(alpha), "forecast_grid_points": len(candidate_forecasts),
            "selection": "First pre-test positive return between the two cost coefficients after 252 prior returns; illustrative, not representative"}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    build(repo / "figures/generated", repo / "data/PSE-PSEI_2018-01-01_latest.csv")
