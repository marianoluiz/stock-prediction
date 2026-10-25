"""Plotting helpers for training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from utils.chart_style import COLOR_MSE, COLOR_PA, apply_style, legend_above, save, title_block
from utils.preprocessing import SplitData


def format_date(d) -> str:
    """Render a numpy datetime64 / Timestamp / str as ``YYYY-MM-DD``."""
    if d is None:
        return ""
    if hasattr(d, "shape") and getattr(d, "size", 1) == 0:
        return ""
    return str(pd.Timestamp(d).date())


def split_description(split: SplitData) -> str:
    """One-line summary of the train/val/test date ranges."""
    t, v, e = split.dates_train, split.dates_val, split.dates_test
    return (
        f"Train {format_date(t[0])} to {format_date(t[-1])}  "
        f"|  validation {format_date(v[0])} to {format_date(v[-1])}  "
        f"|  test {format_date(e[0])} to {format_date(e[-1])}"
    )


def plot_history(
    history: dict,
    output_dir: Path,
    capital: float = 1.0,
    symbol: str = "",
    start: str = "",
    end: str = "",
    loss_label: str = "Loss",
    split: SplitData | None = None,
    file_prefix: str = "",
) -> None:
    """Plot training loss and cumulative profit curves.

    Args:
        history:       History dict returned by :func:`training.train.fit`.
        output_dir:    Directory to save the PNG files into.
        capital:       Notional capital in PHP used to scale the profit curves.
        symbol:        Ticker symbol, shown in the titles.
        start:         Data start date, shown in the titles.
        end:           Actual last data date, shown in the titles.
        loss_label:    Display name of the loss (e.g. ``"PROFIT AWARE"``).
        split:         Optional split, used to annotate date ranges on the plot.
        file_prefix:   Optional prefix for the PNG file names (e.g.
                       ``PSE-ICT_profit_aware_2018-01-01_2026-08-17``).
    """
    apply_style()
    ledger = f"{symbol} ({start} to {end})" if symbol else ""
    subtitle = split_description(split) if split is not None else None
    title_suffix = f": {ledger}" if ledger else ""
    loss_name_display = "MSE" if loss_label.upper() == "MSE" else loss_label.capitalize()

    loss_name = f"{file_prefix}_loss_curve.png" if file_prefix else "loss_curve.png"
    profit_name = f"{file_prefix}_profit_curve.png" if file_prefix else "profit_curve.png"
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, history["train_loss"], color=COLOR_MSE, label="Training")
    ax.plot(epochs, history["val_loss"], color=COLOR_PA, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    legend_above(ax)
    title_block(ax, f"{loss_name_display} loss{title_suffix}", subtitle)
    fig.tight_layout()
    save(fig, output_dir / loss_name)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, [p * capital for p in history["train_profit"]], color=COLOR_MSE, label="Training, additive")
    ax.plot(epochs, [p * capital for p in history["val_profit"]], color=COLOR_PA, label="Validation, additive")
    ax.plot(epochs, [p * capital for p in history["train_profit_geo"]], color=COLOR_MSE, linestyle=(0, (4, 2)),
            label="Training, geometric")
    ax.plot(epochs, [p * capital for p in history["val_profit_geo"]], color=COLOR_PA, linestyle=(0, (4, 2)),
            label="Validation, geometric")
    ax.axhline(0, color="#898781", linewidth=0.9, zorder=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Profit (PHP, on {capital:,.0f} capital)")
    legend_above(ax)
    title_block(ax, f"Cumulative profit, {loss_name_display} loss{title_suffix}", subtitle)
    fig.tight_layout()
    save(fig, output_dir / profit_name)
