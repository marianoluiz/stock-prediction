"""Plotting helpers for training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
        f"split: train {format_date(t[0])}->{format_date(t[-1])} "
        f"/ val {format_date(v[0])}->{format_date(v[-1])} "
        f"/ test {format_date(e[0])}->{format_date(e[-1])}"
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
                       ``AAPL_profit_aware_2018-01-01_2026-08-17``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger = f"{symbol} ({start} -> {end})" if symbol else ""
    subtitle = f"\n{split_description(split)}" if split is not None else ""

    loss_name = f"{file_prefix}_loss_curve.png" if file_prefix else "loss_curve.png"
    profit_name = f"{file_prefix}_profit_curve.png" if file_prefix else "profit_curve.png"

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{loss_label} - {ledger}{subtitle}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / loss_name, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot([p * capital for p in history["train_profit"]], label="Train Cumulative Profit")
    plt.plot([p * capital for p in history["val_profit"]], label="Val Cumulative Profit")
    plt.plot([p * capital for p in history["train_profit_geo"]], label="Train Geometric Profit", linestyle="--")
    plt.plot([p * capital for p in history["val_profit_geo"]], label="Val Geometric Profit", linestyle="--")
    plt.title(f"Cumulative Profit - {ledger} | Capital: {capital:,.0f} PHP{subtitle}")
    plt.xlabel("Epoch")
    plt.ylabel("Profit (PHP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / profit_name, dpi=200)
    plt.close()