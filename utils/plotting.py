"""Plotting helpers for training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history: dict, output_dir: Path, capital: float = 1.0, title: str = "Loss") -> None:
    """Plot training loss and cumulative profit curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot([p * capital for p in history["train_profit"]], label="Train Cumulative Profit")
    plt.plot([p * capital for p in history["val_profit"]], label="Val Cumulative Profit")
    plt.plot([p * capital for p in history["train_profit_geo"]], label="Train Geometric Profit", linestyle="--")
    plt.plot([p * capital for p in history["val_profit_geo"]], label="Val Geometric Profit", linestyle="--")
    plt.title(f"Cumulative Profit per Epoch (Capital: {capital:,.0f} PHP)")
    plt.xlabel("Epoch")
    plt.ylabel("Profit (PHP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "profit_curve.png", dpi=200)
    plt.close()