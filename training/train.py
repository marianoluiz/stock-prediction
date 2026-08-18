"""Training pipeline for stock-price prediction with profit-aware loss.

Provides helpers to run a single epoch (train or eval) and to train a model
for a fixed number of epochs, tracking loss, cumulative profit, directional
accuracy, and a Sharpe-like metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (
    cumulative_profit,
    cumulative_profit_geometric,
    directional_accuracy,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    sharpe_like,
)
from utils.trading import profit_aware_loss, trading_signal


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop."""

    loss_type: str = "profit-aware"
    alpha: float = 1.0
    transaction_cost_rate: float = 0.001
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays in a ``DataLoader`` for batched training or evaluation."""
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(x_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    alpha: float,
    transaction_cost_rate: float,
    device: torch.device,
    loss_type: str = "profit-aware",
    ) -> Dict[str, float]:
    """Run one training or evaluation epoch.

    Args:
        model:  Neural-network module.
        loader: Batched input/target pairs.
        optimizer: An optimizer when training, ``None`` during evaluation.
        alpha:  Scaling factor for the trading signal.
        transaction_cost_rate:  Fractional cost per trade.
        device: Target device for tensors.
        loss_type: ``"mse"`` for baseline or ``"profit-aware"`` for custom loss.

    Returns:
        Dictionary of aggregated metrics: ``loss``, ``directional_acc``,
        ``cum_profit``, ``sharpe_like``, ``mse``, ``mae`` and ``rmse``.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: List[float] = []
    pred_all: List[np.ndarray] = []
    actual_all: List[np.ndarray] = []
    signal_all: List[np.ndarray] = []
    profit_all: List[np.ndarray] = []
    previous_signal: torch.Tensor | None = None

    with torch.set_grad_enabled(is_train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            if loss_type == "mse":
                loss = F.mse_loss(pred, y_batch)
            else:
                loss = profit_aware_loss(
                    pred,
                    y_batch,
                    alpha,
                    transaction_cost_rate=transaction_cost_rate,
                    previous_signal=previous_signal,
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            signal = torch.sign(pred)
            if previous_signal is None:
                prev_signal = torch.zeros(1, device=device, dtype=signal.dtype)
            else:
                prev_signal = previous_signal.reshape(1)

            if signal.numel() == 1:
                shifted_prev = prev_signal
            else:
                shifted_prev = torch.cat([prev_signal, signal[:-1]])

            profit = signal * y_batch - transaction_cost_rate * torch.abs(signal - shifted_prev)

            losses.append(float(loss.detach().cpu().item()))
            pred_all.append(pred.detach().cpu().numpy())
            actual_all.append(y_batch.detach().cpu().numpy())
            signal_all.append(signal.detach().cpu().numpy())
            profit_all.append(profit.detach().cpu().numpy())

            previous_signal = signal[-1].detach()

    pred_np = np.concatenate(pred_all)
    actual_np = np.concatenate(actual_all)
    signal_np = np.concatenate(signal_all)
    profit_np = np.concatenate(profit_all)

    return {
        "loss": float(np.mean(losses)),
        "directional_acc": directional_accuracy(pred_np, actual_np),
        "cum_profit": cumulative_profit(signal_np, actual_np, transaction_cost_rate=transaction_cost_rate),
        "cum_profit_geo": cumulative_profit_geometric(signal_np, actual_np, transaction_cost_rate=transaction_cost_rate),
        "sharpe_like": sharpe_like(profit_np),
        "mse": mean_squared_error(pred_np, actual_np),
        "mae": mean_absolute_error(pred_np, actual_np),
        "rmse": root_mean_squared_error(pred_np, actual_np),
        "pred_np": pred_np,
        "actual_np": actual_np,
        "signal_np": signal_np,
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    capital: float = 1.0,
    ) -> Dict[str, list]:
    """Train *model* for a fixed number of epochs and return training history.

    Uses Adam optimiser and the profit-aware loss defined in
    ``utils.trading``.  Returns a dict whose keys are metric names
    (e.g. ``train_loss``, ``val_profit``) and whose values are per-epoch
    lists.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_profit": [],
        "val_profit": [],
        "train_profit_geo": [],
        "val_profit_geo": [],
        "train_dir_acc": [],
        "val_dir_acc": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            config.alpha,
            config.transaction_cost_rate,
            device,
            loss_type=config.loss_type,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            config.alpha,
            config.transaction_cost_rate,
            device,
            loss_type=config.loss_type,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_profit"].append(train_metrics["cum_profit"])
        history["val_profit"].append(val_metrics["cum_profit"])
        history["train_profit_geo"].append(train_metrics["cum_profit_geo"])
        history["val_profit_geo"].append(val_metrics["cum_profit_geo"])
        history["train_dir_acc"].append(train_metrics["directional_acc"])
        history["val_dir_acc"].append(val_metrics["directional_acc"])

        train_profit_php = train_metrics["cum_profit"] * capital
        val_profit_php = val_metrics["cum_profit"] * capital
        train_profit_geo_php = train_metrics["cum_profit_geo"] * capital
        val_profit_geo_php = val_metrics["cum_profit_geo"] * capital

        tag = f"[{config.loss_type.upper()}] " if config.loss_type != "profit-aware" else ""
        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"{tag}train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} | "
            f"train_profit={train_metrics['cum_profit']:.6f} ({train_profit_php:+,.0f} PHP) "
            f"val_profit={val_metrics['cum_profit']:.6f} ({val_profit_php:+,.0f} PHP) | "
            f"train_geo={train_metrics['cum_profit_geo']:.6f} ({train_profit_geo_php:+,.0f} PHP) "
            f"val_geo={val_metrics['cum_profit_geo']:.6f} ({val_profit_geo_php:+,.0f} PHP)"
        )

    return history
