from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import cumulative_profit, directional_accuracy, sharpe_like
from utils.trading import profit_aware_loss, trading_signal


@dataclass
class TrainingConfig:
    alpha: float = 5.0
    transaction_cost_rate: float = 0.001
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
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
) -> Dict[str, float]:
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

            signal = trading_signal(pred, alpha)
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
        "sharpe_like": sharpe_like(profit_np),
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, list]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_profit": [],
        "val_profit": [],
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
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            config.alpha,
            config.transaction_cost_rate,
            device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_profit"].append(train_metrics["cum_profit"])
        history["val_profit"].append(val_metrics["cum_profit"])
        history["train_dir_acc"].append(train_metrics["directional_acc"])
        history["val_dir_acc"].append(val_metrics["directional_acc"])

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} | "
            f"train_profit={train_metrics['cum_profit']:.6f} val_profit={val_metrics['cum_profit']:.6f}"
        )

    return history
