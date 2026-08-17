from __future__ import annotations

import numpy as np


def directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    pred_sign = np.sign(pred)
    actual_sign = np.sign(actual)
    return float((pred_sign == actual_sign).mean())


def cumulative_profit(
    signal: np.ndarray,
    actual_return: np.ndarray,
    transaction_cost_rate: float = 0.001,
) -> float:
    previous_signal = np.concatenate(([0.0], signal[:-1]))
    costs = transaction_cost_rate * np.abs(signal - previous_signal)
    net_profit = signal * actual_return - costs
    return float(net_profit.sum())


def sharpe_like(profit_series: np.ndarray, annualization: int = 252) -> float:
    std = profit_series.std(ddof=1)
    if std == 0:
        return 0.0
    return float((profit_series.mean() / std) * np.sqrt(annualization))
