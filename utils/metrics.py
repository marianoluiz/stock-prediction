from __future__ import annotations

import numpy as np


def directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of time-steps where predicted and actual share the same sign."""
    pred_sign = np.sign(pred)
    actual_sign = np.sign(actual)
    return float((pred_sign == actual_sign).mean())


def cumulative_profit(
    signal: np.ndarray,
    actual_return: np.ndarray,
    transaction_cost_rate: float = 0.001,
) -> float:
    """Sum of P&L after deducting transaction costs on each position change."""
    previous_signal = np.concatenate(([0.0], signal[:-1]))
    costs = transaction_cost_rate * np.abs(signal - previous_signal)
    net_profit = signal * actual_return - costs
    return float(net_profit.sum())


def sharpe_like(profit_series: np.ndarray, annualization: int = 252) -> float:
    """Annualised Sharpe-like ratio (assumes zero risk-free rate)."""
    std = profit_series.std(ddof=1)
    if std == 0:
        return 0.0
    return float((profit_series.mean() / std) * np.sqrt(annualization))


def mean_squared_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error between predicted and actual returns."""
    return float(np.mean((pred - actual) ** 2))


def mean_absolute_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error between predicted and actual returns."""
    return float(np.mean(np.abs(pred - actual)))


def root_mean_squared_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error between predicted and actual returns."""
    return float(np.sqrt(mean_squared_error(pred, actual)))
