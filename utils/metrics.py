from __future__ import annotations

from typing import Sequence

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


def cumulative_profit_geometric(
    signal: np.ndarray,
    actual_return: np.ndarray,
    transaction_cost_rate: float = 0.001,
) -> float:
    """Geometric (compounding) cumulative profit.

    Unlike the additive version, the balance compounds: losses shrink the
    next trade size and gains expand it, which reflects real-world trading.
    """
    balance = 1.0
    previous_signal = 0.0
    for s, r in zip(signal, actual_return):
        cost = transaction_cost_rate * abs(s - previous_signal)
        net_return = s * r - cost
        balance *= (1.0 + net_return)
        previous_signal = s
    return balance - 1.0


def mean_squared_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean squared error between predicted and actual returns."""
    return float(np.mean((pred - actual) ** 2))


def mean_absolute_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error between predicted and actual returns."""
    return float(np.mean(np.abs(pred - actual)))


def root_mean_squared_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error between predicted and actual returns."""
    return float(np.sqrt(mean_squared_error(pred, actual)))


def trade_log(
    signal: np.ndarray,
    actual_return: np.ndarray,
    pred: np.ndarray,
    dates: Sequence,
    capital: float,
    transaction_cost_rate: float = 0.001,
) -> None:
    """Print a per-trade log showing P&L and running balance for every trade."""
    n = len(signal)
    add_balance = capital
    geo_balance = capital
    prev_sig = 0.0

    wins = 0
    losses = 0
    best_trade = 0.0
    worst_trade = 0.0
    best_date = ""
    worst_date = ""

    header = (
        f"{'#':>4}  {'Date':>12}  {'Return':>8}  {'Pred':>8}  "
        f"{'Signal':>6}  {'Trade P&L':>12}  {'Add Bal':>14}  {'Geo Bal':>14}"
    )
    print(header)
    print("-" * len(header))

    for i in range(n):
        s = float(signal[i])
        r = float(actual_return[i])
        p = float(pred[i])
        cost = transaction_cost_rate * abs(s - prev_sig)

        net_return = s * r - cost
        trade_pnl = net_return * capital

        add_balance += trade_pnl
        geo_balance *= (1.0 + net_return)

        date_str = str(dates[i])[:10] if i < len(dates) else "?"

        sig_label = "+1" if s > 0 else "-1" if s < 0 else "FLAT"
        sign = "+" if net_return >= 0 else ""
        print(
            f"{i+1:4d}  {date_str:>12}  {r*100:>+7.2f}%  {p:>+8.4f}  "
            f"{sig_label:>6}  "
            f"{sign}{net_return*capital:>+11,.0f}  "
            f"{add_balance:>13,.0f}  {geo_balance:>13,.0f}"
        )

        if net_return >= 0:
            wins += 1
        else:
            losses += 1
        if net_return > best_trade:
            best_trade = net_return
            best_date = date_str
        if net_return < worst_trade:
            worst_trade = net_return
            worst_date = date_str

        prev_sig = s

    print("-" * len(header))
    total_add = add_balance - capital
    total_geo = geo_balance - capital
    print(f"  TOTALS: {wins} wins / {losses} losses")
    print(f"  Best trade:   {best_date}  {best_trade*capital:>+,.0f} PHP ({best_trade*100:+.2f}%)")
    print(f"  Worst trade:  {worst_date}  {worst_trade*capital:>+,.0f} PHP ({worst_trade*100:+.2f}%)")
    print(f"  Additive:     {total_add:>+,.0f} PHP (final {add_balance:>,.0f})")
    print(f"  Geometric:    {total_geo:>+,.0f} PHP (final {geo_balance:>,.0f})")
