"""Classic technical-analysis trading strategies, evaluated as baselines
against the GRU models using the exact same profit/cost/metric machinery
(see ``utils/metrics.py``).

Covers the three canonical strategy families so the comparison isn't just
"GRU vs. buy-and-hold": trend-following (SMA crossover), mean-reversion
(RSI), and momentum (MACD). Each signal function is shifted by one day so a
position is decided using only information available *before* the return it
earns realizes -- no lookahead, the same discipline the GRU's sequence
windows already respect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.metrics import (
    cumulative_profit,
    cumulative_profit_geometric,
    directional_accuracy,
    sharpe_like,
)


def sma_crossover_signal(close: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """Trend-following: +1 when the fast SMA is above the slow SMA, else -1."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    raw = np.sign(sma_fast - sma_slow)
    return raw.shift(1).rename("sma_crossover")


def rsi_signal(close: pd.Series, period: int = 14, lower: float = 30.0, upper: float = 70.0) -> pd.Series:
    """Mean-reversion: +1 when RSI < lower (oversold), -1 when RSI > upper
    (overbought), 0 (flat) otherwise.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    raw = pd.Series(0.0, index=close.index)
    raw[rsi < lower] = 1.0
    raw[rsi > upper] = -1.0
    return raw.shift(1).rename("rsi")


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal_span: int = 9) -> pd.Series:
    """Momentum: +1 when the MACD line is above its signal line, else -1."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    raw = np.sign(macd_line - signal_line)
    return raw.shift(1).rename("macd")


STRATEGIES = {
    "sma_crossover": sma_crossover_signal,
    "rsi": rsi_signal,
    "macd": macd_signal,
}


def strategy_signal_on_dates(close: pd.Series, fn, dates: np.ndarray) -> np.ndarray:
    """Compute one strategy's signal on ``close`` and align it to ``dates``.

    ``close`` should be the full price history (including pre-test warm-up)
    so the rolling/EWM windows have enough lookback -- mirrors how the GRU's
    sequence windows also draw on pre-test-boundary history.
    """
    raw_signal = fn(close)
    return raw_signal.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy()


def evaluate_price_strategies(
    close: pd.Series,
    dates_test: np.ndarray,
    actual_return_test: np.ndarray,
    transaction_cost_rate: float = 0.001,
) -> dict[str, dict[str, float | np.ndarray]]:
    """Score every strategy in :data:`STRATEGIES` on the same test window and
    metrics the GRU models are scored on.

    Returns ``{strategy_name: {"signal", "directional_acc", "cum_profit",
    "cum_profit_geo", "sharpe_like"}}``.
    """
    results: dict[str, dict[str, float | np.ndarray]] = {}

    for name, fn in STRATEGIES.items():
        signal = strategy_signal_on_dates(close, fn, dates_test)

        previous_signal = np.concatenate(([0.0], signal[:-1]))
        costs = transaction_cost_rate * np.abs(signal - previous_signal)
        profit = signal * actual_return_test - costs

        results[name] = {
            "signal": signal,
            "directional_acc": directional_accuracy(signal, actual_return_test),
            "cum_profit": cumulative_profit(signal, actual_return_test, transaction_cost_rate),
            "cum_profit_geo": cumulative_profit_geometric(signal, actual_return_test, transaction_cost_rate),
            "sharpe_like": sharpe_like(profit),
        }

    return results
