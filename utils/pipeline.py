"""Shared data-preparation pipeline used by main.py, train.py and evaluate.py.

Guarantees every entry point produces the *same* sequences and the *same*
chronological train/val/test split for a given set of data arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from training.train import to_loader
from utils.preprocessing import SplitData, compute_returns, create_sequences, load_stock_data, train_val_test_split


@dataclass
class PreparedData:
    """Prepared features plus the three batched loaders."""

    split: SplitData
    returns: np.ndarray
    dates: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def cache_path_for(symbol: str, start: str, end: str | None) -> str:
    """Return the CSV cache path for the given data arguments."""
    return str(Path("data") / f"{symbol}_{start}_{end or 'latest'}.csv")


def prepare_data(
    symbol: str,
    start: str,
    end: str | None,
    sequence_length: int,
    batch_size: int,
    shuffle: bool = False,
) -> PreparedData:
    """Load, window and chronologically split stock data into loaders.

    Args:
        symbol:          Ticker symbol (e.g. ``"AAPL"``).
        start:           Start date string (``"YYYY-MM-DD"``).
        end:             End date string or ``None`` (latest available).
        sequence_length: Lookback window (days) per sample.
        batch_size:      Mini-batch size for the DataLoaders.
        shuffle:         Whether to shuffle the *training* loader
                         (default ``False`` to keep evaluation deterministic).

    Returns:
        A :class:`PreparedData` containing the split arrays and loaders.
    """
    df = load_stock_data(symbol, start, end, cache_path_for(symbol, start, end))
    returns = compute_returns(df)
    dates = returns.index.to_numpy()
    x, y, seq_dates = create_sequences(returns.values, sequence_length=sequence_length, dates=dates)
    split = train_val_test_split(x, y, dates=seq_dates)

    return PreparedData(
        split=split,
        returns=returns.values.astype(np.float32),
        dates=dates,
        train_loader=to_loader(split.x_train, split.y_train, batch_size, shuffle=shuffle),
        val_loader=to_loader(split.x_val, split.y_val, batch_size, shuffle=False),
        test_loader=to_loader(split.x_test, split.y_test, batch_size, shuffle=False),
    )