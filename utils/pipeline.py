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
from utils.preprocessing import (
    SplitData,
    build_feature_frame,
    compute_returns,
    create_sequences,
    expanding_walk_forward_folds,
    load_stock_data,
    train_val_test_split,
)


@dataclass
class PreparedData:
    """Prepared features plus the three batched loaders."""

    split: SplitData
    returns: np.ndarray
    dates: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class WalkForwardFold:
    """One expanding-window walk-forward fold: split plus its batched loaders."""

    fold: int
    split: SplitData
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def cache_path_for(symbol: str, start: str, end: str | None) -> str:
    """Return the CSV cache path for the given data arguments."""
    safe_symbol = symbol.replace(":", "-")
    return str(Path("data") / f"{safe_symbol}_{start}_{end or 'latest'}.csv")


def build_sequences(
    symbol: str,
    start: str,
    end: str | None,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, feature-engineer and window a symbol's full history (unsplit).

    Shared by :func:`prepare_data` (single chronological split) and
    :func:`prepare_walk_forward` (expanding-window folds) so the feature
    engineering -- all backward-looking rolling stats, safe to compute once
    over the full range -- never gets recomputed per fold.

    Returns:
        ``(x, y, seq_dates, dates, returns)`` -- the windowed sequence
        arrays, their aligned dates, the full unwindowed per-day dates, and
        the full unwindowed return series.
    """
    df = load_stock_data(symbol, start, end, cache_path_for(symbol, start, end))
    returns = compute_returns(df)
    features = build_feature_frame(df)
    dates = features.index.to_numpy()
    x, y, seq_dates = create_sequences(features.values, sequence_length=sequence_length, dates=dates)
    return x, y, seq_dates, dates, returns.values.astype(np.float32)


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
    x, y, seq_dates, dates, returns = build_sequences(symbol, start, end, sequence_length)
    split = train_val_test_split(x, y, dates=seq_dates)

    return PreparedData(
        split=split,
        returns=returns,
        dates=dates,
        train_loader=to_loader(split.x_train, split.y_train, batch_size, shuffle=shuffle),
        val_loader=to_loader(split.x_val, split.y_val, batch_size, shuffle=False),
        test_loader=to_loader(split.x_test, split.y_test, batch_size, shuffle=False),
    )


def prepare_walk_forward(
    symbol: str,
    start: str,
    end: str | None,
    sequence_length: int,
    batch_size: int,
    n_folds: int = 4,
    eval_fraction: float = 0.30,
    val_ratio: float = 0.15,
    shuffle: bool = False,
) -> list[WalkForwardFold]:
    """Build expanding-window walk-forward folds with loaders for a symbol.

    See :func:`utils.preprocessing.expanding_walk_forward_folds` for the
    fold boundary logic.
    """
    x, y, seq_dates, _dates, _returns = build_sequences(symbol, start, end, sequence_length)
    splits = expanding_walk_forward_folds(
        x, y, n_folds=n_folds, eval_fraction=eval_fraction, val_ratio=val_ratio, dates=seq_dates,
    )

    return [
        WalkForwardFold(
            fold=k,
            split=split,
            train_loader=to_loader(split.x_train, split.y_train, batch_size, shuffle=shuffle),
            val_loader=to_loader(split.x_val, split.y_val, batch_size, shuffle=False),
            test_loader=to_loader(split.x_test, split.y_test, batch_size, shuffle=False),
        )
        for k, split in enumerate(splits)
    ]
