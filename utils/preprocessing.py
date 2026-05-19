from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _normalize_price_frame(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize yfinance frames and cached CSVs into a flat numeric OHLCV table."""

    data = df.copy()

    # yfinance often returns a MultiIndex with the ticker in the last level.
    if isinstance(data.columns, pd.MultiIndex):
        if symbol is not None and symbol in data.columns.get_level_values(-1):
            data = data.xs(symbol, axis=1, level=-1)
        else:
            data.columns = [str(col[0]) for col in data.columns]

    # Cached CSVs may contain the date as a regular column.
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.set_index("Date")

    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[~data.index.isna()]

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(how="any")


def load_stock_data(symbol: str, start: str, end: str | None = None, cache_path: str | None = None) -> pd.DataFrame:
    if cache_path and Path(cache_path).exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        return _normalize_price_frame(cached, symbol=symbol)

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for symbol={symbol}")

    df = _normalize_price_frame(df, symbol=symbol)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache_path, index=False)

    return df


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    returns = df[price_col].pct_change().dropna()
    returns.name = "return"
    return returns


def create_sequences(returns: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(returns) - sequence_length):
        x.append(returns[i : i + sequence_length])
        y.append(returns[i + sequence_length])

    x_arr = np.asarray(x, dtype=np.float32)[..., np.newaxis]  # [N, seq_len, 1]
    y_arr = np.asarray(y, dtype=np.float32)  # [N]
    return x_arr, y_arr


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15, 
) -> SplitData:
    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return SplitData(
        x_train=x[:train_end],
        y_train=y[:train_end],
        x_val=x[train_end:val_end],
        y_val=y[train_end:val_end],
        x_test=x[val_end:],
        y_test=y[val_end:],
    )
