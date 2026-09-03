from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class SplitData:
    """Container for train/validation/test numpy array splits."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dates_train: np.ndarray = field(default_factory=lambda: np.array([]))
    dates_val: np.ndarray = field(default_factory=lambda: np.array([]))
    dates_test: np.ndarray = field(default_factory=lambda: np.array([]))


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


def _normalize_tv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a tvDatafeed OHLCV frame into the same shape as yfinance's."""

    data = df.copy()
    data = data.drop(columns=["symbol"], errors="ignore")
    data = data.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    data.index = pd.to_datetime(data.index, errors="coerce").tz_localize(None).normalize()
    data.index.name = "Date"
    data = data.loc[~data.index.isna()]

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(how="any")


def _load_stock_data_tv(symbol: str, exchange: str, start: str, end: str | None) -> pd.DataFrame:
    """Load OHLCV data from TradingView via tvDatafeed (unofficial, reverse-engineered
    client hitting TradingView's internal endpoints -- not a documented API. Used for
    exchanges yfinance doesn't cover, e.g. PSE. TradingView serves at most 5000 bars
    per request rather than an arbitrary date range, so we pull the max and trim.
    """
    from tvDatafeed import Interval, TvDatafeed

    tv = TvDatafeed()
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=5000)

    if df is None or df.empty:
        raise ValueError(f"No data downloaded for symbol={exchange}:{symbol}")

    df = _normalize_tv_frame(df)

    if start:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end:
        df = df.loc[df.index <= pd.to_datetime(end)]

    return df


def load_stock_data(symbol: str, start: str, end: str | None = None, cache_path: str | None = None) -> pd.DataFrame:
    """Load stock OHLCV data with local CSV caching.

    Downloads via yfinance if no cache exists, otherwise reads from the CSV
    file at ``cache_path``.  Symbols written as ``"EXCHANGE:TICKER"`` (e.g.
    ``"PSE:JFC"``) are instead routed to TradingView via ``tvDatafeed``, for
    exchanges (like the PSE) yfinance doesn't carry.  The result is always
    normalized to a flat numeric DataFrame indexed by ``Date``.

    Args:
        symbol:   Ticker symbol (e.g. ``"AAPL"``), or ``"EXCHANGE:TICKER"``
                  (e.g. ``"PSE:JFC"``) to fetch from TradingView instead.
        start:    Start date string (``"YYYY-MM-DD"``).
        end:      End date string, or ``None`` for the latest available data.
        cache_path: File path for the CSV cache.  If ``None`` no caching is
                    performed.

    Returns:
        A clean OHLCV DataFrame indexed by ``pd.DatetimeIndex``.

    Raises:
        ValueError: If the data source returns no rows for the symbol.
    """
    if cache_path and Path(cache_path).exists():
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        return _normalize_price_frame(cached, symbol=symbol)

    if ":" in symbol:
        exchange, ticker = symbol.split(":", 1)
        df = _load_stock_data_tv(ticker, exchange, start, end)
    else:
        df = yf.download(symbol, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError(f"No data downloaded for symbol={symbol}")

        df = _normalize_price_frame(df, symbol=symbol)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache_path, index=False)

    return df


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Compute simple percentage returns from a price series.

    Calculates r_t = (P_{t+1} - P_t) / P_t using ``pct_change``.

    Args:
        df:        DataFrame containing at least one price column.
        price_col: Name of the column to use (default ``"Close"``).

    Returns:
        A ``pd.Series`` of percentage returns with the first row (NaN) dropped.
    """
    returns = df[price_col].pct_change().dropna()
    returns.name = "return"
    return returns


DEFAULT_LAGS: Tuple[int, ...] = (1, 5, 10)


def build_lagged_features(returns: pd.Series, lags: Tuple[int, ...] = DEFAULT_LAGS) -> pd.DataFrame:
    """Build a feature frame: raw return plus explicit lagged-return channels.

    Column 0 (``"return"``) is the same-day return — used downstream both as
    a model input and as the prediction target. Each additional column is the
    return from ``lag`` days earlier, handed to the GRU directly at every
    timestep instead of relying on its recurrence to bridge that many steps.
    Rows without enough history for the largest lag are dropped.

    Args:
        returns: 1-D return series, e.g. from ``compute_returns``.
        lags:    Lag offsets (in days) to add as separate channels.

    Returns:
        A DataFrame indexed like ``returns`` with columns
        ``["return", "return_lag{lag}", ...]``.
    """
    feats = {"return": returns}
    for lag in lags:
        feats[f"return_lag{lag}"] = returns.shift(lag)
    return pd.DataFrame(feats).dropna()


DEFAULT_VOLUME_WINDOW = 20


def build_volume_zscore(volume: pd.Series, window: int = DEFAULT_VOLUME_WINDOW) -> pd.Series:
    """Rolling z-score of trading volume: how unusual today's volume is versus
    its own trailing window. Volume spikes often precede or accompany real
    price moves; z-scoring makes the signal comparable across tickers with
    very different absolute volume levels (e.g. SPY vs. a small-cap).

    Args:
        volume: 1-D trading-volume series.
        window: Trailing window (days) used for the rolling mean/std.

    Returns:
        A ``pd.Series`` named ``"volume_zscore"``, ``NaN`` for the first
        ``window - 1`` rows (insufficient history).
    """
    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std()
    z = (volume - rolling_mean) / (rolling_std + 1e-8)
    z.name = "volume_zscore"
    return z


DEFAULT_VOLATILITY_WINDOW = 10


def build_rolling_volatility(returns: pd.Series, window: int = DEFAULT_VOLATILITY_WINDOW) -> pd.Series:
    """Rolling volatility: trailing standard deviation of daily returns.

    A realized-volatility regime signal, distinct from the 20-day volume
    z-score's timescale — using the shorter 10-day window so it reacts to
    volatility clustering faster. Unlike raw volume, returns are already on a
    small, comparable scale, so no z-scoring is needed here.

    Args:
        returns: 1-D return series, e.g. from ``compute_returns``.
        window:  Trailing window (days) used for the rolling std.

    Returns:
        A ``pd.Series`` named ``"volatility"``, ``NaN`` for the first
        ``window - 1`` rows (insufficient history).
    """
    vol = returns.rolling(window).std()
    vol.name = "volatility"
    return vol


def build_feature_frame(
    df: pd.DataFrame,
    price_col: str = "Close",
    volume_col: str = "Volume",
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    volume_window: int = DEFAULT_VOLUME_WINDOW,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Combine the raw return, lagged-return channels, a volume z-score, and
    rolling volatility into one aligned feature frame.

    Column 0 (``"return"``) stays the raw same-day return — used downstream
    both as a model input and the prediction target. Rows without enough
    history for the longest lag, the volume window, or the volatility window
    are dropped.
    """
    returns = compute_returns(df, price_col)
    features = build_lagged_features(returns, lags)
    vol_z = build_volume_zscore(df[volume_col], volume_window)
    volatility = build_rolling_volatility(returns, volatility_window)
    return features.join(vol_z, how="left").join(volatility, how="left").dropna()


def create_sequences(features: np.ndarray, sequence_length: int, dates: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Create sliding-window samples from a multi-channel feature array.

    ``features`` has shape ``[T, C]`` where column 0 is the raw return (also
    used as the prediction target) and any remaining columns are auxiliary
    channels (e.g. lagged returns) available to the model at every timestep.
    For each index ``i`` the window ``features[i : i + sequence_length]``
    becomes one sample and the target is ``features[i + sequence_length, 0]``.

    Args:
        features:        2-D array of shape ``[T, C]``; column 0 is the raw return.
        sequence_length: Number of past days per sample (lookback window).
        dates:           Optional array of dates aligned with ``features``.

    Returns:
        ``(x, y, dates_out)`` where ``x`` has shape ``[N, sequence_length, C]``,
        ``y`` has shape ``[N]``, and ``dates_out`` has shape ``[N]`` (or ``None``
        if no dates were provided).
    """
    x, y, d = [], [], []
    for i in range(len(features) - sequence_length):
        x.append(features[i : i + sequence_length])
        y.append(features[i + sequence_length, 0])
        if dates is not None:
            d.append(dates[i + sequence_length])

    x_arr = np.asarray(x, dtype=np.float32)  # [N, seq_len, C]
    y_arr = np.asarray(y, dtype=np.float32)  # [N]
    d_arr = np.asarray(d) if dates is not None else None
    return x_arr, y_arr, d_arr


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    dates: np.ndarray | None = None,
) -> SplitData:
    """Chronologically split features and targets into train/val/test sets.

    The split is strictly sequential (no shuffling) to preserve the temporal
    ordering of the time-series data.

    Args:
        x:            Feature array of shape ``[N, ...]``.
        y:            Target array of shape ``[N]``.
        train_ratio:  Fraction of data used for training (default 0.7).
        val_ratio:    Fraction of data used for validation (default 0.15).
                      The remaining portion is used for testing.
        dates:        Optional array of dates aligned with ``x`` and ``y``.

    Returns:
        A :class:`SplitData` dataclass containing the six arrays (and date arrays
        if dates were provided).
    """
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
        dates_train=dates[:train_end] if dates is not None else np.array([]),
        dates_val=dates[train_end:val_end] if dates is not None else np.array([]),
        dates_test=dates[val_end:] if dates is not None else np.array([]),
    )


def expanding_walk_forward_folds(
    x: np.ndarray,
    y: np.ndarray,
    n_folds: int = 4,
    eval_fraction: float = 0.30,
    val_ratio: float = 0.15,
    dates: np.ndarray | None = None,
) -> list[SplitData]:
    """Expanding-window walk-forward splits for out-of-time validation.

    Reserves the trailing ``eval_fraction`` of the sequence as the
    walk-forward evaluation region and divides it into ``n_folds`` contiguous
    test chunks. Fold ``k`` trains on everything chronologically before its
    test chunk (an expanding window -- later folds see strictly more history
    than earlier ones), with a ``val_ratio`` slice carved from the tail of
    that train pool so each fold has the same train/val/test shape
    ``train_val_test_split`` produces.

    With the defaults (``eval_fraction=0.30``, ``n_folds=4``), fold 0's
    train/val boundary lands at 0.70 and fold 2's at 0.85 -- the same cuts
    ``train_val_test_split``'s defaults (``train_ratio=0.7``, ``val_ratio=
    0.15``) produce -- so the single-split result is recoverable as one point
    in this sequence rather than a separate methodology.

    Args:
        x, y:          Full windowed feature/target arrays (unsplit).
        n_folds:       Number of walk-forward folds.
        eval_fraction: Fraction of the sequence reserved as the evaluation
                       region, split into ``n_folds`` contiguous test chunks.
        val_ratio:     Fraction of each fold's train pool held out as val.
        dates:         Optional dates array aligned with ``x`` and ``y``.

    Returns:
        A list of ``n_folds`` :class:`SplitData`, one per fold, in
        chronological order.
    """
    n = len(x)
    eval_start = int(n * (1 - eval_fraction))
    fold_size = (n - eval_start) // n_folds

    folds: list[SplitData] = []
    for k in range(n_folds):
        test_start = eval_start + k * fold_size
        test_end = n if k == n_folds - 1 else eval_start + (k + 1) * fold_size
        val_start = int(test_start * (1 - val_ratio))

        assert 0 <= val_start <= test_start <= test_end <= n, (
            f"fold {k}: invalid boundaries val_start={val_start} test_start={test_start} "
            f"test_end={test_end} n={n}"
        )
        if dates is not None and val_start > 0 and test_start < test_end:
            assert dates[val_start - 1] < dates[test_start], (
                f"fold {k}: train-pool date {dates[val_start - 1]} not before "
                f"test date {dates[test_start]} -- would leak future data into training"
            )

        folds.append(
            SplitData(
                x_train=x[:val_start],
                y_train=y[:val_start],
                x_val=x[val_start:test_start],
                y_val=y[val_start:test_start],
                x_test=x[test_start:test_end],
                y_test=y[test_start:test_end],
                dates_train=dates[:val_start] if dates is not None else np.array([]),
                dates_val=dates[val_start:test_start] if dates is not None else np.array([]),
                dates_test=dates[test_start:test_end] if dates is not None else np.array([]),
            )
        )
    return folds
