"""Iteratively forecast the next N daily returns with a saved GRU model.

Seed predictions with the last ``sequence_length`` returns from the requested
range, then feed each predicted return back as input for the next day.
Useful for a live thesis-defense demo without retraining.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import numpy as np

from models.gru_model import GRUReturnPredictor
from utils.pipeline import cache_path_for
from utils.preprocessing import (
    DEFAULT_LAGS,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_VOLUME_WINDOW,
    build_lagged_features,
    build_rolling_volatility,
    build_volume_zscore,
    compute_returns,
    load_stock_data,
)


def main() -> None:
    """Print the next N predicted returns/signals for a stock."""
    parser = argparse.ArgumentParser(description="Forecast the next N days after the latest data")

    parser.add_argument("--model", type=str, required=True,                help="Path to saved model weights (.pt)")
    parser.add_argument("--symbol", type=str, default="AAPL",              help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--start", type=str, default="2018-01-01",         help="Start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,                   help="End date for historical data (default: latest available)")
    parser.add_argument("--days", type=int, default=5,                     help="Number of future trading days to forecast")
    parser.add_argument("--sequence-length", type=int, default=30,         help="Number of past days used as input for each sample (lookback window)")

    # Model Architecture
    parser.add_argument("--hidden-size", type=int, default=64,             help="Number of hidden units in the GRU layers")
    parser.add_argument("--num-layers", type=int, default=2,               help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2,              help="Dropout rate between GRU layers")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_stock_data(args.symbol, args.start, args.end, cache_path_for(args.symbol, args.start, args.end))
    returns_series = compute_returns(df)
    volume_series = df["Volume"].astype("float32")
    last_date = pd.Timestamp(df.index[-1])
    last_volume = float(volume_series.iloc[-1])

    min_history = args.sequence_length + max(max(DEFAULT_LAGS), DEFAULT_VOLUME_WINDOW, DEFAULT_VOLATILITY_WINDOW)
    if len(returns_series) < min_history:
        raise ValueError(
            f"Only {len(returns_series)} returns available but sequence_length={args.sequence_length} "
            f"needs {min_history} days of history (lookback + max(longest lag, volume window, "
            "volatility window)). Use an earlier --start date."
        )

    model = GRUReturnPredictor(
        input_size=1 + len(DEFAULT_LAGS) + 1 + 1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def latest_window(returns: pd.Series, volume: pd.Series) -> np.ndarray:
        # Recomputes lag channels (return_lag1/5/10), the volume z-score, and
        # rolling volatility from the running series so each forecasted
        # return is available to later lag/volatility channels too.
        features = build_lagged_features(returns, DEFAULT_LAGS)
        vol_z = build_volume_zscore(volume, DEFAULT_VOLUME_WINDOW)
        volatility = build_rolling_volatility(returns, DEFAULT_VOLATILITY_WINDOW)
        combined = features.join(vol_z, how="left").join(volatility, how="left").dropna()
        return combined.values[-args.sequence_length:].astype("float32")

    window = torch.from_numpy(latest_window(returns_series, volume_series)).unsqueeze(0).to(device)
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=args.days)

    print(f"\nData for {args.symbol} up to {last_date.date()} - forecasting the next {args.days} trading days")
    print(f"  {'Date':>12}  {'Predicted Return':>17}  {'Signal':>9}")

    preds: list[float] = []
    with torch.no_grad():
        for _ in range(args.days):
            pred = float(model(window).item())
            preds.append(pred)
            next_index = returns_series.index[-1] + pd.Timedelta(days=1)
            returns_series = pd.concat([returns_series, pd.Series([pred], index=[next_index])])
            # No real future volume is observable; carry the last known
            # volume forward so its z-score stays neutral instead of
            # fabricating a trend.
            volume_series = pd.concat([volume_series, pd.Series([last_volume], index=[next_index])])
            window = torch.from_numpy(latest_window(returns_series, volume_series)).unsqueeze(0).to(device)

    for day, p in zip(future_dates, preds):
        signal = "LONG (+1)" if p > 0 else ("SHORT (-1)" if p < 0 else "FLAT (0)")
        print(f"  {str(day.date()):>12}  {p:>+16.4%}  {signal:>9}")

    print("\nNote: predictions are iterative - each day's forecast is fed back as input.")


if __name__ == "__main__":
    main()