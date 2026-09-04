"""Evaluate a saved GRU model on the held-out test set.

Uses the exact same data-preparation pipeline as train.py (identical split),
so the test window here is the same dates the model never saw during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.gru_model import GRUReturnPredictor
from training.train import run_epoch
from utils.metrics import trade_log
from utils.pipeline import prepare_data
from utils.plotting import format_date
from utils.trading import calibrate_alpha


def main() -> None:
    """Load a saved model and print test-set metrics."""
    parser = argparse.ArgumentParser(description="Evaluate a saved GRU model on the held-out test set")

    parser.add_argument("--model", type=str, required=True,                help="Path to saved model weights (.pt)")
    parser.add_argument("--loss", type=str, default="profit-aware",        choices=["mse", "profit-aware"],
                        help="Loss function used to train the model (for loss reporting)")

    # Data
    parser.add_argument("--symbol", type=str, default="AAPL",              help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--start", type=str, default="2018-01-01",         help="Start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,                   help="End date for historical data (default: latest available)")
    parser.add_argument("--sequence-length", type=int, default=30,         help="Number of past days used as input for each sample (lookback window)")

    # Model Architecture
    parser.add_argument("--hidden-size", type=int, default=64,             help="Number of hidden units in the GRU layers")
    parser.add_argument("--num-layers", type=int, default=2,               help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2,              help="Dropout rate between GRU layers")

    # Trading / Display
    parser.add_argument("--alpha", type=float, default=None,               help="Sharpness of tanh signal (default: auto-calibrated as 1/std(train_returns) so a 1-std move maps to tanh~=0.76)")
    parser.add_argument("--loss-lambda", type=float, default=0.1,          help="Weight of the MSE calibration term in the profit-aware loss (0 lets pred drift unbounded and saturate tanh)")
    parser.add_argument("--transaction-cost", type=float, default=0.001,   help="Transaction cost rate per unit of signal change")
    parser.add_argument("--capital", type=float, default=100_000.0,        help="Starting capital in PHP for simulated trading display")
    parser.add_argument("--batch-size", type=int, default=64,              help="Mini-batch size for evaluation")
    parser.add_argument("--trade-log", action="store_true",                help="Print a per-trade P&L log for every test trade")
    parser.add_argument("--metrics-out", type=str, default=None,            help="Optional path to write a text summary of the metrics")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data(args.symbol, args.start, args.end, args.sequence_length, args.batch_size)
    split = data.split

    if args.alpha is None:
        args.alpha = calibrate_alpha(split.y_train)
        print(f"Auto-calibrated alpha: {args.alpha:.4f} (train return std = {split.y_train.std():.6f})")

    end_actual = format_date(data.dates[-1])
    print(f"Test period: {format_date(split.dates_test[0])} -> {format_date(split.dates_test[-1])} ({len(split.x_test)} samples)")

    model = GRUReturnPredictor(
        input_size=split.x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    label = args.loss.upper().replace("-", " ")
    print(f"\nTest Metrics [{label}] ({model_path.name})")
    test_metrics = run_epoch(
        model,
        data.test_loader,
        None,
        args.alpha,
        args.transaction_cost,
        device,
        loss_type=args.loss,
        loss_lambda=args.loss_lambda,
    )

    print(f"  Loss:                {test_metrics['loss']:.6f}")
    print(f"  MSE:                 {test_metrics['mse']:.8f}")
    print(f"  MAE:                 {test_metrics['mae']:.8f}")
    print(f"  RMSE:                {test_metrics['rmse']:.8f}")
    print(f"  Directional Acc:     {test_metrics['directional_acc']:.4f}")
    print(f"  Cumulative Return:   {test_metrics['cum_profit']:.6f} ({test_metrics['cum_profit'] * args.capital:+,.2f} PHP)")
    print(f"  Geometric Return:    {test_metrics['cum_profit_geo']:.6f} ({test_metrics['cum_profit_geo'] * args.capital:+,.2f} PHP)")
    print(f"  Sharpe-like Ratio:   {test_metrics['sharpe_like']:.4f}")

    if args.trade_log:
        print(f"\n{'='*60}")
        print(f"  TRADE LOG [{label}]")
        print(f"{'='*60}")
        trade_log(
            test_metrics["signal_np"],
            test_metrics["actual_np"],
            test_metrics["pred_np"],
            split.dates_test,
            args.capital,
            transaction_cost_rate=args.transaction_cost,
        )

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = (
            f"Symbol:         {args.symbol}\n"
            f"Reporting period: {args.start} -> {end_actual}\n"
            f"Model:          {model_path}\n"
            f"Loss:           {args.loss}\n"
            f"Test period:    {format_date(split.dates_test[0])} -> {format_date(split.dates_test[-1])} "
            f"({len(split.x_test)} samples)\n"
            f"\n"
            f"Loss:                  {test_metrics['loss']:.6f}\n"
            f"MSE:                   {test_metrics['mse']:.8f}\n"
            f"MAE:                   {test_metrics['mae']:.8f}\n"
            f"RMSE:                  {test_metrics['rmse']:.8f}\n"
            f"Directional Accuracy:  {test_metrics['directional_acc']:.4f}\n"
            f"Cumulative Return:     {test_metrics['cum_profit']:.6f} ({test_metrics['cum_profit'] * args.capital:+,.2f} PHP)\n"
            f"Geometric Return:      {test_metrics['cum_profit_geo']:.6f} ({test_metrics['cum_profit_geo'] * args.capital:+,.2f} PHP)\n"
            f"Sharpe-like Ratio:     {test_metrics['sharpe_like']:.4f}\n"
        )
        out.write_text(body, encoding="utf-8")
        print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()