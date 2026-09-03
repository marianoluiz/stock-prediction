"""Train a GRU model and save its weights.

This script does NOT touch the test set. Training/validation curves are saved
next to the model so you can evaluate the held-out test set later with
``evaluate.py`` using the exact same split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit
from utils.pipeline import prepare_data
from utils.plotting import format_date, plot_history
from utils.trading import calibrate_alpha


def main() -> None:
    """Train a GRU model with the chosen loss and save it to disk."""
    parser = argparse.ArgumentParser(description="Train a GRU model (test set is left untouched)")

    # Data
    parser.add_argument("--symbol", type=str, default="AAPL",              help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--start", type=str, default="2018-01-01",         help="Start date for historical data (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,                   help="End date for historical data (default: latest available)")
    parser.add_argument("--sequence-length", type=int, default=30,         help="Number of past days used as input for each sample (lookback window)")

    # Model Architecture
    parser.add_argument("--hidden-size", type=int, default=64,             help="Number of hidden units in the GRU layers")
    parser.add_argument("--num-layers", type=int, default=2,               help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2,              help="Dropout rate between GRU layers (regularization)")

    # Trading / Loss
    parser.add_argument("--loss", type=str, default="profit-aware",        choices=["mse", "profit-aware"],
                        help="Loss function: 'mse' (baseline) or 'profit-aware' (custom)")
    parser.add_argument("--alpha", type=float, default=None,               help="Sharpness of tanh signal: higher = more aggressive binary-like positioning (default: auto-calibrated as 1/std(train_returns) so a 1-std move maps to tanh~=0.76)")
    parser.add_argument("--loss-lambda", type=float, default=0.1,          help="Weight of the MSE calibration term in the profit-aware loss (0 = pure profit, larger = more calibration; 0 lets pred drift unbounded and saturate tanh, see TODO.md)")
    parser.add_argument("--transaction-cost", type=float, default=0.001,   help="Transaction cost rate per unit of signal change (0.001 = 0.1%% per trade)")
    parser.add_argument("--capital", type=float, default=100_000.0,        help="Starting capital in PHP for simulated trading display")

    # Training
    parser.add_argument("--epochs", type=int, default=50,                  help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,              help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,                  help="Adam optimizer learning rate")
    parser.add_argument("--early-stop-patience", type=int, default=0,      help="Stop training if --early-stop-metric doesn't improve for this many epochs (0 = disabled, train the full --epochs). Restores the best-epoch weights before saving.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum change in --early-stop-metric to count as an improvement")
    parser.add_argument("--early-stop-metric", type=str, default="val_loss", choices=["val_loss", "val_profit", "val_profit_geo", "val_dir_acc"],
                        help="Validation metric to monitor for early stopping")

    # Output
    parser.add_argument("--save", type=str, default=None,                  help="Path to save model weights (.pt). Defaults to results/<loss>/<SYMBOL>_<loss>_<START>_<END>.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data(args.symbol, args.start, args.end, args.sequence_length, args.batch_size)
    split = data.split

    if args.alpha is None:
        args.alpha = calibrate_alpha(split.y_train)
        print(f"Auto-calibrated alpha: {args.alpha:.4f} (train return std = {split.y_train.std():.6f})")

    print(f"Train: {len(split.x_train)} samples ({split.dates_train[0]} -> {split.dates_train[-1]})")
    print(f"Val:   {len(split.x_val)} samples ({split.dates_val[0]} -> {split.dates_val[-1]})")
    print(f"Test:  {len(split.x_test)} samples ({split.dates_test[0]} -> {split.dates_test[-1]})")
    print("Note: the test set is left untouched. Evaluate it later with evaluate.py.")

    model = GRUReturnPredictor(
        input_size=split.x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    config = TrainingConfig(
        loss_type=args.loss,
        alpha=args.alpha,
        loss_lambda=args.loss_lambda,
        transaction_cost_rate=args.transaction_cost,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_metric=args.early_stop_metric,
    )

    label = args.loss.upper().replace("-", " ")
    print(f"\n{'='*60}")
    print(f"  Training: {label} loss")
    print(f"{'='*60}\n")

    history = fit(model, data.train_loader, data.val_loader, config, device, capital=args.capital)

    loss_key = args.loss.replace("-", "_")
    end_actual = format_date(data.dates[-1])
    tag = f"{args.symbol}_{loss_key}_{args.start}_{end_actual}"

    save_path = Path(args.save) if args.save else Path("results") / loss_key / f"{tag}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")

    plot_history(
        history,
        save_path.parent,
        capital=args.capital,
        symbol=args.symbol,
        start=args.start,
        end=end_actual,
        loss_label=label,
        split=split,
        file_prefix=tag,
    )
    print(f"Saved loss/profit curves to {save_path.parent}")


if __name__ == "__main__":
    main()