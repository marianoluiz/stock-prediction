from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch
from utils.metrics import trade_log
from utils.pipeline import prepare_data
from utils.plotting import plot_history
from utils.preprocessing import SplitData


def run_single(
    loss_type: str,
    args: argparse.Namespace,
    split: SplitData,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    show_trade_log: bool = False,
) -> tuple[dict, dict]:
    """Train one model with the given loss type and return (history, test_metrics)."""
    model = GRUReturnPredictor(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    config = TrainingConfig(
        loss_type=loss_type,
        alpha=args.alpha,
        transaction_cost_rate=args.transaction_cost,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    label = loss_type.upper().replace("-", " ")
    print(f"\n{'='*60}")
    print(f"  Training: {label} loss")
    print(f"{'='*60}\n")

    history = fit(model, train_loader, val_loader, config, device, capital=args.capital)

    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        args.alpha,
        args.transaction_cost,
        device,
        loss_type=loss_type,
    )

    print(f"\nTest Metrics [{label}] ({split.dates_test[0]} -> {split.dates_test[-1]})")
    print(f"  Loss:                {test_metrics['loss']:.6f}")
    print(f"  MSE:                 {test_metrics['mse']:.8f}")
    print(f"  MAE:                 {test_metrics['mae']:.8f}")
    print(f"  RMSE:                {test_metrics['rmse']:.8f}")
    print(f"  Directional Acc:     {test_metrics['directional_acc']:.4f}")
    print(f"  Cumulative Return:   {test_metrics['cum_profit']:.6f} ({test_metrics['cum_profit'] * args.capital:+,.2f} PHP)")
    print(f"  Geometric Return:    {test_metrics['cum_profit_geo']:.6f} ({test_metrics['cum_profit_geo'] * args.capital:+,.2f} PHP)")
    print(f"  Sharpe-like Ratio:   {test_metrics['sharpe_like']:.4f}")

    if show_trade_log:
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

    results_dir = Path("results") / loss_type.replace("-", "_")
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), results_dir / f"gru_{loss_type.replace('-', '_')}.pt")
    plot_history(history, results_dir, capital=args.capital, title=f"{label} Loss")

    return history, test_metrics


def main() -> None:
    """Train and evaluate a GRU model with profit-aware loss."""
    parser = argparse.ArgumentParser(description="Enhanced GRU with Profit-Aware Loss and Transaction Costs")

    #  Data
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
    parser.add_argument("--compare", action="store_true",                  help="Run both MSE and profit-aware, print comparison table")
    parser.add_argument("--trade-log", action="store_true",                help="Print per-trade P&L log for every test trade")
    parser.add_argument("--alpha", type=float, default=1.0,                help="Sharpness of tanh signal: higher = more aggressive binary-like positioning")
    parser.add_argument("--transaction-cost", type=float, default=0.001,   help="Transaction cost rate per unit of signal change (0.001 = 0.1%% per trade)")
    parser.add_argument("--capital", type=float, default=100_000.0,        help="Starting capital in PHP for simulated trading display (default: 100,000)")

    # Training
    parser.add_argument("--epochs", type=int, default=50,                  help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,              help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,                  help="Adam optimizer learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data, compute returns, window, and chronologically split — the exact
    # same pipeline used by train.py and evaluate.py.
    data = prepare_data(args.symbol, args.start, args.end, args.sequence_length, args.batch_size)
    split = data.split

    print(f"Train: {len(split.x_train)} samples ({split.dates_train[0]} -> {split.dates_train[-1]})")
    print(f"Val:   {len(split.x_val)} samples ({split.dates_val[0]} -> {split.dates_val[-1]})")
    print(f"Test:  {len(split.x_test)} samples ({split.dates_test[0]} -> {split.dates_test[-1]})")

    train_loader = data.train_loader
    val_loader = data.val_loader
    test_loader = data.test_loader

    if args.compare:
        _, mse_test = run_single("mse", args, split, train_loader, val_loader, test_loader, device, show_trade_log=args.trade_log)
        _, pa_test = run_single("profit-aware", args, split, train_loader, val_loader, test_loader, device, show_trade_log=args.trade_log)

        print(f"\n{'='*60}")
        print("  COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'MSE Baseline':>15} {'Profit-Aware':>15}")
        print(f"  {'-'*55}")
        print(f"  {'Test Loss':<25} {mse_test['loss']:>15.6f} {pa_test['loss']:>15.6f}")
        print(f"  {'MSE':<25} {mse_test['mse']:>15.8f} {pa_test['mse']:>15.8f}")
        print(f"  {'MAE':<25} {mse_test['mae']:>15.8f} {pa_test['mae']:>15.8f}")
        print(f"  {'RMSE':<25} {mse_test['rmse']:>15.8f} {pa_test['rmse']:>15.8f}")
        print(f"  {'Directional Accuracy':<25} {mse_test['directional_acc']:>15.4f} {pa_test['directional_acc']:>15.4f}")
        print(f"  {'Cumulative Return':<25} {mse_test['cum_profit']:>15.6f} {pa_test['cum_profit']:>15.6f}")
        print(f"  {'  (in PHP)':<25} {mse_test['cum_profit'] * args.capital:>+14,.0f} {pa_test['cum_profit'] * args.capital:>+14,.0f}")
        print(f"  {'Geometric Return':<25} {mse_test['cum_profit_geo']:>15.6f} {pa_test['cum_profit_geo']:>15.6f}")
        print(f"  {'  (in PHP)':<25} {mse_test['cum_profit_geo'] * args.capital:>+14,.0f} {pa_test['cum_profit_geo'] * args.capital:>+14,.0f}")
        print(f"  {'Sharpe-like Ratio':<25} {mse_test['sharpe_like']:>15.4f} {pa_test['sharpe_like']:>15.4f}")
        print(f"  {'-'*55}")
        print()
    else:
        run_single(args.loss, args, split, train_loader, val_loader, test_loader, device, show_trade_log=args.trade_log)


if __name__ == "__main__":
    main()
