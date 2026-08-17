from __future__ import annotations # Enables forward references in type hints

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch, to_loader
from utils.preprocessing import compute_returns, create_sequences, load_stock_data, train_val_test_split


def plot_history(history: dict, output_dir: Path) -> None:
    """Plot training loss and cumulative profit curves.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss',
            'train_profit', and 'val_profit' lists.
        output_dir: Directory where the plots will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Profit-Aware Loss with Transaction Costs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_profit"], label="Train Cumulative Profit")
    plt.plot(history["val_profit"], label="Val Cumulative Profit")
    plt.title("Cumulative Profit per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "profit_curve.png", dpi=200)
    plt.close()


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
    parser.add_argument("--alpha", type=float, default=1.0,                help="Sharpness of tanh signal: higher = more aggressive binary-like positioning")
    parser.add_argument("--transaction-cost", type=float, default=0.001,   help="Transaction cost rate per unit of signal change (0.001 = 0.1% per trade)")

    # Training
    parser.add_argument("--epochs", type=int, default=50,                  help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,              help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3,                  help="Adam optimizer learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = Path("data") / f"{args.symbol}_{args.start}_{args.end or 'latest'}.csv"

    # Load Stock Data
    df = load_stock_data(args.symbol, args.start, args.end, str(cache_path))

    # Convert Prices into Returns
    # return_t = \frac{P_{t+1} - P_t}{P_t}
    returns = compute_returns(df).values

    # Create Sequential Windows (This creates time-series samples)
    # x: shape [N, sequence_length, 1] — N samples, each is a window of 30 returns (1 feature per day)
    # y: shape [N] — N single return values
    x, y = create_sequences(returns, sequence_length=args.sequence_length)

    # Train / Validation / Test Split
    split = train_val_test_split(x, y)

    # DataLoader
    train_loader = to_loader(split.x_train, split.y_train, args.batch_size, shuffle=False)
    val_loader = to_loader(split.x_val, split.y_val, args.batch_size, shuffle=False)
    test_loader = to_loader(split.x_test, split.y_test, args.batch_size, shuffle=False)

    # GRU Model
    model = GRUReturnPredictor(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    config = TrainingConfig(
        alpha=args.alpha,
        transaction_cost_rate=args.transaction_cost,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    history = fit(model, train_loader, val_loader, config, device)

    # run model
    test_metrics = run_epoch(
        model,
        test_loader,
        optimizer=None,
        alpha=args.alpha,
        transaction_cost_rate=args.transaction_cost,
        device=device,
    )

    print("\nTest Metrics")
    print(f"Loss: {test_metrics['loss']:.6f}")
    print(f"Directional Accuracy: {test_metrics['directional_acc']:.4f}")
    print(f"Cumulative Profit: {test_metrics['cum_profit']:.6f}")
    print(f"Sharpe-like Ratio: {test_metrics['sharpe_like']:.4f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), results_dir / "gru_profit_aware.pt")
    plot_history(history, results_dir)


if __name__ == "__main__":
    main()
