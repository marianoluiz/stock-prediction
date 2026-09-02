"""Multi-stock benchmark: train MSE and profit-aware models across many
tickers and report aggregate metrics, so a single-stock result can't be
mistaken for a general improvement.

Uses the exact same data pipeline, model, and training loop as main.py
(build_feature_frame's lagged-return + volume-zscore channels included) —
this script only adds the loop over symbols and the aggregate summary.
Skips torch.save/plotting per run to stay fast; see main.py for a single
stock's full artifacts (weights + curves).

Usage:
    python benchmark.py
    python benchmark.py --symbols AAPL,MSFT,NVDA --epochs 30
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch
from utils.pipeline import prepare_data
from utils.trading import calibrate_alpha

TIERED_SYMBOLS = {
    # persistent uptrend, model's easy case
    "strong": ["NVDA", "AAPL", "MSFT", "LLY", "COST", "AVGO", "V"],
    # choppy / sideways / mixed years, the realistic case
    "mid": ["GOOGL", "JPM", "DIS", "KO", "JNJ", "AMZN", "WMT"],
    # multi-year decline or high downside volatility, stress test
    "weak": ["INTC", "F", "KHC", "BA", "T", "PARA", "GRAB"],
    # passive-index references, not scored as a performance tier
    "index": ["SPY", "PSEI.PS"],
}
DEFAULT_SYMBOLS = [s for tier in TIERED_SYMBOLS.values() for s in tier]
SYMBOL_TIER = {s: tier for tier, symbols in TIERED_SYMBOLS.items() for s in symbols}


def buy_and_hold_return(actual_returns: np.ndarray) -> float:
    """Cumulative return of a naive always-long position over the test window."""
    return float(np.prod(1.0 + actual_returns) - 1.0)


def run_one(symbol: str, loss_type: str, args: argparse.Namespace, device: torch.device, seed: int) -> dict:
    data = prepare_data(symbol, args.start, args.end, args.sequence_length, args.batch_size)
    split = data.split

    alpha = args.alpha if args.alpha is not None else calibrate_alpha(split.y_train)

    torch.manual_seed(seed)
    model = GRUReturnPredictor(
        input_size=split.x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    config = TrainingConfig(
        loss_type=loss_type,
        alpha=alpha,
        loss_lambda=args.loss_lambda,
        signal_threshold=args.signal_threshold,
        transaction_cost_rate=args.transaction_cost,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    fit(model, data.train_loader, data.val_loader, config, device, capital=args.capital)

    test_metrics = run_epoch(
        model, data.test_loader, None, alpha, args.transaction_cost, device,
        loss_type=loss_type, signal_threshold=args.signal_threshold,
    )
    test_metrics["symbol"] = symbol
    test_metrics["loss_type"] = loss_type
    test_metrics["tier"] = SYMBOL_TIER.get(symbol, "?")
    test_metrics["n_test"] = len(split.x_test)
    test_metrics["buy_hold"] = buy_and_hold_return(test_metrics["actual_np"])
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MSE vs profit-aware GRU across many stocks")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="Comma-separated ticker list")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--loss-lambda", type=float, default=0.1)
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: results/benchmark_summary_<timestamp>.csv)")
    args = parser.parse_args()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/benchmark_summary_{stamp}.csv"

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Benchmarking {len(symbols)} symbols x 2 losses: {symbols}\n")

    rows: list[dict] = []
    t_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        for loss_type in ("mse", "profit-aware"):
            t0 = time.time()
            try:
                m = run_one(symbol, loss_type, args, device, seed=args.seed)
            except Exception as e:
                print(f"[{i}/{len(symbols)}] {symbol:10s} {loss_type:13s} FAILED: {e}")
                continue
            dt = time.time() - t0
            rows.append(m)
            print(
                f"[{i}/{len(symbols)}] {symbol:10s} ({m['tier']:<6s}) {loss_type:13s} "
                f"dir_acc={m['directional_acc']:.3f} cum_ret={m['cum_profit']:+.4f} "
                f"geo_ret={m['cum_profit_geo']:+.4f} sharpe={m['sharpe_like']:+.3f} "
                f"(buy&hold={m['buy_hold']:+.4f}) [{dt:.1f}s]"
            )

    total_dt = time.time() - t_start
    print(f"\nTotal benchmark time: {total_dt:.1f}s\n")

    # --- Aggregate summary -------------------------------------------------
    print(f"{'='*78}")
    print("  AGGREGATE SUMMARY (mean across symbols)")
    print(f"{'='*78}")
    header = f"  {'Loss':<15}{'DirAcc':>10}{'CumRet':>12}{'GeoRet':>12}{'Sharpe':>10}{'RMSE':>10}"
    print(header)
    print(f"  {'-'*74}")
    for loss_type in ("mse", "profit-aware"):
        sub = [r for r in rows if r["loss_type"] == loss_type]
        if not sub:
            continue
        dir_acc = np.mean([r["directional_acc"] for r in sub])
        cum_ret = np.mean([r["cum_profit"] for r in sub])
        geo_ret = np.mean([r["cum_profit_geo"] for r in sub])
        sharpe = np.mean([r["sharpe_like"] for r in sub])
        rmse = np.mean([r["rmse"] for r in sub])
        print(f"  {loss_type:<15}{dir_acc:>10.4f}{cum_ret:>+12.4f}{geo_ret:>+12.4f}{sharpe:>+10.3f}{rmse:>10.5f}")

    bh = np.mean([r["buy_hold"] for r in rows if r["loss_type"] == "mse"]) if rows else float("nan")
    print(f"  {'buy & hold':<15}{'':>10}{bh:>+12.4f}")

    # --- Per-tier breakdown ---------------------------------------------------
    print(f"\n{'='*78}")
    print("  BY TIER (mean across symbols in tier)")
    print(f"{'='*78}")
    print(header)
    print(f"  {'-'*74}")
    for tier in ("strong", "mid", "weak", "index"):
        for loss_type in ("mse", "profit-aware"):
            sub = [r for r in rows if r["tier"] == tier and r["loss_type"] == loss_type]
            if not sub:
                continue
            dir_acc = np.mean([r["directional_acc"] for r in sub])
            cum_ret = np.mean([r["cum_profit"] for r in sub])
            geo_ret = np.mean([r["cum_profit_geo"] for r in sub])
            sharpe = np.mean([r["sharpe_like"] for r in sub])
            rmse = np.mean([r["rmse"] for r in sub])
            label = f"{tier}/{loss_type}"
            print(f"  {label:<15}{dir_acc:>10.4f}{cum_ret:>+12.4f}{geo_ret:>+12.4f}{sharpe:>+10.3f}{rmse:>10.5f}")
        print()

    # --- Win-rate: profit-aware vs mse, symbol by symbol --------------------
    by_symbol: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], {})[r["loss_type"]] = r

    wins_geo = wins_sharpe = wins_diracc = n_pairs = 0
    for symbol, d in by_symbol.items():
        if "mse" not in d or "profit-aware" not in d:
            continue
        n_pairs += 1
        wins_geo += d["profit-aware"]["cum_profit_geo"] > d["mse"]["cum_profit_geo"]
        wins_sharpe += d["profit-aware"]["sharpe_like"] > d["mse"]["sharpe_like"]
        wins_diracc += d["profit-aware"]["directional_acc"] > d["mse"]["directional_acc"]

    print(f"\n  Profit-aware beats MSE on (out of {n_pairs} symbols):")
    print(f"    Geometric return: {wins_geo}/{n_pairs}")
    print(f"    Sharpe-like:      {wins_sharpe}/{n_pairs}")
    print(f"    Directional acc:  {wins_diracc}/{n_pairs}")

    # --- Save CSV ------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("symbol,tier,loss_type,n_test,directional_acc,cum_profit,cum_profit_geo,sharpe_like,mse,mae,rmse,buy_hold\n")
        for r in rows:
            f.write(
                f"{r['symbol']},{r['tier']},{r['loss_type']},{r['n_test']},{r['directional_acc']:.6f},"
                f"{r['cum_profit']:.6f},{r['cum_profit_geo']:.6f},{r['sharpe_like']:.6f},"
                f"{r['mse']:.8f},{r['mae']:.8f},{r['rmse']:.8f},{r['buy_hold']:.6f}\n"
            )
    print(f"\nSaved per-symbol results to {out_path}")


if __name__ == "__main__":
    main()
