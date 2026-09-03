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
from utils.pipeline import prepare_data, prepare_walk_forward
from utils.trading import calibrate_alpha

TIERED_SYMBOLS = {
    # persistent uptrend, model's easy case
    "strong": ["NVDA", "MSFT", "AAPL", "COST", "V", "AVGO"],
    # choppy / sideways / mixed years, the realistic case
    "mid": ["DIS", "KO", "JNJ", "XOM", "PFE", "WMT"],
    # multi-year decline, downtrend only (no volatility whipsaws), stress test
    "weak": ["T", "KHC", "F", "PYPL", "BA", "WBA"],
    # passive-index references, not scored as a performance tier
    "index": ["SPY", "PSEI.PS"],
}
DEFAULT_SYMBOLS = [s for tier in TIERED_SYMBOLS.values() for s in tier]
SYMBOL_TIER = {s: tier for tier, symbols in TIERED_SYMBOLS.items() for s in symbols}


def buy_and_hold_return(actual_returns: np.ndarray) -> float:
    """Cumulative return of a naive always-long position over the test window."""
    return float(np.prod(1.0 + actual_returns) - 1.0)


def always_short_return(actual_returns: np.ndarray) -> float:
    """Cumulative return of a naive always-short position over the test window."""
    return float(np.prod(1.0 - actual_returns) - 1.0)


def is_degenerate_signal(signal: np.ndarray) -> bool:
    """True if the executed signal never changes across the whole test window.

    A model that outputs the same position on every day regardless of input
    learned no conditional (day-to-day) signal at all -- it's indistinguishable
    from a naive always-long/always-short baseline, whatever its profit looks
    like.
    """
    return bool(np.unique(signal).size == 1)


def run_one_split(
    symbol: str,
    loss_type: str,
    split,
    train_loader,
    val_loader,
    test_loader,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    fold: int,
) -> dict:
    """Train + evaluate one (symbol, loss_type) pair on an already-built split.

    Shared body for both the legacy single-split path (``fold=0``) and each
    walk-forward fold -- a fresh, identically-seeded model every call, same
    as the legacy per-(symbol, loss) behavior.
    """
    alpha = args.alpha if args.alpha is not None else calibrate_alpha(split.y_train)
    output_scale = (args.output_cap_std / alpha) if args.output_cap_std > 0 else None

    torch.manual_seed(seed)
    model = GRUReturnPredictor(
        input_size=split.x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_scale=output_scale,
    ).to(device)

    config = TrainingConfig(
        loss_type=loss_type,
        alpha=alpha,
        loss_lambda=args.loss_lambda,
        signal_threshold=args.signal_threshold,
        transaction_cost_rate=args.transaction_cost,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_metric=args.early_stop_metric,
    )

    fit(model, train_loader, val_loader, config, device, capital=args.capital)

    test_metrics = run_epoch(
        model, test_loader, None, alpha, args.transaction_cost, device,
        loss_type=loss_type, signal_threshold=args.signal_threshold,
    )
    test_metrics["symbol"] = symbol
    test_metrics["loss_type"] = loss_type
    test_metrics["tier"] = SYMBOL_TIER.get(symbol, "?")
    test_metrics["fold"] = fold
    test_metrics["n_test"] = len(split.x_test)
    test_metrics["buy_hold"] = buy_and_hold_return(test_metrics["actual_np"])
    test_metrics["always_short"] = always_short_return(test_metrics["actual_np"])
    test_metrics["degenerate"] = is_degenerate_signal(test_metrics["signal_np"])
    return test_metrics


def run_one(symbol: str, loss_type: str, args: argparse.Namespace, device: torch.device, seed: int) -> dict:
    data = prepare_data(symbol, args.start, args.end, args.sequence_length, args.batch_size)
    return run_one_split(
        symbol, loss_type, data.split, data.train_loader, data.val_loader, data.test_loader,
        args, device, seed, fold=0,
    )


def run_one_walk_forward(
    symbol: str, loss_type: str, args: argparse.Namespace, device: torch.device, seed: int
) -> list[dict]:
    folds = prepare_walk_forward(
        symbol, args.start, args.end, args.sequence_length, args.batch_size,
        n_folds=args.folds, eval_fraction=args.eval_fraction,
    )
    return [
        run_one_split(
            symbol, loss_type, f.split, f.train_loader, f.val_loader, f.test_loader,
            args, device, seed, fold=f.fold,
        )
        for f in folds
    ]


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
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay; caps pred magnitude from diverging when chasing tanh saturation")
    parser.add_argument("--output-cap-std", type=float, default=5.0, help="Hard-bound predicted return to +-N std devs of train returns via a tanh head (0 = unbounded)")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop and restore best-validation-epoch weights if --early-stop-metric doesn't improve for this many epochs (0 = disabled, always use final-epoch weights). Empirically, on this dataset/model, enabling this made results WORSE (higher degenerate-policy rate, lower win rate vs MSE) regardless of metric watched -- see results/benchmark_full_v2 vs v3/v4. The trivial constant-direction solution is an early, flat local minimum; patience-based stopping locks onto it before training escapes it. Left available for further experimentation, but off by default.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="Minimum change in --early-stop-metric to count as an improvement")
    parser.add_argument("--early-stop-metric", type=str, default="val_loss", choices=["val_loss", "val_profit", "val_profit_geo", "val_dir_acc"], help="Validation metric to monitor for early stopping / best-checkpoint selection. Avoid val_profit/val_profit_geo as the default: on a ~300-day val window they're dominated by a handful of large-return days, so a checkpoint that happens to collapse to a constant-direction bet matching the val window's drift can look 'best' by luck and get locked in (verified empirically -- see results/benchmark_full_v3 vs v4).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: results/benchmark_summary_<timestamp>.csv)")
    parser.add_argument("--walk-forward", action="store_true", help="Use expanding-window walk-forward validation (--folds contiguous out-of-time test chunks) instead of a single train/val/test split")
    parser.add_argument("--folds", type=int, default=4, help="Number of walk-forward folds (only used with --walk-forward)")
    parser.add_argument("--eval-fraction", type=float, default=0.30, help="Fraction of each symbol's sequence reserved as the walk-forward evaluation region, split into --folds contiguous test chunks (only used with --walk-forward; with the defaults, fold 0's train/val boundary matches the single-split train_ratio=0.70 default)")
    args = parser.parse_args()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/benchmark_summary_{stamp}.csv"

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.walk_forward:
        print(f"Walk-forward mode: {args.folds} expanding-window folds, eval_fraction={args.eval_fraction}")
    print(f"Benchmarking {len(symbols)} symbols x 3 losses: {symbols}\n")

    loss_types = ("mse", "profit-aware", "profit-log")

    rows: list[dict] = []
    t_start = time.time()

    for i, symbol in enumerate(symbols, 1):
        for loss_type in loss_types:
            t0 = time.time()
            try:
                if args.walk_forward:
                    fold_metrics = run_one_walk_forward(symbol, loss_type, args, device, seed=args.seed)
                else:
                    fold_metrics = [run_one(symbol, loss_type, args, device, seed=args.seed)]
            except Exception as e:
                print(f"[{i}/{len(symbols)}] {symbol:10s} {loss_type:13s} FAILED: {e}")
                continue
            dt = time.time() - t0
            rows.extend(fold_metrics)
            for m in fold_metrics:
                deg_tag = " [DEGENERATE: constant signal]" if m["degenerate"] else ""
                fold_tag = f" fold={m['fold']}" if args.walk_forward else ""
                print(
                    f"[{i}/{len(symbols)}] {symbol:10s} ({m['tier']:<6s}) {loss_type:13s}{fold_tag} "
                    f"dir_acc={m['directional_acc']:.3f} cum_ret={m['cum_profit']:+.4f} "
                    f"geo_ret={m['cum_profit_geo']:+.4f} sharpe={m['sharpe_like']:+.3f} "
                    f"(buy&hold={m['buy_hold']:+.4f} always_short={m['always_short']:+.4f}){deg_tag}"
                )
            fold_note = f" / {len(fold_metrics)} folds" if args.walk_forward else ""
            print(f"  [{dt:.1f}s total{fold_note}]")

    total_dt = time.time() - t_start

    summary_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        summary_lines.append(line)

    emit(f"\nTotal benchmark time: {total_dt:.1f}s\n")

    # --- Aggregate summary -------------------------------------------------
    emit(f"{'='*78}")
    emit("  AGGREGATE SUMMARY (mean across symbols x folds)" if args.walk_forward else "  AGGREGATE SUMMARY (mean across symbols)")
    emit(f"{'='*78}")
    header = f"  {'Loss':<15}{'DirAcc':>10}{'CumRet':>12}{'GeoRet':>12}{'Sharpe':>10}{'RMSE':>10}"
    emit(header)
    emit(f"  {'-'*74}")
    for loss_type in loss_types:
        sub = [r for r in rows if r["loss_type"] == loss_type]
        if not sub:
            continue
        dir_acc = np.mean([r["directional_acc"] for r in sub])
        cum_ret = np.mean([r["cum_profit"] for r in sub])
        geo_ret = np.mean([r["cum_profit_geo"] for r in sub])
        sharpe = np.mean([r["sharpe_like"] for r in sub])
        rmse = np.mean([r["rmse"] for r in sub])
        emit(f"  {loss_type:<15}{dir_acc:>10.4f}{cum_ret:>+12.4f}{geo_ret:>+12.4f}{sharpe:>+10.3f}{rmse:>10.5f}")

    bh = np.mean([r["buy_hold"] for r in rows if r["loss_type"] == "mse"]) if rows else float("nan")
    emit(f"  {'buy & hold':<15}{'':>10}{bh:>+12.4f}")

    # --- Per-tier breakdown ---------------------------------------------------
    emit(f"\n{'='*78}")
    emit("  BY TIER (mean across symbols in tier)")
    emit(f"{'='*78}")
    emit(header)
    emit(f"  {'-'*74}")
    for tier in ("strong", "mid", "weak", "index"):
        for loss_type in loss_types:
            sub = [r for r in rows if r["tier"] == tier and r["loss_type"] == loss_type]
            if not sub:
                continue
            dir_acc = np.mean([r["directional_acc"] for r in sub])
            cum_ret = np.mean([r["cum_profit"] for r in sub])
            geo_ret = np.mean([r["cum_profit_geo"] for r in sub])
            sharpe = np.mean([r["sharpe_like"] for r in sub])
            rmse = np.mean([r["rmse"] for r in sub])
            label = f"{tier}/{loss_type}"
            emit(f"  {label:<15}{dir_acc:>10.4f}{cum_ret:>+12.4f}{geo_ret:>+12.4f}{sharpe:>+10.3f}{rmse:>10.5f}")
        emit()

    # --- Win-rate: profit-aware/profit-log vs mse -----------------------------
    # by_symbol groups a symbol's rows by loss_type (1 row/loss in legacy mode,
    # `folds` rows/loss in walk-forward mode); by_symbol_fold groups by
    # (symbol, fold) so a walk-forward run can also report a pooled,
    # fold-level win-rate rather than only a per-symbol mean.
    by_symbol: dict[str, dict[str, list[dict]]] = {}
    by_symbol_fold: dict[tuple[str, int], dict[str, dict]] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], {}).setdefault(r["loss_type"], []).append(r)
        by_symbol_fold.setdefault((r["symbol"], r["fold"]), {})[r["loss_type"]] = r

    def sym_mean(symbol: str, loss_type: str, key: str) -> float:
        return float(np.mean([r[key] for r in by_symbol[symbol][loss_type]]))

    for loss_type in loss_types:
        if loss_type == "mse":
            continue

        if args.walk_forward:
            wins_geo = wins_sharpe = wins_diracc = n_pairs = 0
            for d in by_symbol_fold.values():
                if "mse" not in d or loss_type not in d:
                    continue
                n_pairs += 1
                wins_geo += d[loss_type]["cum_profit_geo"] > d["mse"]["cum_profit_geo"]
                wins_sharpe += d[loss_type]["sharpe_like"] > d["mse"]["sharpe_like"]
                wins_diracc += d[loss_type]["directional_acc"] > d["mse"]["directional_acc"]
            emit(f"\n  {loss_type} beats MSE on (out of {n_pairs} symbol-fold runs, pooled):")
            emit(f"    Geometric return: {wins_geo}/{n_pairs}")
            emit(f"    Sharpe-like:      {wins_sharpe}/{n_pairs}")
            emit(f"    Directional acc:  {wins_diracc}/{n_pairs}")

        wins_geo = wins_sharpe = wins_diracc = n_pairs = 0
        for symbol in by_symbol:
            if "mse" not in by_symbol[symbol] or loss_type not in by_symbol[symbol]:
                continue
            n_pairs += 1
            wins_geo += sym_mean(symbol, loss_type, "cum_profit_geo") > sym_mean(symbol, "mse", "cum_profit_geo")
            wins_sharpe += sym_mean(symbol, loss_type, "sharpe_like") > sym_mean(symbol, "mse", "sharpe_like")
            wins_diracc += sym_mean(symbol, loss_type, "directional_acc") > sym_mean(symbol, "mse", "directional_acc")

        label = "per-symbol mean across folds" if args.walk_forward else "symbols"
        emit(f"\n  {loss_type} beats MSE on (out of {n_pairs} {label}):")
        emit(f"    Geometric return: {wins_geo}/{n_pairs}")
        emit(f"    Sharpe-like:      {wins_sharpe}/{n_pairs}")
        emit(f"    Directional acc:  {wins_diracc}/{n_pairs}")

    # --- Fold stability: how consistent is geo-return across a symbol's folds -
    if args.walk_forward and args.folds > 1:
        emit(f"\n{'='*78}")
        emit("  FOLD STABILITY (std-dev of geometric return across a symbol's folds)")
        emit("  Lower = more consistent across time regimes, not just a lucky window")
        emit(f"{'='*78}")
        emit(f"  {'Symbol':<10}" + "".join(f"{lt:>15}" for lt in loss_types))
        emit(f"  {'-'*10}" + "".join(f"{'-'*15}" for _ in loss_types))
        per_loss_stds: dict[str, list[float]] = {lt: [] for lt in loss_types}
        for symbol in sorted(by_symbol):
            row_vals = []
            complete = True
            for lt in loss_types:
                group = by_symbol[symbol].get(lt, [])
                if len(group) < 2:
                    complete = False
                    break
                row_vals.append(float(np.std([r["cum_profit_geo"] for r in group], ddof=1)))
            if not complete:
                continue
            for lt, v in zip(loss_types, row_vals):
                per_loss_stds[lt].append(v)
            emit(f"  {symbol:<10}" + "".join(f"{v:>15.4f}" for v in row_vals))
        emit(f"  {'-'*10}" + "".join(f"{'-'*15}" for _ in loss_types))
        emit(
            f"  {'mean':<10}"
            + "".join(
                f"{np.mean(per_loss_stds[lt]):>15.4f}" if per_loss_stds[lt] else f"{'n/a':>15}"
                for lt in loss_types
            )
        )

    # --- Degenerate (constant-signal) run count -------------------------------
    emit(f"\n{'='*78}")
    emit("  DEGENERATE RUNS (executed signal never changes across the test window)")
    emit(f"{'='*78}")
    for loss_type in loss_types:
        sub = [r for r in rows if r["loss_type"] == loss_type]
        n_deg = sum(1 for r in sub if r["degenerate"])
        deg_symbols = [r["symbol"] for r in sub if r["degenerate"]]
        emit(f"  {loss_type:<15}{n_deg}/{len(sub)}  {deg_symbols}")

    # --- Save per-symbol CSV --------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("symbol,tier,loss_type,fold,n_test,directional_acc,cum_profit,cum_profit_geo,sharpe_like,mse,mae,rmse,buy_hold,always_short,degenerate\n")
        for r in rows:
            f.write(
                f"{r['symbol']},{r['tier']},{r['loss_type']},{r['fold']},{r['n_test']},{r['directional_acc']:.6f},"
                f"{r['cum_profit']:.6f},{r['cum_profit_geo']:.6f},{r['sharpe_like']:.6f},"
                f"{r['mse']:.8f},{r['mae']:.8f},{r['rmse']:.8f},{r['buy_hold']:.6f},"
                f"{r['always_short']:.6f},{r['degenerate']}\n"
            )
    print(f"\nSaved per-symbol results to {out_path}")

    # --- Save aggregate/tier/win-rate summary ---------------------------------
    summary_path = out_path.with_name(out_path.stem + "_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved aggregate summary to {summary_path}")


if __name__ == "__main__":
    main()
