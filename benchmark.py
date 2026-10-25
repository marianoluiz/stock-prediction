"""Multi-stock benchmark: train MSE and profit-aware models across many
tickers and report aggregate metrics, so a single-stock result can't be
mistaken for a general improvement.

Uses the exact same data pipeline, model, and training loop as main.py
(build_feature_frame's lagged-return + volume-zscore channels included) —
this script only adds the loop over symbols and the aggregate summary.
Skips torch.save/plotting per run to stay fast; see main.py for a single
stock's full artifacts (weights + curves).

The universe is the Philippine market only: the PSEi constituents
(PSEI_STOCKS), reported as one group, plus the PSEi itself (INDEX_SYMBOL) as a
separate passive market reference that is not averaged in with the stocks.
Trading costs default to the PSE's asymmetric buy (0.296%) / sell (0.396%)
rates.

Usage:
    python benchmark.py
    python benchmark.py --charts
    python benchmark.py --symbols PSE:ICT,PSE:BDO --epochs 30
    python benchmark.py --from-csv results/benchmark_summary_ph_<time>.csv --charts   # re-summarize, no training
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
from utils.baselines import evaluate_price_strategies
from utils.pipeline import cache_path_for, prepare_data, prepare_walk_forward
from utils.preprocessing import load_stock_data
from utils.trading import PH_BUY_COST, PH_SELL_COST, calibrate_alpha

STRATEGY_NAMES = ("sma_crossover", "rsi", "macd")

# PSEi constituents as of the 3 August 2026 rebalancing (Maynilad in,
# Converge out), excluding MYNLD: listed November 2025, so it has ~190 usable
# windows -- too few for a 70/15/15 split. Tickers use tvDatafeed's
# "EXCHANGE:TICKER" convention (routed off yfinance in
# utils.preprocessing.load_stock_data) since yfinance doesn't carry the PSE.
PSEI_STOCKS = [
    "PSE:AC", "PSE:ACEN", "PSE:AEV", "PSE:ALI", "PSE:AREIT", "PSE:BDO", "PSE:BPI", "PSE:CBC",
    "PSE:CNPF", "PSE:DMC", "PSE:EMI", "PSE:GLO", "PSE:GTCAP", "PSE:ICT", "PSE:JFC", "PSE:JGS",
    "PSE:LTG", "PSE:MBT", "PSE:MER", "PSE:MONDE", "PSE:PGOLD", "PSE:PLUS", "PSE:RCR", "PSE:SCC",
    "PSE:SM", "PSE:SMC", "PSE:SMPH", "PSE:TEL", "PSE:URC",
]
# The index itself: a passive market reference, reported separately and never
# averaged in with the stocks.
INDEX_SYMBOL = "PSE:PSEI"

DEFAULT_SYMBOLS = [*PSEI_STOCKS, INDEX_SYMBOL]
LOSS_TYPES = ("mse", "profit-aware")


def group_of(symbol: str) -> str:
    """``"index"`` for the PSEi reference, ``"stock"`` for everything else."""
    return "index" if symbol == INDEX_SYMBOL else "stock"


def buy_and_hold_return(actual_returns: np.ndarray) -> float:
    """Cumulative return of a naive always-long position over the test window."""
    return float(np.prod(1.0 + actual_returns) - 1.0)


def always_short_return(actual_returns: np.ndarray) -> float:
    """Cumulative return of a naive always-short position over the test window."""
    return float(np.prod(1.0 - actual_returns) - 1.0)


def is_degenerate_signal(signal: np.ndarray) -> bool:
    """True if the executed signal never crosses zero across the whole test window.

    The signal is now the continuous tanh(alpha*pred) position, so exact-value
    repeats are essentially impossible -- what matters is whether the model
    ever changes its *side* of the market. A model that stays the same sign
    (or sits exactly at 0) the entire window learned no conditional
    (day-to-day) signal at all -- it's indistinguishable from a naive
    always-long/always-short/never-traded baseline, whatever its profit
    looks like.
    """
    signs = np.sign(signal)
    nonzero_signs = signs[signs != 0]
    if nonzero_signs.size == 0:
        return True
    return bool(np.unique(nonzero_signs).size == 1)


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
        buy_cost_rate=args.buy_cost,
        sell_cost_rate=args.sell_cost,
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
        model, test_loader, None, alpha, args.buy_cost, args.sell_cost, device,
        loss_type=loss_type,
    )
    test_metrics["symbol"] = symbol
    test_metrics["loss_type"] = loss_type
    test_metrics["group"] = group_of(symbol)
    test_metrics["fold"] = fold
    test_metrics["n_test"] = len(split.x_test)
    test_metrics["buy_hold"] = buy_and_hold_return(test_metrics["actual_np"])
    test_metrics["always_short"] = always_short_return(test_metrics["actual_np"])
    test_metrics["degenerate"] = is_degenerate_signal(test_metrics["signal_np"])

    price_df = load_stock_data(symbol, args.start, args.end, cache_path_for(symbol, args.start, args.end))
    strategies = evaluate_price_strategies(
        price_df["Close"], split.dates_test, test_metrics["actual_np"],
        buy_cost_rate=args.buy_cost, sell_cost_rate=args.sell_cost,
    )
    for name, m in strategies.items():
        test_metrics[f"{name}_dir_acc"] = m["directional_acc"]
        test_metrics[f"{name}_cum_ret"] = m["cum_profit"]
        test_metrics[f"{name}_geo_ret"] = m["cum_profit_geo"]
        test_metrics[f"{name}_sharpe"] = m["sharpe_like"]

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


def _mean_row(label: str, sub: list[dict], keys: tuple[str, ...], with_rmse: bool) -> str:
    dir_acc, cum_ret, geo_ret, sharpe = (np.mean([r[k] for r in sub]) for k in keys)
    line = f"  {label:<22}{dir_acc:>10.4f}{cum_ret:>+12.4f}{geo_ret:>+12.4f}{sharpe:>+10.3f}"
    if with_rmse:
        line += f"{np.mean([r['rmse'] for r in sub]):>10.5f}"
    return line


def summarize(
    rows: list[dict],
    buy_cost: float,
    sell_cost: float,
    loss_lambda: float,
    walk_forward: bool = False,
    total_dt: float | None = None,
) -> list[str]:
    """Build the text summary: the stocks as one group, the PSEi as a separate
    reference, win counts, strategy baselines, fold stability and degenerate runs.
    """
    lines: list[str] = []
    emit = lines.append
    rule = "=" * 85
    header = f"  {'Model':<22}{'DirAcc':>10}{'CumRet':>12}{'GeoRet':>12}{'Sharpe':>10}{'RMSE':>10}"
    model_keys = ("directional_acc", "cum_profit", "cum_profit_geo", "sharpe_like")

    if total_dt is not None:
        emit(f"Total benchmark time: {total_dt:.1f}s")
    emit(f"Trading costs: buy {buy_cost:.3%}, sell {sell_cost:.3%}; loss lambda {loss_lambda}")
    emit("")

    stock_rows = [r for r in rows if r["group"] == "stock"]
    index_rows = [r for r in rows if r["group"] == "index"]
    n_stocks = len({r["symbol"] for r in stock_rows})
    per = "symbols x folds" if walk_forward else "stocks"

    # --- Stocks, one group ---------------------------------------------------
    emit(rule)
    emit(f"  PSEi CONSTITUENTS (mean across {n_stocks} {per})")
    emit(rule)
    emit(header)
    emit(f"  {'-' * 81}")
    for loss_type in LOSS_TYPES:
        sub = [r for r in stock_rows if r["loss_type"] == loss_type]
        if sub:
            emit(_mean_row(loss_type, sub, model_keys, with_rmse=True))
    # Reference columns (buy & hold, strategies) don't depend on loss_type, so
    # take them from one loss_type's rows to count each stock once.
    ref = [r for r in stock_rows if r["loss_type"] == "mse"]
    if ref:
        emit(f"  {'buy & hold':<22}{'':>10}{np.mean([r['buy_hold'] for r in ref]):>+12.4f}")
        for name in STRATEGY_NAMES:
            keys = (f"{name}_dir_acc", f"{name}_cum_ret", f"{name}_geo_ret", f"{name}_sharpe")
            emit(_mean_row(name, ref, keys, with_rmse=False))

    # --- Index reference -------------------------------------------------------
    if index_rows:
        emit("")
        emit(rule)
        emit(f"  PSEi INDEX ({INDEX_SYMBOL}, passive market reference, not in the mean above)")
        emit(rule)
        emit(header)
        emit(f"  {'-' * 81}")
        for loss_type in LOSS_TYPES:
            sub = [r for r in index_rows if r["loss_type"] == loss_type]
            if sub:
                emit(_mean_row(loss_type, sub, model_keys, with_rmse=True))
        ref_idx = [r for r in index_rows if r["loss_type"] == "mse"]
        if ref_idx:
            emit(f"  {'buy & hold':<22}{'':>10}{np.mean([r['buy_hold'] for r in ref_idx]):>+12.4f}")

    # --- Win counts: profit-aware vs MSE, stocks only ---------------------------
    by_symbol: dict[str, dict[str, list[dict]]] = {}
    by_symbol_fold: dict[tuple[str, int], dict[str, dict]] = {}
    for r in stock_rows:
        by_symbol.setdefault(r["symbol"], {}).setdefault(r["loss_type"], []).append(r)
        by_symbol_fold.setdefault((r["symbol"], r["fold"]), {})[r["loss_type"]] = r

    def sym_mean(symbol: str, loss_type: str, key: str) -> float:
        return float(np.mean([r[key] for r in by_symbol[symbol][loss_type]]))

    def emit_wins(title: str, pairs: list[tuple]) -> None:
        n = len(pairs)
        emit("")
        emit(f"  profit-aware beats MSE on (out of {n} {title}):")
        for label, key in (("Geometric return", "cum_profit_geo"), ("Sharpe-like", "sharpe_like"),
                           ("Directional acc", "directional_acc")):
            wins = sum(pa(key) > mse(key) for pa, mse in pairs)
            emit(f"    {label + ':':<18}{wins}/{n}")

    if walk_forward:
        emit_wins("stock-fold runs, pooled", [
            ((lambda k, d=d: d["profit-aware"][k]), (lambda k, d=d: d["mse"][k]))
            for d in by_symbol_fold.values() if "mse" in d and "profit-aware" in d
        ])
    emit_wins("stocks, mean across folds" if walk_forward else "stocks", [
        ((lambda k, s=s: sym_mean(s, "profit-aware", k)), (lambda k, s=s: sym_mean(s, "mse", k)))
        for s, d in by_symbol.items() if "mse" in d and "profit-aware" in d
    ])

    # --- Profit-aware vs classic TA strategy baselines, stocks only -------------
    pa_rows = [r for r in stock_rows if r["loss_type"] == "profit-aware"]
    if pa_rows:
        emit("")
        emit(rule)
        emit(f"  PROFIT-AWARE vs CLASSIC STRATEGY BASELINES (out of {len(pa_rows)} stock runs)")
        emit(rule)
        n = len(pa_rows)
        for name in STRATEGY_NAMES:
            wins_geo = sum(r["cum_profit_geo"] > r[f"{name}_geo_ret"] for r in pa_rows)
            wins_sharpe = sum(r["sharpe_like"] > r[f"{name}_sharpe"] for r in pa_rows)
            emit(f"  profit-aware beats {name:<15} geo_ret: {wins_geo}/{n}   sharpe: {wins_sharpe}/{n}")

    # --- Fold stability: how consistent is geo-return across a symbol's folds -
    if walk_forward:
        emit("")
        emit(rule)
        emit("  FOLD STABILITY (std-dev of geometric return across a stock's folds)")
        emit("  Lower = more consistent across time regimes, not just a lucky window")
        emit(rule)
        emit(f"  {'Symbol':<10}" + "".join(f"{lt:>15}" for lt in LOSS_TYPES))
        emit(f"  {'-' * 10}" + "".join(f"{'-' * 15}" for _ in LOSS_TYPES))
        per_loss_stds: dict[str, list[float]] = {lt: [] for lt in LOSS_TYPES}
        for symbol in sorted(by_symbol):
            groups = [by_symbol[symbol].get(lt, []) for lt in LOSS_TYPES]
            if any(len(g) < 2 for g in groups):
                continue
            vals = [float(np.std([r["cum_profit_geo"] for r in g], ddof=1)) for g in groups]
            for lt, v in zip(LOSS_TYPES, vals):
                per_loss_stds[lt].append(v)
            emit(f"  {symbol:<10}" + "".join(f"{v:>15.4f}" for v in vals))
        emit(f"  {'-' * 10}" + "".join(f"{'-' * 15}" for _ in LOSS_TYPES))
        emit(f"  {'mean':<10}" + "".join(
            f"{np.mean(per_loss_stds[lt]):>15.4f}" if per_loss_stds[lt] else f"{'n/a':>15}" for lt in LOSS_TYPES
        ))

    # --- Degenerate (constant-signal) runs, every symbol ------------------------
    emit("")
    emit(rule)
    emit("  DEGENERATE RUNS (executed signal never changes side across the test window)")
    emit(rule)
    for loss_type in LOSS_TYPES:
        sub = [r for r in rows if r["loss_type"] == loss_type]
        deg_symbols = [r["symbol"] for r in sub if r["degenerate"]]
        emit(f"  {loss_type:<15}{len(deg_symbols)}/{len(sub)}  {deg_symbols}")
    return lines


def write_csv(rows: list[dict], out_path: Path, buy_cost: float, sell_cost: float, loss_lambda: float) -> None:
    strategy_cols = ",".join(f"{name}_dir_acc,{name}_cum_ret,{name}_geo_ret,{name}_sharpe" for name in STRATEGY_NAMES)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "symbol,group,loss_type,fold,n_test,directional_acc,cum_profit,cum_profit_geo,sharpe_like,"
            f"mse,mae,rmse,buy_hold,always_short,degenerate,{strategy_cols},buy_cost,sell_cost,loss_lambda\n"
        )
        for r in rows:
            strategy_vals = ",".join(
                f"{r[f'{name}_dir_acc']:.6f},{r[f'{name}_cum_ret']:.6f},"
                f"{r[f'{name}_geo_ret']:.6f},{r[f'{name}_sharpe']:.6f}"
                for name in STRATEGY_NAMES
            )
            f.write(
                f"{r['symbol']},{r['group']},{r['loss_type']},{r['fold']},{r['n_test']},{r['directional_acc']:.6f},"
                f"{r['cum_profit']:.6f},{r['cum_profit_geo']:.6f},{r['sharpe_like']:.6f},"
                f"{r['mse']:.8f},{r['mae']:.8f},{r['rmse']:.8f},{r['buy_hold']:.6f},"
                f"{r['always_short']:.6f},{r['degenerate']},{strategy_vals},"
                f"{buy_cost},{sell_cost},{loss_lambda}\n"
            )


def rows_from_csv(path: Path) -> list[dict]:
    """Per-run rows back from a benchmark CSV (older CSVs with a ``tier`` column included)."""
    import pandas as pd

    df = pd.read_csv(path)
    df["group"] = df["symbol"].map(group_of)
    return df.drop(columns=["tier"], errors="ignore").to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MSE vs profit-aware GRU across the PSEi")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated EXCHANGE:TICKER list (default: PSEI_STOCKS plus the PSEi index)")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--loss-lambda", type=float, default=0.7)
    parser.add_argument("--buy-cost", type=float, default=PH_BUY_COST, help="Cost rate per unit of position increase (purchase); default 0.296%%")
    parser.add_argument("--sell-cost", type=float, default=PH_SELL_COST, help="Cost rate per unit of position decrease (sale, incl. 0.1%% stock transaction tax); default 0.396%%")
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
    parser.add_argument("--charts", action="store_true", help="Also save the MSE-vs-profit-aware comparison charts into <output stem>_charts/")
    parser.add_argument("--from-csv", type=str, default=None, help="Skip training: rebuild the _summary.txt (and charts, with --charts) from an existing benchmark CSV")
    args = parser.parse_args()

    if args.from_csv:
        out_path = Path(args.from_csv)
        rows = rows_from_csv(out_path)
        first = rows[0]
        buy_cost = first.get("buy_cost", args.buy_cost)
        sell_cost = first.get("sell_cost", args.sell_cost)
        loss_lambda = first.get("loss_lambda", args.loss_lambda)
        if "buy_cost" not in first:
            print(f"Note: {out_path.name} does not record its costs/lambda; assuming the current defaults.")
        summary_lines = summarize(rows, buy_cost, sell_cost, loss_lambda,
                                  walk_forward=max(r["fold"] for r in rows) > 0)
        print("\n".join(summary_lines))
    else:
        buy_cost, sell_cost, loss_lambda = args.buy_cost, args.sell_cost, args.loss_lambda
        if args.symbols is not None:
            symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
            market_tag = "custom"
        else:
            symbols = DEFAULT_SYMBOLS
            market_tag = "ph"

        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"results/benchmark_summary_{market_tag}_{stamp}.csv"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if args.walk_forward:
            print(f"Walk-forward mode: {args.folds} expanding-window folds, eval_fraction={args.eval_fraction}")
        print(f"Trading costs: buy {args.buy_cost:.3%}, sell {args.sell_cost:.3%} per unit of position change")
        print(f"Benchmarking {len(symbols)} symbols x {len(LOSS_TYPES)} losses: {symbols}\n")

        rows: list[dict] = []
        t_start = time.time()
        for i, symbol in enumerate(symbols, 1):
            for loss_type in LOSS_TYPES:
                t0 = time.time()
                try:
                    if args.walk_forward:
                        fold_metrics = run_one_walk_forward(symbol, loss_type, args, device, seed=args.seed)
                    else:
                        fold_metrics = [run_one(symbol, loss_type, args, device, seed=args.seed)]
                except Exception as e:
                    print(f"[{i}/{len(symbols)}] {symbol:10s} {loss_type:13s} FAILED: {e}")
                    continue
                rows.extend(fold_metrics)
                for m in fold_metrics:
                    deg_tag = " [DEGENERATE: constant signal]" if m["degenerate"] else ""
                    fold_tag = f" fold={m['fold']}" if args.walk_forward else ""
                    print(
                        f"[{i}/{len(symbols)}] {symbol:10s} ({m['group']:<5s}) {loss_type:13s}{fold_tag} "
                        f"dir_acc={m['directional_acc']:.3f} cum_ret={m['cum_profit']:+.4f} "
                        f"geo_ret={m['cum_profit_geo']:+.4f} sharpe={m['sharpe_like']:+.3f} "
                        f"(buy&hold={m['buy_hold']:+.4f} always_short={m['always_short']:+.4f}){deg_tag}"
                    )
                fold_note = f" / {len(fold_metrics)} folds" if args.walk_forward else ""
                print(f"  [{time.time() - t0:.1f}s total{fold_note}]")

        summary_lines = summarize(rows, args.buy_cost, args.sell_cost, args.loss_lambda,
                                  walk_forward=args.walk_forward, total_dt=time.time() - t_start)
        print("\n" + "\n".join(summary_lines))

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out_path, args.buy_cost, args.sell_cost, args.loss_lambda)
        print(f"\nSaved per-symbol results to {out_path}")

    summary_path = out_path.with_name(out_path.stem + "_summary.txt")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")

    if args.charts:
        import pandas as pd
        from scripts.plot_loss_comparison import plot_comparisons

        subtitle = (f"PSEi constituents, held-out test window; costs: buy {buy_cost:.3%}, "
                    f"sell {sell_cost:.3%}; loss lambda = {loss_lambda:g}")
        plot_comparisons(pd.read_csv(out_path), out_path.parent / f"{out_path.stem}_charts", subtitle=subtitle)


if __name__ == "__main__":
    main()
