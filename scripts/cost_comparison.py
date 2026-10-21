"""With-cost vs without-cost comparison of the GRU trading conditions.

Per symbol, trains three GRUs on the same split/seed:

* ``mse``            -- MSE baseline (cost-agnostic training)
* ``pa_nocost``      -- profit-aware loss trained with transaction cost = 0
* ``pa_cost``        -- profit-aware loss trained with the position-change cost
                        ``c * |signal_t - signal_{t-1}|`` inside the loss

Each model's test-window signal is then re-simulated under BOTH cost
assumptions (0 and ``--transaction-cost``), so every condition is scored under
the same cost. "Without cost" = evaluated at cost 0; "with cost" = evaluated at
cost ``c``. ``pa_nocost`` evaluated with cost isolates what putting the cost
into the loss actually buys.

Usage:
    .venv/Scripts/python.exe scripts/cost_comparison.py
    .venv/Scripts/python.exe scripts/cost_comparison.py --symbols AAPL,MSFT --epochs 30
    .venv/Scripts/python.exe scripts/cost_comparison.py --cost 0.002
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.gru_model import GRUReturnPredictor
from training.train import TrainingConfig, fit, run_epoch
from utils.metrics import cumulative_profit, cumulative_profit_geometric, sharpe_like
from utils.pipeline import prepare_data
from utils.trading import calibrate_alpha

DEFAULT_SYMBOLS = ["NVDA", "AAPL", "MSFT", "KO", "XOM", "F"]
CONDITIONS = {
    "mse": "MSE baseline",
    "pa_nocost": "Profit-aware (no cost in loss)",
    "pa_cost": "Profit-aware (cost in loss)",
}
COLORS = {"mse": "#4c78a8", "pa_nocost": "#f58518", "pa_cost": "#54a24b"}


def train_and_test(loss_type: str, train_cost: float, data, args, device, alpha: float) -> dict:
    split = data.split
    output_scale = (args.output_cap_std / alpha) if args.output_cap_std > 0 else None
    torch.manual_seed(args.seed)
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
        transaction_cost_rate=train_cost,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    fit(model, data.train_loader, data.val_loader, config, device, capital=args.capital)
    return run_epoch(model, data.test_loader, None, alpha, train_cost, device, loss_type=loss_type)


def simulate(signal: np.ndarray, actual: np.ndarray, cost: float) -> dict:
    prev = np.concatenate(([0.0], signal[:-1]))
    net = signal * actual - cost * np.abs(signal - prev)
    return {
        "cum_profit": cumulative_profit(signal, actual, cost),
        "geo_return": cumulative_profit_geometric(signal, actual, cost),
        "sharpe": sharpe_like(net),
        "turnover": float(np.abs(signal - prev).mean()),
    }


def plot(df: pd.DataFrame, cost: float, out: Path) -> None:
    symbols = list(df.symbol.unique()) + ["MEAN"]
    metrics = [
        ("geo_return", f"Geometric return, evaluated WITHOUT cost", 0.0),
        ("geo_return", f"Geometric return, evaluated WITH cost ({cost:.2%}/unit)", cost),
        ("turnover", "Mean daily position change", cost),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    width = 0.27
    x = np.arange(len(symbols))
    for ax, (metric, title, eval_cost) in zip(axes, metrics):
        sub = df[np.isclose(df.eval_cost, eval_cost)]
        for k, cond in enumerate(CONDITIONS):
            vals = [sub[(sub.symbol == s) & (sub.condition == cond)][metric].iloc[0] for s in symbols[:-1]]
            vals.append(float(np.mean(vals)))
            ax.bar(x + (k - 1) * width, vals, width, label=CONDITIONS[cond], color=COLORS[cond])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(symbols)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("GRU conditions: without vs with position-change cost (same cost applied to all)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description="With-cost vs without-cost GRU comparison")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--loss-lambda", type=float, default=0.7)
    p.add_argument("--cost", "--transaction-cost", dest="cost", type=float, default=0.001, help="Position-change cost rate used for the 'with cost' condition")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--output-cap-std", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="results/cost_comparison")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    rows: list[dict] = []
    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        try:
            data = prepare_data(symbol, args.start, args.end, args.sequence_length, args.batch_size)
            alpha = args.alpha if args.alpha is not None else calibrate_alpha(data.split.y_train)
            runs = {
                "mse": train_and_test("mse", 0.0, data, args, device, alpha),
                "pa_nocost": train_and_test("profit-aware", 0.0, data, args, device, alpha),
                "pa_cost": train_and_test("profit-aware", args.cost, data, args, device, alpha),
            }
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol} FAILED: {e}")
            continue
        for cond, m in runs.items():
            for eval_cost in (0.0, args.cost):
                sim = simulate(m["signal_np"], m["actual_np"], eval_cost)
                rows.append({
                    "symbol": symbol, "condition": cond, "eval_cost": eval_cost,
                    "directional_acc": m["directional_acc"], "mse": m["mse"],
                    **sim, "final_balance": args.capital * (1 + sim["geo_return"]),
                })
        print(f"[{i}/{len(symbols)}] {symbol} done in {time.time() - t0:.0f}s")

    if not rows:
        raise SystemExit("no successful runs")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)

    lines: list[str] = []
    for label, ec in (("WITHOUT cost (eval cost = 0)", 0.0), (f"WITH cost (eval cost = {args.cost})", args.cost)):
        sub = df[np.isclose(df.eval_cost, ec)]
        lines += [f"\n=== {label} ===",
                  f"{'condition':<32}{'GeoRet':>10}{'AddRet':>10}{'Sharpe':>9}{'Turnover':>10}{'Final PHP':>12}"]
        for cond, name in CONDITIONS.items():
            c = sub[sub.condition == cond]
            lines.append(f"{name:<32}{c.geo_return.mean():>+10.4f}{c.cum_profit.mean():>+10.4f}"
                         f"{c.sharpe.mean():>+9.3f}{c.turnover.mean():>10.4f}{c.final_balance.mean():>12,.0f}")
    drag = df[np.isclose(df.eval_cost, 0.0)].set_index(["symbol", "condition"]).geo_return - \
        df[np.isclose(df.eval_cost, args.cost)].set_index(["symbol", "condition"]).geo_return
    lines += ["\n=== Cost drag on geometric return (no-cost minus with-cost), mean over symbols ==="]
    for cond, name in CONDITIONS.items():
        lines.append(f"{name:<32}{drag.xs(cond, level='condition').mean():>+10.4f}")
    wc = df[np.isclose(df.eval_cost, args.cost)].pivot(index="symbol", columns="condition", values="geo_return")
    lines.append(f"\nWith cost: pa_cost beats pa_nocost on {(wc.pa_cost > wc.pa_nocost).sum()}/{len(wc)} symbols, "
                 f"beats MSE on {(wc.pa_cost > wc.mse).sum()}/{len(wc)}")
    text = "\n".join(lines)
    print(text)
    out.with_name(out.name + "_summary.txt").write_text(text + "\n", encoding="utf-8")

    plot(df, args.cost, out.with_suffix(".png"))
    print(f"\nwrote {out}.csv, {out}_summary.txt, {out}.png")


if __name__ == "__main__":
    main()
