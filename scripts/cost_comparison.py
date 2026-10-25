"""With-cost vs without-cost comparison of the GRU trading conditions.

Per symbol, trains three GRUs on the same split/seed:

* ``mse``            -- MSE baseline (cost-agnostic training)
* ``pa_nocost``      -- profit-aware loss trained with zero trading cost
* ``pa_cost``        -- profit-aware loss trained with the PSE's asymmetric
                        position-change cost inside the loss:
                        ``c_buy * max(delta, 0) + c_sell * max(-delta, 0)``

Each model's test-window signal is then re-simulated under BOTH cost
assumptions (zero, and ``--buy-cost`` / ``--sell-cost``), so every condition is
scored under the same cost. "Without cost" = evaluated at zero cost; "with
cost" = evaluated at the buy/sell rates. ``pa_nocost`` evaluated with cost
isolates what putting the cost into the loss actually buys.

Usage:
    .venv/Scripts/python.exe scripts/cost_comparison.py
    .venv/Scripts/python.exe scripts/cost_comparison.py --symbols PSE:ICT,PSE:BDO --epochs 30
    .venv/Scripts/python.exe scripts/cost_comparison.py --buy-cost 0.00296 --sell-cost 0.00396
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
from utils.chart_style import (
    COLOR_MSE,
    COLOR_PA,
    COLOR_THIRD,
    apply_style,
    footnote,
    label_bars,
    percent_axis,
    save,
    zero_line,
)
from utils.metrics import cumulative_profit, cumulative_profit_geometric, sharpe_like
from utils.pipeline import prepare_data
from utils.trading import PH_BUY_COST, PH_SELL_COST, calibrate_alpha, position_change_cost

# Six PSEi constituents (a subset of benchmark.PSEI_STOCKS) to keep the run short.
DEFAULT_SYMBOLS = ["PSE:ICT", "PSE:CNPF", "PSE:BDO", "PSE:TEL", "PSE:JFC", "PSE:ALI"]
CONDITIONS = {
    "mse": "MSE baseline",
    "pa_nocost": "Profit-aware, cost not in loss",
    "pa_cost": "Profit-aware, cost in loss",
}
COLORS = {"mse": COLOR_MSE, "pa_nocost": COLOR_PA, "pa_cost": COLOR_THIRD}


def train_and_test(loss_type: str, buy_cost: float, sell_cost: float, data, args, device, alpha: float) -> dict:
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
        buy_cost_rate=buy_cost,
        sell_cost_rate=sell_cost,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    fit(model, data.train_loader, data.val_loader, config, device, capital=args.capital)
    return run_epoch(model, data.test_loader, None, alpha, buy_cost, sell_cost, device, loss_type=loss_type)


def simulate(signal: np.ndarray, actual: np.ndarray, buy_cost: float, sell_cost: float) -> dict:
    prev = np.concatenate(([0.0], signal[:-1]))
    net = signal * actual - position_change_cost(signal, prev, buy_cost, sell_cost)
    return {
        "cum_profit": cumulative_profit(signal, actual, buy_cost, sell_cost),
        "geo_return": cumulative_profit_geometric(signal, actual, buy_cost, sell_cost),
        "sharpe": sharpe_like(net),
        "turnover": float(np.abs(signal - prev).mean()),
    }


def plot(df: pd.DataFrame, buy_cost: float, sell_cost: float, out: Path) -> None:
    apply_style()
    symbols = list(df.symbol.unique())
    groups = [s.split(":")[-1] for s in symbols] + ["Mean"]
    panels = [
        ("geo_return", "Geometric return, evaluated without cost", False, True),
        ("geo_return", f"Geometric return, evaluated with cost (buy {buy_cost:.3%}, sell {sell_cost:.3%})", True, True),
        ("turnover", "Mean daily position change", True, False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    width = 0.26
    x = np.arange(len(groups))
    for ax, (metric, title, with_cost, pct) in zip(axes, panels):
        sub = df[df.with_cost == with_cost]
        for k, cond in enumerate(CONDITIONS):
            vals = [sub[(sub.symbol == s) & (sub.condition == cond)][metric].iloc[0] for s in symbols]
            vals.append(float(np.mean(vals)))
            bars = ax.bar(x + (k - 1) * width, vals, width * 0.92, label=CONDITIONS[cond], color=COLORS[cond], zorder=3)
            label_bars(ax, bars[-1:], "{:+.0%}" if pct else "{:.2f}", fontsize=7)
        zero_line(ax)
        if pct:
            percent_axis(ax)
        ax.margins(y=0.15)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.12), ncol=3, handlelength=1.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(groups)
    fig.suptitle("Effect of trading cost on the GRU conditions (same cost applied to all)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    footnote(fig, "Values are labelled for the mean group; per-symbol values are in the accompanying CSV.")
    save(fig, out)


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
    p.add_argument("--buy-cost", type=float, default=PH_BUY_COST, help="Cost rate per unit of position increase for the 'with cost' condition (default 0.296%%)")
    p.add_argument("--sell-cost", type=float, default=PH_SELL_COST, help="Cost rate per unit of position decrease for the 'with cost' condition (default 0.396%%)")
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
                "mse": train_and_test("mse", 0.0, 0.0, data, args, device, alpha),
                "pa_nocost": train_and_test("profit-aware", 0.0, 0.0, data, args, device, alpha),
                "pa_cost": train_and_test("profit-aware", args.buy_cost, args.sell_cost, data, args, device, alpha),
            }
        except Exception as e:
            print(f"[{i}/{len(symbols)}] {symbol} FAILED: {e}")
            continue
        for cond, m in runs.items():
            for with_cost in (False, True):
                buy, sell = (args.buy_cost, args.sell_cost) if with_cost else (0.0, 0.0)
                sim = simulate(m["signal_np"], m["actual_np"], buy, sell)
                rows.append({
                    "symbol": symbol, "condition": cond, "with_cost": with_cost,
                    "eval_buy_cost": buy, "eval_sell_cost": sell,
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

    cost_label = f"buy {args.buy_cost:.3%}, sell {args.sell_cost:.3%}"
    lines: list[str] = []
    for label, with_cost in (("WITHOUT cost (evaluated at zero cost)", False), (f"WITH cost (evaluated at {cost_label})", True)):
        sub = df[df.with_cost == with_cost]
        lines += [f"\n=== {label} ===",
                  f"{'condition':<32}{'GeoRet':>10}{'AddRet':>10}{'Sharpe':>9}{'Turnover':>10}{'Final PHP':>12}"]
        for cond, name in CONDITIONS.items():
            c = sub[sub.condition == cond]
            lines.append(f"{name:<32}{c.geo_return.mean():>+10.4f}{c.cum_profit.mean():>+10.4f}"
                         f"{c.sharpe.mean():>+9.3f}{c.turnover.mean():>10.4f}{c.final_balance.mean():>12,.0f}")
    drag = df[~df.with_cost].set_index(["symbol", "condition"]).geo_return - \
        df[df.with_cost].set_index(["symbol", "condition"]).geo_return
    lines += ["\n=== Cost drag on geometric return (no-cost minus with-cost), mean over symbols ==="]
    for cond, name in CONDITIONS.items():
        lines.append(f"{name:<32}{drag.xs(cond, level='condition').mean():>+10.4f}")
    wc = df[df.with_cost].pivot(index="symbol", columns="condition", values="geo_return")
    lines.append(f"\nWith cost: pa_cost beats pa_nocost on {(wc.pa_cost > wc.pa_nocost).sum()}/{len(wc)} symbols, "
                 f"beats MSE on {(wc.pa_cost > wc.mse).sum()}/{len(wc)}")
    text = "\n".join(lines)
    print(text)
    out.with_name(out.name + "_summary.txt").write_text(text + "\n", encoding="utf-8")

    plot(df, args.buy_cost, args.sell_cost, out.with_suffix(".png"))
    print(f"\nwrote {out}.csv, {out}_summary.txt, {out}.png")


if __name__ == "__main__":
    main()
