# Enhanced GRU via Profit-Aware Loss Learning

A GRU predicts each stock's next-day **return**. The prediction becomes a
continuous trading position, and the model is trained on the simulated,
cost-adjusted profit of that position instead of on prediction error alone.
An MSE-trained GRU with the same architecture, features and data split is the
baseline.

```
signal_t = tanh(alpha * r_hat_t)                 position in (-1, 1)
delta_t  = signal_t - signal_{t-1}
cost_t   = c_buy * max(delta_t, 0) + c_sell * max(-delta_t, 0)
profit_t = signal_t * r_t - cost_t
Loss     = -mean(profit_t) + lambda * MSE(r_hat, r)  lambda = 0.7
```

`alpha = 1 / std(train returns)` is calibrated per stock. The MSE term keeps
predictions on a realistic scale so that `tanh` does not saturate. The
original algorithm (Steps 0-11) is in [docs/algorithm.md](docs/algorithm.md)
and the methodology chapter is in
[docs/methodology_chapter3.md](docs/methodology_chapter3.md).

## Market and trading costs

The study covers the **Philippine Stock Exchange only**:

- **Universe:** the 30 PSEi constituents as of the 3 August 2026 rebalancing,
  minus Maynilad (MYNLD). MYNLD listed in November 2025 and has about 190
  usable samples, too few for a 70/15/15 split. That leaves 29 stocks, plus
  the PSEi itself as a passive reference.
- **Tiers:** each stock is grouped by its total price return from 2018 to
  October 2026:
  - **strong:** return of +25% or more (8 stocks)
  - **mid:** between -25% and +25% (7 stocks)
  - **weak:** return of -25% or less (14 stocks)

  The universe and tiers are defined in `benchmark.py` (`PH_TIERED_SYMBOLS`).
- **Data:** daily OHLCV from TradingView via `tvDatafeed`. Symbols are written
  as `PSE:TICKER`, for example `PSE:ICT`. Data is cached in `data/` on first
  download.

Trading costs are asymmetric because the stock transaction tax applies only
to sales:

| Item | Buy | Sell |
|---|---|---|
| Broker commission (0.25%) | yes | yes |
| VAT on commission (12%) | yes | yes |
| PSE transaction fee + SCCP fee | yes | yes |
| Stock transaction tax (0.1%) | no | yes |
| **Total (used by the code)** | **0.296%** | **0.396%** |

How the rates are charged:

- A **position increase** (opening a long, or covering a short) is a purchase.
  It is charged at the buy rate, `--buy-cost 0.00296`.
- A **position decrease** (closing a long, or opening a short) is a sale. It
  is charged at the sell rate, `--sell-cost 0.00396`.

One function, `utils.trading.position_change_cost`, applies these costs
everywhere: the training loss, every metric, the trade log, the classic
strategy baselines and the web demo.

## Setup (Windows)

```
python -m venv .venv
source .venv/Scripts/activate
pip install pip-tools
pip-sync requirements-win.txt
```

The bare `python` on PATH is a different interpreter without the
dependencies. Always use `.venv/Scripts/python.exe` or activate the venv
first. Training uses CUDA when it is available.

## Thesis run

One command produces every result and figure:

```
.venv/Scripts/python.exe run_thesis.py
```

It runs four steps in order. All of them use the defaults: 50 epochs, lambda
0.7, buy cost 0.296%, sell cost 0.396%, seed 42. With a GPU, expect roughly
20-30 minutes in total. You can also run any step on its own.

| # | Command | What it produces (in `results/`) |
|---|---|---|
| 1 | `benchmark.py --charts` | `benchmark_summary_ph_<time>.csv`: one row per symbol and loss. `benchmark_summary_ph_<time>_summary.txt`: aggregate, per-tier, win-rate and baseline tables. `benchmark_summary_ph_<time>_charts/`: the figures listed below. |
| 2 | `scripts/significance.py` | `significance.csv` and `significance.png`: paired t-test (profit-aware minus MSE), overall and per tier, on the latest benchmark CSV |
| 3 | `scripts/cost_comparison.py` | `cost_comparison.csv`, `cost_comparison_summary.txt` and `cost_comparison.png`: MSE, profit-aware without cost in the loss, and profit-aware with cost in the loss, each scored without and with cost. Uses two symbols per tier. |
| 4 | `main.py --compare --trade-log --symbol PSE:ICT` | Worked example. The `mse/` and `profit_aware/` folders get the model weights plus loss and profit curves, and the full trade log is printed. |

Figures in `_charts/`:

- `profit_comparison.png`
- `profit_geo_comparison.png`
- `buy_hold_comparison.png`
- `sharpe_comparison.png`
- `rmse_comparison.png`
- `directional_acc_comparison.png`
- `strategy_comparison.png`: MSE vs profit-aware vs buy & hold vs SMA, RSI and
  MACD
- `per_symbol_geo_diff.png`: profit-aware minus MSE geometric return, per stock

To redraw figures or retest significance from an existing benchmark CSV,
without retraining:

```
.venv/Scripts/python.exe scripts/plot_loss_comparison.py --csv results/benchmark_summary_ph_<time>.csv
.venv/Scripts/python.exe scripts/significance.py --csv results/benchmark_summary_ph_<time>.csv
```

Both scripts default to the newest `results/benchmark_summary_ph_*.csv` when
`--csv` is omitted.

## Metrics

| Metric | Meaning |
|---|---|
| Directional accuracy | Share of days where `sign(prediction) == sign(actual return)` |
| Cumulative return (additive) | `sum(profit_t)`: each trade sized on the starting capital |
| Geometric return | `prod(1 + profit_t) - 1`: each trade sized on the current balance (compounding) |
| Sharpe-like ratio | `mean(profit_t) / std(profit_t) * sqrt(252)`, zero risk-free rate |
| MSE / MAE / RMSE | Error of the predicted daily return |
| Buy & hold | `prod(1 + r_t) - 1` over the test window, no costs (reference only) |
| Degenerate run | The position never changes side over the whole test window. The model learned no day-to-day signal. |

## Other entry points

All entry points use the same data pipeline (`utils/pipeline.py`) and the same
chronological 70/15/15 split, so the test window is always identical.

```
# Train + evaluate one stock (one loss, or both with --compare)
.venv/Scripts/python.exe main.py --symbol PSE:BDO --compare --trade-log

# Benchmark a subset, or with walk-forward validation
.venv/Scripts/python.exe benchmark.py --symbols PSE:ICT,PSE:BDO --epochs 30
.venv/Scripts/python.exe benchmark.py --walk-forward --folds 4 --charts

# Train only (test set untouched), then evaluate, then forecast
.venv/Scripts/python.exe scripts/train.py --symbol PSE:ICT --loss profit-aware
.venv/Scripts/python.exe scripts/evaluate.py --model results/profit_aware/PSE-ICT_profit_aware_2018-01-01_<END>.pt --trade-log
.venv/Scripts/python.exe scripts/predict.py --model results/profit_aware/PSE-ICT_profit_aware_2018-01-01_<END>.pt --days 5

# Live demo: train both models on a chosen ticker and compare trade logs
streamlit run webapp.py
```

Saved models and curves are named `<SYMBOL>_<loss>_<start>_<end>`. The `:`
in the symbol becomes `-`, for example
`results/profit_aware/PSE-ICT_profit_aware_2018-01-01_2026-10-23.pt`. Every
argument and its default is listed in [docs/arguments.md](docs/arguments.md).

## Project structure

```
├── run_thesis.py             # Runs every thesis step (see "Thesis run")
├── main.py                   # Train + evaluate one stock (--compare for both losses)
├── benchmark.py              # MSE vs profit-aware across the PSEi universe
├── webapp.py                 # Streamlit live demo
├── scripts/
│   ├── train.py              # Train only, save the model
│   ├── evaluate.py           # Evaluate a saved model on the test set
│   ├── predict.py            # Forecast the next N days with a saved model
│   ├── significance.py       # Paired t-test, profit-aware minus MSE
│   ├── cost_comparison.py    # With-cost vs without-cost comparison
│   └── plot_loss_comparison.py  # Benchmark figures, by tier
├── models/gru_model.py       # 2-layer GRU -> LayerNorm -> Linear (tanh-bounded head)
├── training/train.py         # Training loop, per-epoch metrics, early stopping
├── utils/
│   ├── preprocessing.py      # Download, returns, features, windows, splits
│   ├── pipeline.py           # Shared data preparation for every entry point
│   ├── trading.py            # tanh signal, PSE buy/sell cost model, profit-aware loss
│   ├── metrics.py            # Accuracy, additive/geometric return, Sharpe, trade log
│   ├── baselines.py          # SMA crossover, RSI and MACD strategy baselines
│   ├── plotting.py           # Training loss/profit curves
│   └── chart_style.py        # Shared figure style and colors
├── data/                     # Cached daily OHLCV CSVs, one per symbol
├── results/                  # Models, curves, benchmark tables and figures
└── docs/                     # Algorithm, methodology chapter, arguments
```

To change dependencies, edit `requirements-win.in` and regenerate the lock
file with `pip-compile requirements-win.in`.
