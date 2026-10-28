# Enhanced GRU via Profit-Aware Loss Learning

A GRU predicts the PSEi index's next-day **return**. The prediction becomes a
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

`alpha = 1 / std(train returns)` is calibrated per training window. The MSE
term keeps predictions on a realistic scale so that `tanh` does not saturate.
The original algorithm (Steps 0-11) is in [docs/algorithm.md](docs/algorithm.md).
**The full description of the current system, the evaluation design and the
latest results is in [docs/system-and-results.md](docs/system-and-results.md)**,
written as the reference for the thesis write-up.

## Market and trading costs

The study covers the **Philippine Stock Exchange only**:

- **Primary series:** the PSEi index itself (`PSE:PSEI`), evaluated with
  4-fold expanding-window walk-forward validation (four out-of-time test
  windows, March 2024 to October 2026).
- **Optional stock universe:** the 30 PSEi constituents as of the 3 August
  2026 rebalancing, minus Maynilad (MYNLD, listed November 2025, too little
  history). Run with `--stocks`. Then the 29 stocks are reported as one group
  and the PSEi separately, never averaged in. Defined in `benchmark.py`
  (`PSEI_STOCKS`, `INDEX_SYMBOL`).
- **Data:** daily OHLCV from TradingView via `tvDatafeed`. Symbols are written
  as `PSE:TICKER`, for example `PSE:PSEI` or `PSE:ICT`. Data is cached in
  `data/` on first download. The PSEi has no volume data (all zeros), so its
  volume z-score feature is a constant 0.

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

The separate [`with-sop-2` paper](docs/with-sop-2.docx) has a numbered
[11-figure gallery](figures/README.md). Open the
[PSEi figure notebook](notebooks/PSEi_Thesis_Figures_Colab.ipynb) in Google
Colab and run all cells to display the exact paper figures and rebuild their
available source plots from the committed results. This gallery does not
retrain the models; the full experiment command below does.

One command produces every result and figure:

```
.venv/Scripts/python.exe run_thesis.py
```

It runs five steps on the PSEi, in order. All of them use the defaults: 50
epochs, lambda 0.7, buy cost 0.296%, sell cost 0.396%, seed 42; step 4 varies
lambda. With a GPU, steps 1-3 and 5 take roughly 5 minutes; step 4 trains on
CPU and adds about 15. You can also run any step on its own.
`run_thesis.py --stocks` runs the 29-stock version of steps 1-3 instead.

| # | Command | What it produces (in `results/`) |
|---|---|---|
| 1 | `benchmark.py --walk-forward --folds 4 --charts` | `benchmark_summary_psei_<time>.csv`: one row per fold and loss. `..._daily.csv`: every test day's position and net return for both GRUs, buy & hold and the TA baselines. `..._summary.txt`: means, per-fold table, win counts, baselines, position behaviour, fold stability, degenerate runs. `..._charts/`: the figures listed below. |
| 2 | `scripts/significance.py` | `significance.csv` and `significance.png`: daily paired test of profit-aware minus MSE (Newey-West standard errors, ~630 test days), hit rate vs 50%, and a fold-level sign test. With a `--stocks` CSV it runs the paired t-test across the 29 stocks instead. |
| 3 | `scripts/cost_comparison.py --walk-forward --folds 4` | `cost_comparison.csv`, `cost_comparison_summary.txt` and `cost_comparison.png`: MSE, profit-aware without cost in the loss, and profit-aware with cost in the loss, each scored without and with cost, on the same 4 folds. |
| 4 | `scripts/sop2_sweep.py` | SOP 2 calibration-weight sweep, all in `results/sop2/`: one `lambda_<x>/` folder per λ (0, 0.3, 0.7, 1.0, 2.0; `0p3` = 0.3) with that run's `benchmark.csv`, `_daily.csv`, `_summary.txt` and `run.log`, plus `comparison.csv`/`.md`, `sensitivity.png` and `manifest.json` (settings, git revision, data hash). Runs on CPU to match the committed sweep; `--compare-only` rebuilds the comparison without training. |
| 5 | `main.py --compare --trade-log --symbol PSE:PSEI` | Worked example on the single 70/15/15 split. The `mse/` and `profit_aware/` folders get the model weights plus loss and profit curves, and the full trade log is printed. |

Figures in `_charts/` (index-only walk-forward run: one bar group per fold,
plus the mean):

- `profit_comparison.png`, `profit_geo_comparison.png`, `buy_hold_comparison.png`
- `sharpe_comparison.png`, `rmse_comparison.png`, `directional_acc_comparison.png`
- `strategy_comparison.png`: MSE vs profit-aware vs buy & hold vs SMA, RSI and
  MACD, mean over the folds
- `per_fold_geo_diff.png`: profit-aware minus MSE geometric return, per fold
- `equity_curve.png`: growth of 1 PHP over the whole out-of-sample period
  (folds stitched end to end) for every model and baseline
- `per_symbol_geo_diff.png`: per stock (only in a `--stocks` run)

To rebuild the summary, figures or significance test from an existing
benchmark CSV, without retraining:

```
.venv/Scripts/python.exe benchmark.py --from-csv results/benchmark_summary_psei_<time>.csv --charts
.venv/Scripts/python.exe scripts/significance.py --csv results/benchmark_summary_psei_<time>.csv
```

Both scripts default to the newest `results/benchmark_summary_*.csv` when
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
| Mean position / mean \|position\| / turnover | Average signed position, average position size, and average daily position change. A mean position close to ± the mean size means the model sits on one side of the market. |

## Other entry points

All entry points use the same data pipeline (`utils/pipeline.py`). Without
`--walk-forward` they use the same chronological 70/15/15 split, so the test
window is always identical. Every entry point defaults to `PSE:PSEI`.

```
# Train + evaluate the PSEi (one loss, or both with --compare); --symbol for a stock
.venv/Scripts/python.exe main.py --compare --trade-log
.venv/Scripts/python.exe main.py --symbol PSE:BDO --compare --trade-log

# Benchmark: PSEi only (default), single split or walk-forward;
# all constituents + the index (--stocks); or a custom list
.venv/Scripts/python.exe benchmark.py --walk-forward --folds 4 --charts
.venv/Scripts/python.exe benchmark.py --charts
.venv/Scripts/python.exe benchmark.py --stocks --charts
.venv/Scripts/python.exe benchmark.py --symbols PSE:ICT,PSE:BDO --epochs 30

# Train only (test set untouched), then evaluate, then forecast
.venv/Scripts/python.exe scripts/train.py --loss profit-aware
.venv/Scripts/python.exe scripts/evaluate.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_<END>.pt --trade-log
.venv/Scripts/python.exe scripts/predict.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_<END>.pt --days 5

# Live demo: train both models on a chosen ticker and compare trade logs
streamlit run webapp.py
```

Saved models and curves are named `<SYMBOL>_<loss>_<start>_<end>`. The `:`
in the symbol becomes `-`, for example
`results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_2026-10-23.pt`. Every
argument and its default is listed in [docs/arguments.md](docs/arguments.md).

## Project structure

```
├── run_thesis.py             # Runs every thesis step (see "Thesis run")
├── main.py                   # Train + evaluate one symbol (--compare for both losses)
├── benchmark.py              # MSE vs profit-aware on the PSEi (or --stocks universe)
├── webapp.py                 # Streamlit live demo
├── scripts/
│   ├── train.py              # Train only, save the model
│   ├── evaluate.py           # Evaluate a saved model on the test set
│   ├── predict.py            # Forecast the next N days with a saved model
│   ├── significance.py       # Profit-aware minus MSE significance tests
│   ├── cost_comparison.py    # With-cost vs without-cost comparison
│   ├── sop2_sweep.py         # SOP 2: profit-aware run at several lambdas -> results/sop2/
│   ├── sop2_compare.py       # SOP 2 comparison table and sensitivity chart
│   └── plot_loss_comparison.py  # Benchmark comparison figures
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
