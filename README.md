# Enhanced GRU via Profit-Aware Loss Learning

This project implements your thesis idea:

- Predict next-step **returns** with a GRU
- Convert prediction into a continuous trading signal: `signal_t = tanh(alpha * r_hat_t)`
- Optimize a **profit-aware objective**:

   cost_t = c * |signal_t - signal_{t-1}|

   profit_t = signal_t * r_t - cost_t

   Loss = -(1 / N) * sum(profit_t)

## Return Metrics

The evaluation tracks two types of cumulative returns:

- **Additive Return** (simple sum): `sum(signal * return - cost)`. Each trade uses the original capital. Simple but unrealistic — ignores compounding.
- **Geometric Return** (compounding): `product(1 + signal * return - cost) - 1`. Each trade uses the current balance. Losses shrink future trade sizes; gains expand them. Reflects real-world trading.

## Project Structure

```
├── main.py                 # All-in-one: train + evaluate (+ compare) in one run
├── train.py                # Train only, save the model (test set untouched)
├── evaluate.py             # Evaluate a saved model on the held-out test set
├── predict.py              # Forecast the next N days using a saved model
├── models/gru_model.py     # GRU(input=1, hidden=64, layers=2) -> LayerNorm -> Linear
├── training/train.py       # Training loop with profit-aware loss
├── utils/
│   ├── preprocessing.py    # Yahoo Finance download, returns calc, sliding windows
│   ├── pipeline.py         # Shared data-prep pipeline used by main/train/evaluate
│   ├── trading.py          # tanh-based trading signal, transaction costs, loss
│   ├── metrics.py          # Directional accuracy, cumulative profit (additive + geometric), Sharpe ratio
│   └── plotting.py         # Loss/profit curve plotting
├── data/AAPL_prices.csv    # 2,105 days of AAPL data (2018-2026)
├── results/                # Saved model + loss/profit curves
└── requirements-*.txt      # pip-tools managed deps (Win + Linux)
```

## Setup

1. Create and activate a virtual environment (windows):

   `python -m venv .venv`

   `source .venv/Scripts/activate`

2. Install the dependencies from the locked file:

   `pip install pip-tools`

   `pip-sync requirements-win.txt`

3. If you change [requirements.in](requirements.in), regenerate the lock file:

   `pip-compile requirements-win.in`

This last step is usually done by the repository owner or maintainer.

## Run

Example:

`python main.py --symbol AAPL --start 2018-01-01 --sequence-length 30 --alpha 5 --transaction-cost 0.001 --epochs 50`

`python main.py --symbol GOOGL --start 2018-01-01 --capital 50000`

### Compare MSE vs Profit-Aware

`python main.py --compare --epochs 50`

Artifacts are saved in [results](results):

- trained model (`gru_profit_aware.pt` or `gru_mse.pt`)
- loss curve (`loss_curve.png`)
- profit curve (`profit_curve.png`)

## After the main run: train, evaluate, and predict separately

`main.py` does everything in one go. If you want to reuse a trained model —
re-evaluate it on the test set, compare runs without retraining, or demo live
predictions at the defense — use the three separate scripts. They all share the
same data-prep pipeline (`utils/pipeline.py`), so the test window is always
identical.

### 1. Train (test set is left untouched)

`python train.py --symbol AAPL --loss profit-aware --epochs 50`

This trains on 70% of data, monitors on the 15% validation split, saves the
model and its loss/profit curves to `results/profit_aware/`, and **never
touches the test set**.

```bash
# Custom save path
python train.py --symbol NVDA --hidden-size 128 --epochs 100 --save models/nvda_pa.pt

# Train an MSE baseline instead
python train.py --loss mse
```

### 2. Evaluate (reload a saved model, no retraining)

`python evaluate.py --model results/profit_aware/gru_profit_aware.pt`

This rebuilds the identical chronological split, loads the saved weights, and
prints the test-set metrics (directional accuracy, cumulative/geometric
return, Sharpe-like ratio) for exactly the same window the model never saw.

```bash
# MSE model + full per-trade log
python evaluate.py --model results/mse/gru_mse.pt --loss mse --trade-log

# Evaluate the AAPL model on a different ticker's test window
python evaluate.py --model results/profit_aware/gru_profit_aware.pt --symbol GOOGL --capital 50000
```

### 3. Predict (live forecast for a defense demo)

`python predict.py --model results/profit_aware/gru_profit_aware.pt`

Seeds the model with the last 30 returns, then forecasts the next 5 trading
days (default), feeding each prediction back as input.

```bash
# 10 days on a different stock
python predict.py --model models/nvda_pa.pt --symbol NVDA --days 10
```

### Typical workflow

1. `python train.py --symbol AAPL --loss mse` and `python train.py --symbol AAPL --loss profit-aware`
2. `python evaluate.py --model results/mse/gru_mse.pt --loss mse`
3. `python evaluate.py --model results/profit_aware/gru_profit_aware.pt`
4. Compare the two metric tables in your thesis write-up.
5. `python predict.py --model results/profit_aware/gru_profit_aware.pt` for the live demo.

> `main.py --compare` still exists as a shortcut that trains both losses in one
> run and prints a side-by-side comparison table.
