# Enhanced GRU via Profit-Aware Loss Learning

This project implements your thesis idea:

- Predict next-step **returns** with a GRU
- Convert prediction into a continuous trading signal: `signal_t = tanh(alpha * r_hat_t)`
- Optimize a **profit-aware objective**:

   cost_t = c * |signal_t - signal_{t-1}|

   profit_t = signal_t * r_t - cost_t

   Loss = -(1 / N) * sum(profit_t)

## Project Structure

```
├── main.py                 # CLI entry point, orchestrates everything
├── models/gru_model.py     # GRU(input=1, hidden=64, layers=2) -> LayerNorm -> Linear
├── training/train.py       # Training loop with profit-aware loss
├── utils/
│   ├── preprocessing.py    # Yahoo Finance download, returns calc, sliding windows
│   ├── trading.py          # tanh-based trading signal, transaction costs, loss
│   └── metrics.py          # Directional accuracy, cumulative profit, Sharpe ratio
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

`python main.py --symbol GOOGL --start 2018-01-01`

Artifacts are saved in [results](results):

- trained model (`gru_profit_aware.pt`)
- loss curve (`loss_curve.png`)
- profit curve (`profit_curve.png`)
