# Enhanced GRU via Profit-Aware Loss Learning

This project implements your thesis idea:

- Predict next-step **returns** with a GRU
- Convert prediction into a continuous trading signal: `signal_t = tanh(alpha * r_hat_t)`
- Optimize a **profit-aware objective**:

   cost_t = c * |signal_t - signal_{t-1}|

   profit_t = signal_t * r_t - cost_t

   Loss = -(1 / N) * sum(profit_t)

## Project Structure

- [main.py](main.py)
- [models/gru_model.py](models/gru_model.py)
- [training/train.py](training/train.py)
- [utils/preprocessing.py](utils/preprocessing.py)
- [utils/trading.py](utils/trading.py)
- [utils/metrics.py](utils/metrics.py)

## Setup

1. Create and activate a virtual environment:

   `python -m venv .venv`

   On macOS/Linux:

   `source .venv/bin/activate`

   On Windows Git Bash:

   `source .venv/Scripts/activate`

   On Windows PowerShell:

   `.\.venv\Scripts\Activate.ps1`

2. Install the dependencies from the locked file:

   `pip install pip-tools`

   `pip-sync requirements.txt`

3. If you change [requirements.in](requirements.in), regenerate the lock file:

   `pip-compile requirements.in`

This last step is usually done by the repository owner or maintainer.

## Run

Example:

`python main.py --symbol AAPL --start 2018-01-01 --sequence-length 30 --alpha 5 --transaction-cost 0.001 --epochs 50`

Artifacts are saved in [results](results):

- trained model (`gru_profit_aware.pt`)
- loss curve (`loss_curve.png`)
- profit curve (`profit_curve.png`)
