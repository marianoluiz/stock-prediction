# Command-Line Arguments

All arguments are optional and have sensible defaults.

There are now three entry points:

- `main.py` — everything in one run (train + evaluate + compare)
- `train.py` — train only (test set untouched)
- `evaluate.py` — evaluate a saved model on the test set
- `predict.py` — forecast the next N days with a saved model

`train.py` and `evaluate.py` use the exact same chronological 70/15/15 split, so
the test window is identical between them.

## Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--symbol` | `str` | `AAPL` | Stock ticker symbol (e.g. `AAPL`, `MSFT`, `GOOGL`). Passed directly to Yahoo Finance. |
| `--start` | `str` | `2018-01-01` | Start date for historical data (`YYYY-MM-DD`). |
| `--end` | `str` | `None` | End date for historical data. If `None`, uses up to today. |

## Model Architecture

| Argument | Type | Default | Description |
|---|---|---|---|
| `--sequence-length` | `int` | `30` | Number of past trading days the model sees to predict the next day's return. |
| `--hidden-size` | `int` | `64` | Number of neurons in each GRU layer. Larger values increase model capacity but slow down training. |
| `--num-layers` | `int` | `2` | Number of stacked GRU layers. |
| `--dropout` | `float` | `0.2` | Fraction of neurons randomly zeroed during training (regularization). |

## Training

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `50` | Number of full passes over the training data. |
| `--batch-size` | `int` | `64` | Number of sequences per training step. |
| `--lr` | `float` | `1e-3` | Learning rate for the Adam optimizer. |

## Loss Function

| Argument | Type | Default | Description |
|---|---|---|---|
| `--loss` | `str` | `profit-aware` | Loss function: `mse` (baseline) or `profit-aware` (custom). |
| `--compare` | flag | `False` | Run both loss functions and print a side-by-side comparison table. |
| `--alpha` | `float` | `1.0` | Sharpness of tanh signal: higher = more aggressive binary-like positioning. |
| `--transaction-cost` | `float` | `0.001` | Simulated trading cost per trade as a rate (0.001 = 0.1%). |
| `--capital` | `float` | `100000.0` | Starting capital in PHP for simulated trading display. Converts percentage returns to PHP amounts in output. |

## Signal Conversion (Evaluation)

During evaluation, predicted returns are converted to trading signals using **thresholding**:

- `pred > 0` → signal = `1.0` (full long position)
- `pred < 0` → signal = `-1.0` (full short position)
- `pred = 0` → signal = `0` (no position)

This ensures both MSE and profit-aware models receive identical position sizing, making the comparison purely about **directional accuracy** rather than prediction magnitude.

## Usage Examples

### main.py — everything in one run

```bash
# Default (AAPL with profit-aware loss)
python main.py

# Run MSE baseline only
python main.py --loss mse

# Run profit-aware only (default)
python main.py --loss profit-aware

# Run both and print comparison table
python main.py --compare --symbol AAPL --epochs 50

# Different stock
python main.py --symbol MSFT --start 2015-01-01

# Custom training
python main.py --symbol GOOGL --hidden-size 128 --num-layers 3 --epochs 100
```

### train.py — train only, save a model

```bash
# Train the default profit-aware model (saves to results/profit_aware/)
python train.py

# Train an MSE baseline with a custom save path
python train.py --loss mse --save results/mse/gru_mse.pt

# Custom training run
python train.py --symbol NVDA --hidden-size 128 --epochs 100 --save models/nvda_pa.pt
```

### evaluate.py — test a saved model on the held-out test set

```bash
# Evaluate the default profit-aware model on AAPL's test window
python evaluate.py --model results/profit_aware/gru_profit_aware.pt

# Evaluate an MSE baseline with a trade log
python evaluate.py --model results/mse/gru_mse.pt --loss mse --trade-log

# Evaluate on a different ticker (uses that ticker's test window)
python evaluate.py --model results/profit_aware/gru_profit_aware.pt --symbol GOOGL --capital 50000
```

### predict.py — forecast the next N days (defense demo)

```bash
# Forecast 5 trading days after the latest AAPL data
python predict.py --model results/profit_aware/gru_profit_aware.pt

# 10 days on a different stock
python predict.py --model models/nvda_pa.pt --symbol NVDA --days 10

# Use a different history window as the seed
python predict.py --model results/profit_aware/gru_profit_aware.pt --symbol MSFT --days 7 --sequence-length 30
```
