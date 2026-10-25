# Command-Line Arguments

All arguments are optional and have sensible defaults.

There are now three entry points:

- `main.py` — everything in one run (train + evaluate + compare)
- `scripts/train.py` — train only (test set untouched)
- `scripts/evaluate.py` — evaluate a saved model on the test set
- `scripts/predict.py` — forecast the next N days with a saved model

`train.py` and `evaluate.py` use the exact same chronological 70/15/15 split, so
the test window is identical between them.

## Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--symbol` | `str` | `PSE:PSEI` | PSE ticker in `EXCHANGE:TICKER` form (e.g. `PSE:PSEI`, `PSE:ICT`, `PSE:BDO`). Fetched from TradingView via `tvDatafeed`. The study's primary series is the PSEi index (`benchmark.py`'s `INDEX_SYMBOL`); the optional stock universe is `PSEI_STOCKS`. |
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
| `--loss-lambda` | `float` | `0.7` | Weight of the MSE calibration term added to the profit-aware loss. `0` = pure profit loss `-mean(net_profit)`. Larger values add MSE calibration, which prevents gradient saturation but steers the model toward an MSE-like fit. |
| `--buy-cost` | `float` | `0.00296` | Cost per unit of position increase, i.e. a purchase (0.296%: PSE broker commission 0.25% + 12% VAT on it + PSE/SCCP fees). |
| `--sell-cost` | `float` | `0.00396` | Cost per unit of position decrease, i.e. a sale (0.396%: the buy-side fees + 0.1% stock transaction tax). |
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
# Default (PSE:PSEI with profit-aware loss)
python main.py

# Run MSE baseline only
python main.py --loss mse

# Run profit-aware only (default)
python main.py --loss profit-aware

# Run both and print comparison table
python main.py --compare --symbol PSE:BDO --epochs 50

# Different stock
python main.py --symbol PSE:JFC --start 2015-01-01

# Custom training
python main.py --symbol PSE:ALI --hidden-size 128 --num-layers 3 --epochs 100
```

### train.py — train only, save a model

Models and curves are named with the symbol + date range (the actual last date
is filled in automatically, and `:` becomes `-`, e.g. `PSE-PSEI_profit_aware_2018-01-01_2026-10-23`):

```bash
# Train the default profit-aware model
#   saves to results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_<END>.pt
python scripts/train.py

# Custom save path (curves still go next to it, named with symbol + range)
python scripts/train.py --symbol PSE:JFC --hidden-size 128 --epochs 100 --save models/jfc_pa.pt
```

### evaluate.py — test a saved model on the held-out test set

```bash
# Evaluate the default profit-aware model on the PSEi's test window
python scripts/evaluate.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_2026-10-23.pt

# Evaluate an MSE baseline with a trade log + save a metrics summary file
python scripts/evaluate.py --model results/mse/PSE-PSEI_mse_2018-01-01_2026-10-23.pt --loss mse --trade-log --metrics-out results/mse/PSE-PSEI_mse_metrics.txt

# Evaluate on a different ticker (uses that ticker's test window)
python scripts/evaluate.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_2026-10-23.pt --symbol PSE:BDO --capital 50000
```

### predict.py — forecast the next N days (defense demo)

```bash
# Forecast 5 trading days after the latest PSEi data
python scripts/predict.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_2026-10-23.pt

# 10 days on a different stock
python scripts/predict.py --model models/jfc_pa.pt --symbol PSE:JFC --days 10

# Use a different history window as the seed
python scripts/predict.py --model results/profit_aware/PSE-PSEI_profit_aware_2018-01-01_2026-10-23.pt --symbol PSE:PSEI --days 7 --sequence-length 30
```
