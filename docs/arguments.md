# Command-Line Arguments

All arguments are optional and have sensible defaults.

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

## Usage Examples

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
