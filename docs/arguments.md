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
| `--alpha` | `float` | `5.0` | Weight of the profit-aware component in the loss function. Higher values emphasize profit optimization over pure MSE. |
| `--transaction-cost` | `float` | `0.001` | Simulated trading cost per trade as a rate (0.001 = 0.1%). |

## Usage Examples

```bash
# Default (AAPL)
python main.py

# Different stock
python main.py --symbol MSFT --start 2015-01-01

# Custom training
python main.py --symbol GOOGL --hidden-size 128 --num-layers 3 --epochs 100
```
