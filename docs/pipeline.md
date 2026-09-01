# Pipeline

There are three entry points that share one data-preparation pipeline
(`utils/pipeline.py` -> `prepare_data()`):

- `train.py` — trains and saves a model; the test set is never touched.
- `evaluate.py` — loads a saved model and evaluates it on the exact same test
  window (identical split) without retraining.
- `main.py` — the original all-in-one run: train + evaluate (+ `--compare`).

`predict.py` — standalone live-forecast demo (no train/test split; seeds from
the last `sequence_length` returns and iteratively forecasts the next N days).

## 1. Data Ingestion

`utils/preprocessing.py` - `load_stock_data()`

- Downloads OHLCV data from Yahoo Finance (via `yfinance`) or reads from a local CSV cache.
- Returns a clean DataFrame indexed by date.

## 2. Return Computation

`utils/preprocessing.py` - `compute_returns()`

- Converts raw closing prices into daily percentage returns:
  `r_t = (P_{t+1} - P_t) / P_t`
- This normalizes the data and removes price-level bias.

## 3. Sliding Window Sequences

`utils/preprocessing.py` - `create_sequences()`

- Slides a fixed-size window (default 30 days) over the return series.
- Each sample: input = `r[t] .. r[t+30]`, target = `r[t+31]`.
- Produces tensors of shape `[N, sequence_length, 1]` (input) and `[N]` (target).

## 4. Chronological Split

`utils/preprocessing.py` - `train_val_test_split()`

- Splits data sequentially (no shuffling) into train (70%), validation (15%), test (15%).
- Preserves temporal ordering to prevent look-ahead bias.

## 5. Training Loop

`training/train.py` - `fit()` | `utils/trading.py` - `profit_aware_loss()` | `models/gru_model.py` - `GRUReturnPredictor`

Each epoch, for every batch:

1. **Forward pass**: GRU takes a 30-day return window and outputs a predicted next-day return.
2. **Signal generation (training)**: the model outputs a continuous **position**
   `signal = tanh(alpha * predicted_return)` — a position size between -1 and +1.
   Positive = go long, negative = go short; magnitude scales confidence. This is
   the smooth, differentiable signal used inside the *loss*.
3. **Transaction costs**: Penalizes position changes between consecutive steps:
   `cost = cost_rate * |signal_t - signal_{t-1}|`
4. **Net profit per step**: `signal_t * actual_return_t - cost_t`
5. **Loss (profit-aware)**: `-mean(net_profit)` — minimizing loss = maximizing profit:
   `loss = -mean(net_profit)`
   - An **MSE calibration term** `loss_lambda * mean((pred - actual)^2)` is added via
     `--loss-lambda` (default `0.1`). At `0`, `pred` drifts unbounded and `tanh`
     saturates, killing the gradient (see `TODO.md`). The default `0.1` pins `pred`
     to realistic magnitudes so `tanh` never saturates and the model keeps learning —
     but it also steers the model toward an MSE-like fit and away from the raw profit
     objective.
   - The MSE baseline uses just `mean((pred - actual)^2)`.
6. **Backpropagation**: because the loss uses the smooth `tanh` position, forward
   and backward are the *same* function — there is **no straight-through estimator**
   and no fake gradient. Gradients stay alive for every prediction magnitude.
7. **Weight update**: Adam optimizer adjusts parameters.

**Evaluation**: during eval, the continuous prediction is discretized to a
long / flat / short action: `signal = sign(predicted_return)` when
`|tanh(alpha * predicted_return)| >= --signal-threshold`, else `0` (flat). At
the default `--signal-threshold 0.0` this is always full `sign(predicted_return)`
(no flat state, the original rule). Raising the threshold sits out low-confidence
predictions instead of always taking a full ±1 position, which also cuts
transaction-cost churn on those days. This is the signal fed to the profit
metrics (`directional_accuracy` compares `sign(pred)` to `sign(actual)`
directly and is unaffected by the threshold). So training optimizes a smooth
partial position, while reported profit uses this discrete action rule.

Repeated for all batches across all epochs. Model weights saved to `results/gru_profit_aware.pt`. Loss and profit curves saved as PNGs to `results/`.

`train.py` saves the final-epoch weights to `--save` (default
`results/<loss>/gru_<loss>.pt`) alongside its loss/profit curves, without
evaluating the test set.

## 6. Evaluation Metrics

`utils/metrics.py`

After training, the model is evaluated on the **test set** (unseen data) using these metrics:

| Metric | Formula | What it measures | Good vs Bad |
|---|---|---|---|
| `directional_accuracy` | `(sign(pred) == sign(actual)).mean()` | How often the model gets the direction right (up vs down). | Bad: ~50% (random guess). Good: 55%+. Great: 60%+. |
| `cumulative_profit` | `sum(signal * return - cost)` | **Additive** total profit/loss. Each trade uses original capital. Simple but ignores compounding. | Bad: negative. Good: positive. Great: beats buy-and-hold. |
| `cumulative_profit_geometric` | `product(1 + signal * return - cost) - 1` | **Geometric** total profit/loss. Balance compounds: losses shrink next trade size, gains expand it. Reflects real-world trading. | Bad: negative. Good: positive. Great: beats buy-and-hold. |
| `sharpe_like` | `(mean / std) * sqrt(252)` | Risk-adjusted return — return per unit of volatility. | Bad: < 1. Good: 1–2. Great: 2–3. Suspicious: > 3 (likely overfitting). |

**Additive vs Geometric:** The additive model assumes unlimited capital per trade. The geometric model tracks your actual bankroll — if you lose 2% on day 1, you trade with 98% on day 2. Over many volatile trades, geometric returns are typically lower than additive returns (compounding works against you on losses). The geometric return is the more realistic metric.

