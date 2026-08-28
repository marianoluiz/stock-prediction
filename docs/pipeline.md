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
2. **Signal generation**: `signal = sign(predicted_return)` — a binary full long (+1) / full short (-1) position.
   - Positive prediction = go long. Negative = go short.
   - The **same** binary signal is used in the loss, validation, and test metrics.
3. **Transaction costs**: Penalizes signal changes between consecutive time steps:
   `cost = cost_rate * |signal_t - signal_{t-1}|`
4. **Net profit per step**: `signal_t * actual_return_t - cost_t`
5. **Loss**: `-mean(net_profit)` — minimizing loss = maximizing profit.
   - Profit is good (positive) → loss is negative (small) → optimizer is happy
   - Profit is bad (negative) → loss is positive (large) → optimizer pushes to improve
6. **Backpropagation**: `sign` has zero slope everywhere, so gradients would die. A **straight-through estimator** (STE, `utils/trading.py` — `StraightThroughSignal`) replaces the backward derivative with the smooth slope of `tanh(alpha * predicted_return)`:
   `backward_slope = alpha * (1 - tanh(alpha * predicted_return)^2)`
   - `alpha` tunes how smooth that backward slope is (2–5 typical; higher = more aggressive/binary-like). It does **not** affect the forward signal or the reported profits.
   - The real ±1 signal is used in the loss; only the gradient is fudged, so the model still learns direction while being scored exactly as it trades.
7. **Weight update**: Adam optimizer adjusts parameters.

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

