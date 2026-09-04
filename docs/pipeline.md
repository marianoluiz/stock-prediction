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

## 3. Lagged-Return Feature Channels

`utils/preprocessing.py` - `build_lagged_features()`

- Adds `return_lag1`, `return_lag5`, `return_lag10` as explicit channels alongside
  the raw same-day return (`DEFAULT_LAGS = (1, 5, 10)`), so the GRU gets direct
  access to specific historical returns at every timestep instead of relying on
  its recurrence to bridge that many steps on its own.
- Column 0 stays the raw return (also the prediction target downstream);
  rows without enough history for the longest lag are dropped.

## 3b. Volume Z-Score and Rolling Volatility

`utils/preprocessing.py` - `build_volume_zscore()`, `build_rolling_volatility()`

- `volume_zscore`: rolling z-score of trading volume over a 20-day window
  (`DEFAULT_VOLUME_WINDOW = 20`) — how unusual today's volume is versus its
  own trailing history. Z-scored (unlike returns) because raw volume levels
  vary by orders of magnitude across tickers.
- `volatility`: rolling standard deviation of daily returns over a 10-day
  window (`DEFAULT_VOLATILITY_WINDOW = 10`) — a realized-volatility regime
  signal on a shorter timescale than the volume z-score, so it reacts to
  vol clustering faster. Fed raw (no z-scoring) since returns are already on
  a small, comparable scale.
- Both are joined onto the lagged-return frame in `build_feature_frame()`;
  rows without enough history for the longest lag, the volume window, or the
  volatility window are dropped.

## 4. Sliding Window Sequences

`utils/preprocessing.py` - `create_sequences()`

- Slides a fixed-size window (default 30 days) over the multi-channel feature array.
- Each sample: input = `features[t : t+30]`, target = `features[t+30, 0]` (the raw return).
- Produces tensors of shape `[N, sequence_length, 1 + len(lags) + 2]` (return +
  lags + volume z-score + volatility) (input) and `[N]` (target).

## 5. Chronological Split

`utils/preprocessing.py` - `train_val_test_split()`

- Splits data sequentially (no shuffling) into train (70%), validation (15%), test (15%).
- Preserves temporal ordering to prevent look-ahead bias.

## 6. Training Loop

`training/train.py` - `fit()` | `utils/trading.py` - `profit_aware_loss()` | `models/gru_model.py` - `GRUReturnPredictor`

Each epoch, for every batch:

1. **Forward pass**: GRU takes a 30-day window of the raw return, lagged-return
   channels (`return_lag1/5/10`), the volume z-score, and rolling volatility,
   and outputs a predicted next-day return. The output head hard-bounds this
   prediction: `pred = output_scale * tanh(raw_logit)`, where `output_scale =
   --output-cap-std / alpha` (default `--output-cap-std 5.0`, i.e. +-5 standard
   deviations of that symbol's training returns). Previously the head was a
   plain unbounded `Linear`, and nothing stopped `pred` from drifting to
   arbitrary magnitude while chasing marginal loss improvement past the point
   where `tanh(alpha * pred)` in the signal/loss below had already saturated
   (observed empirically as RMSE blowing up several-fold past the true return
   scale). The cap makes that structurally impossible instead of relying on
   `loss_lambda` alone (see step 5) to keep `pred` in a sane range.
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
     `--loss-lambda` (default `0.1`). Originally this was the *only* thing keeping
     `pred` from drifting unbounded and saturating `tanh` (see `TODO.md`); now that
     the output head is hard-capped (step 1), `loss_lambda` is no longer solely
     responsible for magnitude control — it still steers the model toward an
     MSE-like fit within the capped range, pulling it somewhat away from the raw
     profit objective.
   - The MSE baseline uses just `mean((pred - actual)^2)`.
6. **Backpropagation**: because the loss uses the smooth `tanh` position, forward
   and backward are the *same* function — there is **no straight-through estimator**
   and no fake gradient. Gradients stay alive for every prediction magnitude.
7. **Weight update**: AdamW optimizer adjusts parameters, with weight decay
   (`--weight-decay`, default `1e-3`) penalizing large weights directly — an
   additional, independent brake on runaway prediction magnitude alongside
   the output cap in step 1 (previously plain `Adam`, no weight decay).

**Evaluation**: eval reuses the exact same continuous position as training --
`signal = tanh(alpha * predicted_return)`, magnitude and all -- rather than
discretizing to a fixed-size long/flat/short action. This is the signal fed to
the profit metrics (`directional_accuracy` compares `sign(pred)` to
`sign(actual)` directly and is unaffected by this). So training and evaluation
optimize and report the exact same objective, with no separate decision rule
in between: the model's own confidence *is* the position size, both times.

Repeated for all batches across all epochs. Model weights saved to `results/gru_profit_aware.pt`. Loss and profit curves saved as PNGs to `results/`.

`train.py` saves the final-epoch weights to `--save` (default
`results/<loss>/gru_<loss>.pt`) alongside its loss/profit curves, without
evaluating the test set.

## 7. Evaluation Metrics

`utils/metrics.py`

After training, the model is evaluated on the **test set** (unseen data) using these metrics:

| Metric | Formula | What it measures | Good vs Bad |
|---|---|---|---|
| `directional_accuracy` | `(sign(pred) == sign(actual)).mean()` | How often the model gets the direction right (up vs down). | Bad: ~50% (random guess). Good: 55%+. Great: 60%+. |
| `cumulative_profit` | `sum(signal * return - cost)` | **Additive** total profit/loss. Each trade uses original capital. Simple but ignores compounding. | Bad: negative. Good: positive. Great: beats buy-and-hold. |
| `cumulative_profit_geometric` | `product(1 + signal * return - cost) - 1` | **Geometric** total profit/loss. Balance compounds: losses shrink next trade size, gains expand it. Reflects real-world trading. | Bad: negative. Good: positive. Great: beats buy-and-hold. |
| `sharpe_like` | `(mean / std) * sqrt(252)` | Risk-adjusted return — return per unit of volatility. | Bad: < 1. Good: 1–2. Great: 2–3. Suspicious: > 3 (likely overfitting). |

**Additive vs Geometric:** The additive model assumes unlimited capital per trade. The geometric model tracks your actual bankroll — if you lose 2% on day 1, you trade with 98% on day 2. Over many volatile trades, geometric returns are typically lower than additive returns (compounding works against you on losses). The geometric return is the more realistic metric.

