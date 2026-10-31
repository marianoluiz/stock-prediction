# Profit-aware GRU on the PSEi: system, evaluation design and results

This document is the reference for the thesis write-up. It describes what the
code does now (November 2026), how the model is tested, and what the latest
run found. When this document and the code disagree, the code is right.
Please report the mismatch so this document can be fixed.

Contents:

1. [What the study compares](#1-what-the-study-compares)
2. [Data](#2-data)
3. [Features and samples](#3-features-and-samples)
4. [Train / validation / test splits](#4-train--validation--test-splits)
5. [Model architecture](#5-model-architecture)
6. [Trading signal, costs and the two losses](#6-trading-signal-costs-and-the-two-losses)
7. [Training procedure](#7-training-procedure)
8. [Evaluation: metrics, baselines and statistical tests](#8-evaluation-metrics-baselines-and-statistical-tests)
9. [Results](#9-results)
10. [Limitations and caveats](#10-limitations-and-caveats)
11. [Reproducing every number and figure](#11-reproducing-every-number-and-figure)
12. [What changed from the earlier 29-stock design](#12-what-changed-from-the-earlier-29-stock-design)
13. [Code map](#13-code-map)

---

## 1. What the study compares

The principal comparison retains two GRU models forecasting the **next-day return of the PSEi
index** (`PSE:PSEI`). They are identical in every respect except the training
objective:

| | Baseline | Proposed |
|---|---|---|
| Name in code / outputs | `mse` | `profit-aware` |
| Loss | Mean squared error between predicted and actual return | Negative simulated net trading profit, after PSE transaction costs, plus 0.7 × MSE |
| Same architecture, features, data, splits, seed, optimiser, epochs | yes | yes |

So any difference between the two comes from the loss function alone. Both
models' predictions are turned into trading positions in the same way and
scored with the same cost model.

The three thesis objectives map onto the code as follows:

| Objective | Where it lives |
|---|---|
| 1. Profit-aware loss learning | `utils/trading.py::profit_aware_loss`, used in `training/train.py::run_epoch` |
| 2. Downside-risk-aware loss learning | `utils/trading.py::downside_second_moment`, optional gamma in `profit_aware_loss`, and `scripts/downside_compare.py` for matched validation-selected testing |
| 3. Trading friction (realistic costs) | `utils/trading.py::position_change_cost`: asymmetric PSE buy/sell costs, charged on every position change, in the loss **and** in every evaluation |

---

## 2. Data

| Item | Value |
|---|---|
| Series | PSEi composite index, `PSE:PSEI` |
| Source | TradingView, through the `tvDatafeed` library (Yahoo Finance does not carry the PSE) |
| Frequency | Daily OHLCV |
| Range used | 2018-01-03 to 2026-10-23 (2,151 trading days) |
| Cache | `data/PSE-PSEI_2018-01-01_latest.csv`. Once this file exists it is always reused, so every rerun sees the same data. Delete it to download fresh data. |
| Volume | **Always 0 for the index.** TradingView publishes no volume for the PSEi, so the volume feature (Section 3) is constant. Stocks do have volume. |

The PSEi fell over most of the sample. Calendar-year returns: 2018 −14.4%,
2019 +4.4%, 2020 −7.8%, 2021 −1.0%, 2022 −6.7%, 2023 −2.1%, 2024 −0.4%,
2025 −7.6%, 2026 to date −6.6%. This matters for interpreting results: in a
falling market, a model that simply stays short looks profitable without any
forecasting skill. Section 8 explains the checks for this.

---

## 3. Features and samples

Code: `utils/preprocessing.py::build_feature_frame`, `create_sequences`.

**Step 1, returns.** Close prices become simple daily returns,
`r_t = (P_t − P_{t−1}) / P_{t−1}`.

**Step 2, six feature channels per day:**

| # | Feature | Definition |
|---|---|---|
| 1 | `return` | `r_t`, the same day's return (also the prediction target, see below) |
| 2 | `return_lag1` | `r_{t−1}` |
| 3 | `return_lag5` | `r_{t−5}` |
| 4 | `return_lag10` | `r_{t−10}` |
| 5 | `volume_zscore` | `(V_t − mean_20) / (std_20 + 1e−8)` over a trailing 20-day window. **Always 0 for the PSEi** (no volume). |
| 6 | `volatility` | Standard deviation of the last 10 daily returns |

Every feature uses only past and current information. Rows without enough
history for the longest window are dropped. Features are **not** min-max or
z-score normalised. Returns are already on a small, comparable scale.

**Step 3, sliding windows.** Each sample is a window of the last **30 trading
days** of the six features (shape 30 × 6). The target is the **next day's**
return:

```
input  = features[i], ..., features[i+29]      (days t−29 ... t)
target = return[i+30]                          (day t+1)
```

The model never sees the target day's return in its input. After dropping
warm-up rows, the PSEi gives **2,102 samples**, with target dates from
2018-03-14 to 2026-10-23.

---

## 4. Train / validation / test splits

All splits are **chronological**: no shuffling, and training data always comes
before test data in time.

### 4.1 Main evaluation: expanding-window walk-forward, 4 folds

Code: `utils/preprocessing.py::expanding_walk_forward_folds`, with the
defaults `--folds 4 --eval-fraction 0.30`.

The last 30% of the samples (631 days) is the **evaluation region**. It is
cut into 4 back-to-back test windows. For each fold, a **fresh model** is
trained on everything before that fold's test window (an "expanding window":
later folds have more history), and then tested on that window only. Inside
each fold's history, the last 15% is the validation set.

| Fold | Train (target dates) | Validation | Test (out-of-sample) | Train / Val / Test samples |
|---|---|---|---|---|
| 0 | 2018-03-14 to 2023-04-28 | 2023-05-02 to 2024-03-22 | **2024-03-25 to 2024-11-13** | 1,250 / 221 / 157 |
| 1 | 2018-03-14 to 2023-11-13 | 2023-11-14 to 2024-11-13 | **2024-11-14 to 2025-07-10** | 1,383 / 245 / 157 |
| 2 | 2018-03-14 to 2024-06-03 | 2024-06-04 to 2025-07-10 | **2025-07-11 to 2026-03-02** | 1,517 / 268 / 157 |
| 3 | 2018-03-14 to 2024-12-13 | 2024-12-16 to 2026-03-02 | **2026-03-03 to 2026-10-23** | 1,650 / 292 / 160 |

The four test windows don't overlap, and together they cover 2024-03-25 to
2026-10-23 continuously (631 trading days). That is why the equity-curve
figure can join them end to end.

**Why walk-forward and not a single split?** One symbol with one test window
gives exactly one number, and one window is one market regime, which could
easily be luck. Four windows give four nonoverlapping out-of-time results and
test whether the result holds across regimes. Four folds (about 7.5 months
each) was chosen over six (about 5 months each) because shorter windows make
returns and Sharpe ratios too noisy.

A built-in check (an `assert` in the code) confirms that, for every fold, the
last training date is before the first test date, so there is no look-ahead.

### 4.2 Secondary: single 70 / 15 / 15 split

Used by `main.py` (the worked example with a full trade log), `scripts/train.py`,
`scripts/evaluate.py` and `scripts/predict.py`.

| Part | Dates | Samples |
|---|---|---|
| Train (70%) | 2018-03-14 to 2024-03-22 | 1,471 |
| Validation (15%) | 2024-03-25 to 2025-07-11 | 315 |
| Test (15%) | 2025-07-14 to 2026-10-23 | 316 |

These cut points line up with the walk-forward folds. The single-split
training set is exactly fold 0's training-plus-validation history (ending
2024-03-22). The single-split test window starts one trading day after fold
2's, so it covers almost the same days as folds 2 and 3 together.

### 4.3 What the validation set is used for

With the default settings (`--early-stop-patience 0`), the validation set is
only **monitored**. Its loss and profit are printed every epoch, but it isn't
used to pick weights or stop early. The model after the final (50th) epoch is
the one tested. Early stopping exists as an option but made results worse in
earlier experiments, so it is off.

---

## 5. Model architecture

Code: `models/gru_model.py::GRUReturnPredictor`.

```
input  [batch, 30 days, 6 features]
  -> GRU layer 1 (64 hidden units)
  -> dropout 0.2
  -> GRU layer 2 (64 hidden units)
  -> hidden state of the last day only        [batch, 64]
  -> LayerNorm(64)
  -> Linear(64 -> 1)                          raw output
  -> output_scale × tanh(raw)                 predicted next-day return r̂
```

| Setting | Value |
|---|---|
| GRU layers | 2 |
| Hidden units | 64 |
| Dropout between GRU layers | 0.2 |
| Trainable parameters | 38,977 |
| Output bound | `output_scale = 5 / alpha` = 5 standard deviations of the training returns (about ±6.6% per day for the PSEi) |

The bounded output head (`output_scale × tanh`) constrains forecast magnitude.
The position magnitude is simulated exposure, not calibrated confidence.
Neither the cap nor MSE calibration guarantees nonsaturated gradients or
varying direction; `tanh` can still saturate within the cap. The position
diagnostics are therefore necessary.

---

## 6. Trading signal, costs and the two losses

Code: `utils/trading.py`.

### 6.1 From prediction to position

```
alpha   = 1 / std(training-set returns)              recalibrated for every fold
pos_t   = tanh(alpha × r̂_t)                         position in (−1, +1)
```

`pos_t = +1` means fully long, `−1` fully short, and `0` flat. A one-standard-
deviation predicted move gives a position of `tanh(1) ≈ 0.76`. Because `alpha`
comes from each fold's own training data, the position scale adapts to the
index's volatility (for the PSEi, `alpha` is about 72-78, since daily return
std ≈ 1.3%).

The position is **continuous**. The model can hold a small position when
unsure. The same continuous position is used in the loss and when scoring
the test set.

### 6.2 Transaction costs (PSE, asymmetric)

| Item | Buy | Sell |
|---|---|---|
| Broker commission 0.25% | yes | yes |
| 12% VAT on commission | yes | yes |
| PSE transaction fee + SCCP clearing fee | yes | yes |
| Stock transaction tax 0.1% | no | yes |
| **Total per unit traded** | **0.296%** | **0.396%** |

Costs are charged on every **change** in position:

```
delta_t = pos_t − pos_{t−1}
cost_t  = 0.00296 × max(delta_t, 0)  +  0.00396 × max(−delta_t, 0)
```

An increase (opening a long, or covering a short) is a purchase. A decrease
(selling a long, or opening a short) is a sale. Holding the same position
costs nothing. Each test window starts flat (`pos = 0`), so the first day pays
the cost of opening the position. One function, `position_change_cost`, is
used everywhere: the loss, every metric, the trade log, the TA baselines and
the web demo.

### 6.3 Daily net profit

```
profit_t = pos_t × r_t − cost_t
```

### 6.4 The two losses

**Baseline (MSE):**

```
L_MSE = mean( (r̂_t − r_t)^2 )
```

**Proposed (profit-aware):**

```
L_PA = − mean( profit_t )  +  lambda × mean( (r̂_t − r_t)^2 ),      lambda = 0.7
```

- The first term rewards the model for **net profit after costs**. Because
  the cost term depends on `pos_t − pos_{t−1}`, the model is penalised for
  flipping its position too often (Objective 3).
- The second term penalizes return-prediction error. The supplementary
  matched lambda sweep shows smaller forecast magnitudes and exposure at
  stronger calibration, but all five tested weights retained one market side
  within each fold. The inherited .7 setting is a control, not an optimal
  weight demonstrated by that test sweep. The new downside term is separate.
- `tanh` is used in both the forward and the backward pass (no
  straight-through `sign()` trick). Some older docs (`docs/profit-loss-saturation.md`,
  `docs/plan-directional-smooth.md`) describe a straight-through estimator
  that no longer exists. Don't copy those mechanism descriptions.

---

## 7. Training procedure

Code: `training/train.py::fit`, `run_epoch`.

| Setting | Value |
|---|---|
| Optimiser | AdamW, learning rate 0.001, weight decay 0.001 |
| Batch size | 64 |
| Epochs | 50, no early stopping, final-epoch weights used |
| Batch order | **Chronological, not shuffled.** The cost term needs yesterday's position, so batches are processed in date order, and the last position of one batch is carried into the next. |
| Random seed | 42 (`torch.manual_seed` before each model is created). Every fold and both losses start from the same initial weights. |
| Hardware | CUDA GPU when available (an RTX 3050 here), otherwise CPU |
| Training time | About 10-15 seconds per model on the GPU for the PSEi; the whole `run_thesis.py` takes about 5 minutes |

For each fold: compute `alpha` from that fold's training returns, build a
fresh model with seed 42, train 50 epochs, then run the test window once
with no gradient updates.

---

## 8. Evaluation: metrics, baselines and statistical tests

### 8.1 Metrics (per test window)

Code: `utils/metrics.py`.

| Metric | Formula | Notes |
|---|---|---|
| Directional accuracy | share of days with `sign(r̂_t) = sign(r_t)` | 50% = coin flip. The PSEi has no zero-return days, so 50% is the right chance level. |
| Cumulative return (additive) | `Σ profit_t` | Each trade sized on the starting capital |
| **Geometric return** (main profit metric) | `Π (1 + profit_t) − 1` | Compounding: each trade is sized on the current balance |
| Sharpe-like ratio | `mean(profit_t) / std(profit_t) × √252` | Annualised, zero risk-free rate |
| MSE, MAE, RMSE | error of `r̂_t` vs `r_t` | Forecasting error, not trading |
| Mean position | `mean(pos_t)` | −1 = always fully short, +1 = always fully long |
| Mean \|position\| | `mean(|pos_t|)` | How big the bets are on average |
| Turnover | `mean(|pos_t − pos_{t−1}|)` | How much the position changes per day |
| Short days | share of days with `pos_t < 0` | |
| Degenerate run | the position never changes sign in the whole test window | The model learned no day-to-day signal. It is a constant long or short bet, whatever its profit. |

**How to read profit against position behaviour.** If a model's mean
position is close to ± its mean |position|, and the run is degenerate, then
its profit is just the market's drift times a constant bet size. For example,
a constant small short earns money when the index falls. That isn't
forecasting. The summary file reports all these columns so the claim can
be checked.

### 8.2 Baselines (scored on the same days, same costs)

Code: `utils/baselines.py`. Each baseline uses only information available
before the day it trades (its signal is shifted by one day).

| Baseline | Rule | Cost charged? |
|---|---|---|
| Buy & hold | Always fully long | **No** (pure market reference) |
| SMA crossover (trend) | Long if 10-day SMA > 50-day SMA, else short | Yes |
| RSI (mean reversion) | Long if RSI(14) < 30, short if > 70, flat otherwise | Yes |
| MACD (momentum) | Long if MACD(12,26) > its 9-day signal line, else short | Yes |

The RSI rule is flat on most days, and directional accuracy counts flat days
as "wrong". So **RSI's directional accuracy (a few percent) is meaningless**.
Use its returns and Sharpe only.

### 8.3 Statistical tests

Code: `scripts/significance.py`. Two levels, since there is only one series:

1. **Daily paired test (main test).** For each of the 631 out-of-sample days,
   take profit-aware's net return minus MSE's net return. Test whether the
   mean difference is zero, using a **Newey-West (HAC) standard error**.
   That correction is needed because positions persist over days, so the
   daily differences are autocorrelated and a plain t-test would be
   over-confident. The same test is run on the daily directional "hit"
   (correct direction = 1, wrong = 0).
2. **Hit rate vs 50% (binomial test)** for each model: is its directional
   accuracy over all 631 days distinguishable from a coin flip?
3. **Fold-level sign test.** On how many of the 4 folds does profit-aware
   beat MSE? With only 4 folds, even 4/4 gives p = 0.125, so this test can
   never reach significance. It is reported for transparency only.

A difference is called significant at p < 0.05 (two-sided).

### 8.4 Cost ablation

Code: `scripts/cost_comparison.py --walk-forward --folds 4`. On the same 4
folds, three models are trained:

- `mse`: MSE baseline
- `pa_nocost`: profit-aware loss, trained with zero cost in the loss
- `pa_cost`: profit-aware loss, trained with the real PSE costs in the loss (the proposed model)

Each model's test positions are then re-scored twice: at zero cost and at
real PSE costs. This separates two questions. How much do costs reduce
returns ("cost drag")? And does putting the cost **into the loss** make the
model trade more sensibly than ignoring it during training?

---

## 9. Results

Run of 2026-10-25, all defaults (4-fold walk-forward, 50 epochs,
lambda 0.7, PSE costs, seed 42). Source files in `results/`:
`benchmark_summary_psei_20261025_203842.csv` / `_daily.csv` /
`_summary.txt` / `_charts/`, `significance.csv`, `cost_comparison*`.

### 9.1 Headline numbers (mean over the 4 test folds)

| Model / policy | Dir. accuracy | Geometric return | Sharpe-like | RMSE |
|---|---|---|---|---|
| MSE baseline | 49.6% | −0.33% | −0.28 | 0.01092 |
| **Profit-aware** | 50.7% | **+0.95%** | **+0.27** | 0.01199 |
| Buy & hold (no cost) | | −4.40% | | |
| SMA crossover | 52.4% | −2.10% | −0.07 | |
| RSI | (meaningless, see 8.2) | −0.91% | −0.23 | |
| MACD | 49.3% | −7.92% | −0.70 | |

Over the whole 631-day out-of-sample period (folds joined end to end, see
`equity_curve.png`), 1 PHP became 1.038 with profit-aware, 0.987 with MSE,
and 0.832 with buy & hold.

### 9.2 Per fold

| Fold | Test window | Buy & hold | MSE | Profit-aware | SMA | RSI | MACD |
|---|---|---|---|---|---|---|---|
| 0 | 2024-03-25 to 2024-11-13 | −2.44% | −0.03% | +0.33% | +3.51% | −3.03% | +4.24% |
| 1 | 2024-11-14 to 2025-07-10 | −3.74% | +0.14% | +0.81% | −13.36% | +0.67% | −23.96% |
| 2 | 2025-07-11 to 2026-03-02 | −0.56% | −0.15% | −0.32% | +9.87% | −1.18% | −8.51% |
| 3 | 2026-03-03 to 2026-10-23 | −10.86% | −1.29% | +2.97% | −8.41% | −0.08% | −3.45% |

Win counts for profit-aware: it beats MSE on geometric return in 3/4 folds,
on Sharpe in 4/4, and on directional accuracy in 1/4. It beats buy & hold
4/4, RSI 4/4, MACD 3/4 and SMA 2/4.

### 9.3 Statistical significance

| Test | n | Profit-aware minus MSE | 95% CI | p |
|---|---|---|---|---|
| Daily net return, Newey-West (annualised) | 631 days | +2.2% per year | −4.8% to +9.2% | 0.53 |
| Daily directional hit, Newey-West | 631 days | +1.1 pp | −3.8 to +6.0 pp | 0.66 |
| MSE hit rate vs 50% (binomial) | 631 days | 49.6% | 45.6% to 53.6% | 0.87 |
| Profit-aware hit rate vs 50% (binomial) | 631 days | 50.7% | 46.7% to 54.7% | 0.75 |
| Folds won, geometric return (sign test) | 4 folds | 3/4 | | 0.63 |
| Folds won, Sharpe (sign test) | 4 folds | 4/4 | | 0.13 |

**None of the differences is statistically significant.** Neither model's
directional accuracy is distinguishable from a coin flip.

### 9.4 What the models actually do: position behaviour

This is the most important result for interpreting everything above.

| | Mean position | Mean \|position\| | Days short | Degenerate folds |
|---|---|---|---|---|
| MSE | −0.001 (folds 0-1: about −0.07 to −0.14; folds 2-3: about +0.10) | 0.105 | 50% (2 folds all short, 2 folds all long) | 4/4 |
| Profit-aware | −0.377 | 0.377 | **100%** | **4/4** |

- **Profit-aware is short on every single test day in every fold**, at a
  nearly fixed size: its position stays between −0.30 and −0.44 and moves by
  only about ±0.003 within a fold. It never changes side.
- Its profit is the same as a **constant short of that size**, to within
  0.05 percentage points in every fold (for example, fold 3: +2.97% for the
  model vs +2.99% for a fixed −0.31 position). Across all 631 days, its daily
  net return correlates **0.99** with simply shorting the index.
- Its directional accuracy in each fold is **exactly the share of down days**
  in that window (50.3%, 50.3%, 49.7%, 52.5%). That is what an always-short
  position scores, whatever it predicts.
- MSE does the same thing with a much smaller bet (about ±0.1). Its sign is
  fixed within each fold, short in folds 0-1 and long in folds 2-3.

**Interpretation.** Under this setup, neither GRU learned a day-to-day
forecast of the PSEi. Both converge to a constant position: in effect, an
estimate of the training period's average drift. The PSEi's training history
(2018 onwards) has a slightly negative average daily return. The profit-aware
loss rewards acting on that drift, so it takes a sizeable short position.
The MSE loss only needs a forecast close to the (near-zero) mean, so its
position is tiny. Profit-aware "wins" because the test period (2024-2026)
kept falling (buy & hold −16.8%), so a constant short paid off. In a rising
market it would lose. In fold 2, where the index was nearly flat (−0.56%),
the short already lost money after costs (−0.32%).

So the correct conclusion is **not** "the profit-aware loss trades better".
It is: "the profit-aware loss turns a weak drift estimate into a larger
directional bet. On the PSEi in 2024-2026 that bet happened to be right,
but the model shows no forecasting skill and the advantage is not
statistically significant."

This replicates an earlier PSEi experiment (5 walk-forward folds,
lambda ∈ {0, 0.3, 0.7, 1.0, 2.0}, seeds 42-44; summary preserved in git at
`results/psei_lambda_seed/SUMMARY.md`, commit `c671119`). There, profit-aware
was a constant short in 15/15 runs at every lambda. Raising lambda only
shrank the bet size, and changing the seed changed almost nothing.

### 9.5 Cost ablation: does putting the cost into the loss help?

Same 4 folds (`cost_comparison_summary.txt`, `cost_comparison.png`). Means over
the folds:

| Model | Geo return, scored without cost | Geo return, scored with PSE cost | Cost drag | Turnover / day |
|---|---|---|---|---|
| MSE baseline | −0.15% | −0.33% | 0.18 pp | 0.0033 |
| Profit-aware, cost **not** in loss | +0.79% | **−0.57%** | **1.36 pp** | **0.0248** |
| Profit-aware, cost in loss (proposed) | +1.13% | **+0.95%** | 0.18 pp | 0.0030 |

Scored with real costs, the cost-in-loss model beats the no-cost-in-loss
model in **4/4 folds**.

This is the clearest positive finding. A profit-aware model trained
**without** costs trades about **8 times as much**. It earns slightly more
before costs (it shows some gross Sharpe in folds 0 and 2), but PSE costs
eat all of that and more, leaving it below MSE. Putting the asymmetric PSE
cost **into the loss** (Objective 3) brings turnover down to MSE's level and
removes almost all of the cost drag. The caveat from 9.4 still applies: the
low-turnover model gets there by holding one position, not by trading
selectively. So this shows the cost term works as a **turnover penalty**. It
doesn't show that the model learned when it is worth trading.

### 9.6 Worked example (single 70/15/15 split, `main.py`)

Test window 2025-07-14 to 2026-10-23 (316 days), starting capital 100,000 PHP:

| | MSE | Profit-aware |
|---|---|---|
| Directional accuracy | 49.05% | 50.95% |
| Geometric return | −1.15% (−1,149 PHP) | +2.70% (+2,697 PHP) |
| Sharpe-like | −0.59 | +0.45 |
| RMSE | 0.01097 | 0.01144 |
| Days long / short | 316 long / 0 short | 0 long / 316 short |

The same pattern on a different split: each model holds one direction for the
entire test window. The printed trade log's "Signal" column shows only the
**direction** (+1 / −1). The actual position size is the continuous
`tanh(alpha × r̂)` value, and the P&L columns use that size.

### 9.7 Summary of findings for the thesis

1. **Objective 1 (profit-aware loss).** Implemented and working as designed:
   the model optimises after-cost profit. On the PSEi 2024-2026 it produced
   higher returns and Sharpe than MSE (+0.95% vs −0.33% mean per fold;
   +3.8% vs −1.3% over the whole period). But it did this by holding a
   **constant short position**, not by forecasting, and the difference is not
   statistically significant (p = 0.53).
2. **Objective 2 (downside-risk-aware learning).** The additional zero-target
   squared-negative-net-return term is implemented. The matched CPU control,
   validation scores, selected weights, held-out downside risk, exposure and
   uncertainty are in `results/downside/`. Use these outputs for this objective;
   neither old directional results nor lambda-sweep results answer this ablation.
3. **Objective 3 (trading friction).** The cost term clearly does its job as a
   turnover penalty. Without it, the same loss trades 8× more and loses money
   after PSE costs. With it, cost drag falls from 1.36 pp to 0.18 pp, and it
   wins 4/4 folds against the no-cost version.
4. **Overall.** On a single, slowly falling index with daily data and these
   features, a GRU of this size finds no exploitable day-to-day pattern under
   either loss. The profit-aware loss changes the model's **behaviour** (bet
   size, direction, turnover) far more than its **skill**. That is a
   legitimate and defensible thesis result if it is reported as such. It
   should not be presented as "the profit-aware GRU outperforms the
   baseline".

---

## 10. Limitations and caveats

- **One series, one market regime.** Everything is measured on the PSEi
  between 2024 and 2026, a period when the index mostly fell. A constant short
  wins in that environment. Results can't be generalised to rising markets
  or to other indices.
- **Small effective sample.** 631 test days, but only 4 separately fitted model
  fits. The fold-level test can't reach significance even at 4/4.
- **Degenerate models.** Both models hold one direction for every test
  window (Section 9.4). Profit numbers therefore reflect market drift times
  bet size, not forecasting.
- **No volume information** for the index (the feature is constant 0).
- **The validation set is unused for model selection** (no early stopping).
  The final-epoch model is always used, and the most recent 15% of each
  fold's history is never trained on.
- **One seed per run** (42). An earlier experiment with seeds 42-44 found the
  seed barely changes the outcome (Section 9.4, last paragraph). That
  experiment isn't part of this run.
- **Fixed hyperparameters** (30-day window, 64 hidden units, 50 epochs,
  lambda 0.7) were not tuned on the PSEi. Lambda was explored separately
  (same earlier experiment): higher lambda only shrinks the constant bet.
- **Costs are proportional only.** No minimum broker fee, bid-ask spread,
  slippage, or borrowing cost for short positions (short selling is very
  restricted on the PSE in practice). The short positions the profit-aware
  model takes would be hard to implement on the actual PSEi.
- **Buy & hold is shown without costs**, so it is slightly flattered relative
  to the strategies (one purchase of about 0.3% is ignored).
- **GPU non-determinism** (cuDNN) can change the last digits between reruns.

---

## 11. Reproducing every number and figure

```
.venv/Scripts/python.exe run_thesis.py
```

This runs the six steps below with 50 epochs, lambda .7, PSE costs, seed 42
and four folds. The downside ablation runs matched CPU conditions and adds
training time. Each step can also run on its own:

| Step | Command | Output in `results/` |
|---|---|---|
| 1 | `benchmark.py --walk-forward --folds 4 --charts` | `benchmark_summary_psei_<time>.csv` (one row per fold × loss), `..._daily.csv` (every test day's position and net return for every model and baseline), `..._summary.txt` (all tables), `..._charts/` (figures) |
| 2 | `scripts/significance.py` | `significance.csv`, `significance.png` |
| 3 | `scripts/cost_comparison.py --walk-forward --folds 4` | `cost_comparison.csv`, `cost_comparison_summary.txt`, `cost_comparison.png` |
| 4 | `main.py --compare --trade-log --symbol PSE:PSEI` | `mse/` and `profit_aware/`: weights, loss and profit curves on the single split; the trade log is printed to the console |
| 5 | `scripts/downside_compare.py --train` | `downside/`: selection manifest, validation scores, fold/daily results, summary, paired tests and chart |
| 6 | `scripts/build_paper_figures.py --latest` | Chapter 1 and 3 diagrams; saved-result Chapter 4 figures and tables, including supplementary lambda outputs |

Figures in `..._charts/`:

| File | Shows |
|---|---|
| `equity_curve.png` | Growth of 1 PHP over 2024-03 to 2026-10 for both GRUs, buy & hold, SMA, RSI and MACD. Fold boundaries are dotted. **Best single figure for the thesis.** |
| `buy_hold_comparison.png` | Geometric return per fold: MSE vs profit-aware vs buy & hold |
| `per_fold_geo_diff.png` | Profit-aware minus MSE, per fold, with that fold's buy & hold |
| `strategy_comparison.png` | Mean return and Sharpe over the folds: both GRUs vs all baselines |
| `profit_geo_comparison.png`, `profit_comparison.png`, `sharpe_comparison.png`, `directional_acc_comparison.png`, `rmse_comparison.png` | One metric each, per fold plus the mean |

Notes:

- Rebuild the summary and figures from a saved CSV without retraining:
  `benchmark.py --from-csv results/benchmark_summary_psei_<time>.csv --charts`.
- The GRU uses NVIDIA cuDNN, which isn't bit-for-bit deterministic. Reruns
  with the same seed can differ slightly in the last digits, and occasionally
  more for a model that's on the edge between a long and a short bet.
- On this machine, any script that ran a GRU on the GPU crashes **after** it
  has finished and written all its output (Windows exit code 0xC0000409).
  The results are complete and valid. `run_thesis.py` knows to ignore that
  one exit code.
- The previous 29-stock design can still be run: `run_thesis.py --stocks`.

---

## 12. What changed from the earlier 29-stock design

The study was first designed around the 29 PSEi constituent stocks, with the
index only as a side reference. It is now **index-first**. If any thesis
chapter still describes the stock design, these are the things to update:

| Topic | Earlier (stocks) | Now (index) |
|---|---|---|
| Sample | 29 PSEi constituents (+ PSEi as a reference, kept out of averages) | The PSEi index only |
| Test design | One 70/15/15 split per stock | 4-fold expanding-window walk-forward (Section 4.1). The single split is kept only for the worked example. |
| "n" for statistics | 29 stocks, paired t-test across stocks | 631 test days, daily paired test with Newey-West SE, plus a 4-fold sign test |
| Main figures | Means across stocks, per-stock difference chart | Per-fold bars, per-fold difference chart, stitched equity curve |
| Cost ablation | 6 stocks, single split | PSEi, same 4 folds |
| Worked example | `PSE:ICT` | `PSE:PSEI` |
| Volume feature | Real volume z-score | Constant 0 (the index has no volume) |

`docs/methodology_chapter3.md` still describes the 29-stock sample in
Section 3.1 (the paragraph on the sample, and the one on the 70/15/15 split
and walk-forward). It should be updated using the table above.
`docs/algorithm.md` (Steps 0-11) is still accurate for the model, signal, cost
and loss.

---

## 13. Code map

| File | Role |
|---|---|
| `run_thesis.py` | Runs every thesis step (Section 11) |
| `benchmark.py` | Trains and tests both losses on each fold; writes the results CSV, the daily CSV, the summary and the charts |
| `main.py` | One symbol, single split, both losses (`--compare`), with a trade log |
| `scripts/significance.py` | Statistical tests (Section 8.3) |
| `scripts/cost_comparison.py` | Cost ablation (Section 8.4) |
| `scripts/plot_loss_comparison.py` | All benchmark figures |
| `scripts/train.py`, `evaluate.py`, `predict.py` | Train only / evaluate a saved model / forecast the next N days |
| `webapp.py` | Streamlit live demo (`streamlit run webapp.py`); PSEi is selected by default |
| `models/gru_model.py` | The GRU network (Section 5) |
| `training/train.py` | Training loop and per-epoch metrics (Section 7) |
| `utils/preprocessing.py` | Data download and caching, returns, features, windows, splits (Sections 2-4) |
| `utils/pipeline.py` | Shared data preparation used by every entry point |
| `utils/trading.py` | Position, cost model, profit-aware loss (Section 6) |
| `utils/metrics.py` | All metrics and the trade log (Section 8.1) |
| `utils/baselines.py` | SMA, RSI and MACD strategies (Section 8.2) |
| `docs/algorithm.md` | The original thesis algorithm, Steps 0-11 |
| `docs/arguments.md` | Every command-line argument and its default |

## November 2026: downside-loss extension (SOP 2)

The original principal benchmark and cost ablation above evaluate the base
profit-aware objective at gamma=0. Their numerical results do not describe the
new extension. The historical lambda experiment is supplementary.

The new loss is `L_PA + gamma*mean(relu(-net_return)**2)` with lambda .7,
the unchanged continuous positions and the shared asymmetric cost rule.
The fixed protocol, scale convention, positive candidate set, common
validation utility and held-out reporting rules are in
[downside-protocol.md](downside-protocol.md). It selects a weight per fold
before test evaluation and never selects on test return. Run
`scripts/downside_compare.py --train`; omit `--train` for saved-result reporting.
All new comparisons use matched CPU runs. Lower downside must be considered
alongside net return, exposure, turnover, cash and validation-static controls.
The risk term is a new mechanism in this implementation, not a claim that
lower partial moments or cost-aware learning are newly invented theories.

The actual matched CPU run selected kappa 1, .25, .25 and .25 for folds 0–3.
Arithmetic fold means, control versus downside: daily downside deviation
.2885% versus .0390%, mean absolute position .3832 versus .0486, geometric
return +.9688% versus +.0129%, and Sharpe-like ratio .2729 versus .1286.
All four folds remained on one market side; the downside model was short in
three and long in one. The paired daily squared-negative-net-return change
was lower (exploratory HAC p<.001); the net-return difference was uncertain
(p=.641). Validation-static controls produced similar geometric outcomes
(+.9892% and +.0225%), so the result supports a lower-exposure risk–return
tradeoff, not an improvement in timing or profitability. Full precision,
all fold outcomes, validation scores and the selection manifest are retained
in `results/downside/`.
