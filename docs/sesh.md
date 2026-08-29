# Session Problem: Why the Results Look Great But Prove Nothing

## Context / What we were doing

We restructured the profit-aware training to:

1. Use a smooth `tanh` trading position `pos = tanh(α·r̂)` (no Straight-Through Estimator hack),
2. Focus on **bidirectional (directional) accuracy** at evaluation,
3. Simulate buy/sell profit after training using `sign(pred)`.

The thesis claim we were trying to support:

> A model with higher directional accuracy also realizes higher simulated P&L
> (i.e. better direction prediction -> better profit).

We ran a comparison across grouped tickers and it *looked* like the profit-aware
model started winning — higher directional accuracy AND higher profit. But the
numbers are not what they seem.

---

## The problem, stated plainly

On drift-heavy stocks, both models collapse to a **constant signal** ("always long"
or "always short"). Because of this:

- **Directional accuracy is NOT learned skill** — it equals the stock's test-window
  "up-day rate" (fraction of days the return was positive).
- **P&L is NOT trading skill** — it equals the buy-and-hold drift of the test window,
  signed by whatever constant side the model froze on.
- Therefore **"higher accuracy" and "higher profit" move together not because of
  cause-and-effect** (accuracy -> profit) but because **both are the same thing:
  the drift direction**. That is circular; it will not survive a thesis defense.

---

## The evidence (directly from our saved models)

We loaded the actual trained models and inspected their test predictions.

### 1. Predictions are tiny or huge, but the sign collapses to a constant

- **NVDA profit-aware:** `sign = +1` for **321/321** test days (always long).
- **NVDA MSE:** `sign = +1` for **321/321** (always long).
- **MSFT profit-aware:** 314 long / 8 short; **MSE:** 319 / 3 long-short.
- **intc profit-aware:** `mean=+4.87`, `sign = +1` for **322/322** (always long).
- **intc MSE:** `mean=-0.0038`, `sign = -1` for **318/322** (always short).
- **amzn profit-aware:** `sign = +1` for **322/322**; **MSE:** `+1` 322/322.

### 2. Directional accuracy = the stock's up-day rate (not skill)

Computed test-window up-day rates vs. reported accuracy:

| Stock | Up-day rate | Reported accuracy |
|---|---|---|
| NVDA | 0.5358 | 0.5358 |
| GOOG | 0.5171 | 0.5171 |
| MSFT | 0.5217 | 0.5217 |
| amzn | 0.5248 | 0.5248 |
| intc | 0.5031 | 0.5031 |

A model that just says "long every day" scores exactly the up-day rate. The models
add **zero** per-day predictive information.

### 3. P&L = drift, signed by the frozen side

- intc test window is net-up: always-long gives `+1.79`, always-short gives `-1.79`.
- profit-aware went long -> `+1.79` (win); MSE went short -> `-1.79` (lose).

So "profit-aware beat MSE" reduces to: *PA landed long, MSE landed short, and the
test window happened to go up.* Pure side-luck, not forecasting.

### 4. amzn produces BYTE-IDENTICAL metrics for both models

`0.5248/0.5248` accuracy, `0.288709/0.288709` cumulative, identical Sharpe.
Two randomly-initialized, differently-trained models making **exactly the same
trades every day** is only possible if both are the same constant signal.
It is the degenerate collapse, not a coincidence.

---

## Why the MSE "baseline" differs between runs (and it's NOT me, NOT a cache bug)

The MSE baseline gives **different results every run** for the same stock:

| Stock | run-result-w-llambda | run-result (λ=0) | no-plusmse |
|---|---|---|---|
| nvda | +71,229 | +71,229 | **-71,429** |
| GOOG | +85,723 | +86,509 | **-77,202** |
| msft | -55,969 | +609 | **-21,113** |
| intc | -179,220 | -179,220 | **-161,563** |
| amzn | +28,871 | -29,071 | +28,871 |

Reason: each `--compare` run builds a **brand-new GRUReturnPredictor with random
weights** (`main.py:27-31`). Neural nets re-trained from random init land in
different local minima every time, so the baseline is a moving target.

Crucially: **loss_lambda does not touch the MSE model at all** (the MSE baseline is
always just `mean((pred-actual)^2)`). The differences are pure run-to-run randomness.

Also note the profit-aware model's reported "MSE" jumped to ~22-36 in the
`no-plusmse` run — that is the **pure-profit saturation** returning (predictions
blow up to ~±5 and the model freezes). It is not a bug; it is the expected
behavior when you drop the MSE calibration term.

---

## Why "accuracy is up AND profit is up" is not a win

Both metrics are the same drift bet dressed up two ways:

- Accuracy  = did the model's constant side match the up-day rate?
- Profit   = did the constant side match the cumulative drift?

When they agree, it is **because they are the same number**, not because accuracy
caused profit. This is circular reasoning and cannot be claimed as evidence of
directional forecasting.

---

## The root cause (unsolvable by tuning α/λ alone)

With `sign(pred)` + ε=0 (full long/short, no hold) evaluation:

- Any model whose mean prediction is positive -> `sign` says long every day.
- The profit loss actively rewards persistent drift-harvesting positions.
- So the "optimal" strategy under this eval *is* always-long on a rising stock.

No amount of tuning `alpha` or `loss_lambda` changes this, because it is inherent
to the eval rule + the drift incentive. We have verified this multiple times.

---

## What we need to decide (options)

### Option 1 — Keep it, reframe the claim (simplest, honest)
Keep the current loss + `sign` eval, but stop claiming the model "foresees
direction." Frame the thesis as:

> The profit-aware loss, by directly maximizing net profit (incl. transaction
> costs), learns persistent positioning. On trend stocks it captured the realized
> drift while MSE's noisy sign flips bled transaction costs.

Defensible, but it is NOT a directional-forecasting claim. Must not be labeled as
one.

### Option 2 — Make direction real (only path to a genuine accuracy->profit claim)
Change the evaluation so the signal can actually **flip** day-to-day (e.g. a
hold/threshold dead-zone, or a position that reflects prediction magnitude). Then
directional accuracy measures real per-day skill and profit comes from getting
those flips right. **But** this changes the eval rule (currently ε=0, buy/sell only)
that the thesis text specifies.

### Option 3 — Report against a baseline so the drift is visible (recommended)
Keep everything, but report accuracy and profit **relative to an always-long /
buy-and-hold baseline** per stock. Show the model's **excess** accuracy over
always-long and **excess** profit over buy-and-hold. This makes the drift explicit
and keeps the comparison honest, while remaining a working experiment.

---

## Recommendation

**Option 3 + parts of Option 1** for a master's thesis:

- Keep the current implementation (λ default 0, pure profit loss).
- Always report against always-long / buy-and-hold so a reader can *see* the drift.
- Frame the claim as "profit-aware captures realized drift on trend stocks" rather
  than "the model predicts day-to-day direction."

Only pursue Option 2 if the thesis specifically needs a true directional-forecasting
claim — which the current data does not support.

---

## Open questions before finalizing the plan

1. **Can the eval rule change** (add a hold/threshold so `sign` can flip), or must it
   stay `sign(pred)` buy/sell only (per `arguments.md` and Q3=ε=0)?
2. **Is it acceptable** that the thesis claim be "profit-aware captures realized
   drift / persistent positioning" (honest, defensible) rather than "the model
   foresees day-to-day direction" (unsupported by the data)?
