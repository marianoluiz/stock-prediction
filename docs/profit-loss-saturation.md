# Why the Profit-Aware Loss Goes Flat (Gradient Saturation)

This document explains, from the ground up, why the profit-aware training runs
show flat loss/profit curves after the first epoch, and how a **hybrid loss**
fixes it. It was written so you can understand and explain the problem at your
thesis defense.

---

## 1. The profit-aware loss

For each day `t` the model predicts a return `r̂_t`. We turn that into a trading
position with:

```
signal_t = sign(r̂_t) = +1 (long)  if r̂_t > 0
                        -1 (short) if r̂_t < 0
```

Switching position costs a transaction fee `c = 0.001` per unit of change:

```
cost_t = c * |signal_t - signal_{t-1}|
```

The day's **net profit** is:

```
profit_t = signal_t * r_t - cost_t
```

and the profit-aware loss is the *negative average* profit (minimizing loss =
maximizing profit):

```
L = -mean(profit_t) = -mean(signal_t * r_t - cost_t)
```

### The loss is bounded — flat can be partly normal

Because `signal` can only be `±1`, each day's profit is bounded by the actual
return: `profit_t = signal_t * r_t - cost_t ≤ |r_t|`. So:

```
L ≥ -mean(|r_t|)
```

The loss can never go below roughly the average absolute daily return
(≈ 0.01–0.02 for a typical stock, i.e. 1–2% per day). A loss curve that stops
decreasing is **not automatically a bug** — once the model is getting the sign
right most of the time, there is nothing left to squeeze. But it becomes a
problem when it stops *in epoch 1*, which is what we observed.

---

## 2. The sign function has no gradient

`signal = sign(r̂)` is flat almost everywhere — its derivative is 0 for every
`r̂ ≠ 0`. If the loss used `sign` directly, backpropagation would give
`dL/d r̂ = 0` and the model would **never learn anything**. That is the whole
reason `utils/trading.py` uses a **Straight-Through Estimator (STE)**:

- **Forward pass** uses the real binary signal `sign(r̂)` (so the loss is
  exactly how the model trades).
- **Backward pass** pretends the derivative is the smooth slope of
  `tanh(α · r̂)`:

```
fake gradient = α * (1 - tanh(α * r̂)²)
```

This is a hack so gradients flow "as if" the signal were soft. `α` is the
`--alpha` flag (the `tanh` sharpness).

---

## 3. Where it breaks: the tanh slope dies when |r̂| grows

Look at the fake gradient for large predictions:

```
r̂ = 0.05   ->  tanh(α·r̂) ≈ 0.05  ->  slope ≈ 1 - 0.0025 ≈ 1.00   (healthy)
r̂ = 1.0    ->  tanh(α·r̂) ≈ 0.76  ->  slope ≈ 1 - 0.58   ≈ 0.42   (weak)
r̂ = 3.0    ->  tanh(α·r̂) ≈ 1.00  ->  slope ≈ 1 - 1.00   ≈ 0.001  (dead)
r̂ = 5.5    ->  tanh(α·r̂) ≈ 1.00  ->  slope ≈ 0.00006            (dead)
```

The **more confident the model becomes, the smaller the gradient becomes**.
Once `|r̂| > ~3`, the gradient is effectively zero and the model freezes.

---

## 4. Why the model drives itself into that trap

The profit loss, by itself, has a dangerous incentive. Consider a stock with a
steady upward drift (like AAPL 2018–2026):

- The loss for **always being long** (+1 every day) is
  `-mean(r_t)` — a very good, low loss, because the stock usually rises.
- The loss for always being short is `+mean(r_t)` — terrible.
- Switching sides frequently adds transaction costs.

So the "easiest" way to make the loss look good is to pick the direction of the
drift, go `+1`, and **never trade**. The model has no intrinsic pressure to
*forecast* — harvesting the drift is free money in the loss.

Then the trap snaps shut:

1. The model pushes `r̂` larger and larger, because a bigger prediction is
   "more certain".
2. Once `|r̂|` is big, `tanh(α·r̂)` saturates → the STE gradient → 0.
3. The model can no longer adjust anything. It is permanently frozen.

We verified this on the saved models (read their weights, ran the real data
through them):

| Check | Profit-aware model | MSE model |
|---|---|---|
| Predicted return `r̂` | **constant 5.53 for every sample** | small, ±0.008 |
| STE gradient slope | **0.00006** (100% of samples dead) | 0.9999 (healthy) |
| Trading signal | always long (+1), **zero trades** | always long (+1), zero trades |

The profit-aware model collapsed into a single constant number for *all*
inputs. It predicts nothing — it just echoes "long" because AAPL went up. That
is why:

- the **loss curve** dives in epoch 1 (it found the drift floor) and then goes
  **flat** (gradient dead);
- the **profit curve** is flat — it is not a strategy, it is buy-and-hold
  drift, with no trades to show.

The MSE model did **not** saturate — MSE loss keeps predictions small — but its
evaluation signal also collapsed to always-long, because `sign(r̂)` on any
small positive average prediction is still `+1`.

---

## 5. How a hybrid loss fixes it

The fix adds a standard MSE term to the profit-aware loss:

```
L' = L_profit + λ * mean((r̂_t - r_t)²)
```

```
L' = -mean(signal_t * r_t - cost_t) + λ * MSE(r̂, r)
```

Why this works:

1. **MSE punishes large, wrong predictions.** If the model outputs 5.53 while
   real daily returns are ~±0.02, the MSE term is enormous `λ * 5.5²`. The
   optimization immediately pulls `r̂` back down to realistic magnitude.
2. **That keeps `tanh` unsaturated.** With `|r̂| ~ 0.02`, the STE slope is
   ≈ 1 — gradients flow fully again, so the model can keep learning and
   refining direction across all 50 epochs.
3. **It removes the free-lunch.** The model can no longer "cheat" by harvesting
   drift with a constant output; it has to actually predict returns, which is
   exactly what you want to claim in the thesis.
4. **It makes the comparison honest.** The MSE baseline and the profit-aware
   model now share a calibration objective; any profit improvement is
   attributable to the profit-aware part, not to the model hiding behind a
   constant.

`λ` (`--loss-lambda`) controls the balance:

- `λ` too small → saturation may creep back.
- `λ` too large → the profit-aware part is diluted into plain MSE.
- A reasonable starting point is `λ = 1.0`, then tune.

---

## 6. What healthy curves should look like after the fix

- Predictions vary per input (not a constant), with `mean|r̂|` well below 1.
- The STE slope stays > 0.5 through training (gradients alive).
- Signals actually change sign over time (real trades, transaction costs paid).
- The loss keeps descending (not a vertical drop + a flat line) and the
  reported directional accuracy / profit come from actual forecasting.

---

## 7. The one-line summary for your thesis

> The custom profit-aware loss rewarded constant "always-long" positioning
> because of market drift, and the straight-through tanh gradient vanished as
> the model grew over-confident—collapsing its output to a constant. Adding a
> weighted MSE term keeps predictions calibrated, keeps the gradient alive, and
> forces the model to actually forecast returns instead of harvesting drift.