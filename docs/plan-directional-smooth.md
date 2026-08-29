# Plan: Directional-Accuracy Thesis + Smooth `tanh` Signal

## 1. Goal

Restructure the profit-aware training so that:

1. The **training signal** is the continuous, position-based rule already claimed in
   the thesis:

   ```
   signal_t = tanh(α · r̂_t)        # continuous between -1 and +1
   ```

   `signal_t` is a *position size*, not an action. "Buy then buy" = increasing the
   position size. No "already bought" state needs to be tracked.

2. The **evaluation** focuses on **direction** (which model predicts the sign of the
   return better) and then *simulates* who profits more from a discrete
   buy / sell / hold rule.

3. This removes the Straight-Through Estimator (STE) hack entirely, so gradients
   never saturate and we stop "changing/complicating things" with fake forward /
   backward passes.

The headline thesis statement becomes an **empirical, falsifiable claim**:

> A model with higher directional accuracy on the held-out test set also realizes
> higher simulated P&L (net of transaction costs).

We test that correlation directly, instead of claiming the training loss is the
exact profit we report.

---

## 2. Current design (what exists and why we're changing it)

Today `utils/trading.py` uses a **Straight-Through Estimator**:

- **Forward pass:** `sign(r̂)` -> a real `+1 / -1` position (identical to eval).
- **Backward pass:** pretends the derivative is the slope of `tanh(α·r̂)` so
  gradients keep flowing through the flat `sign` curve.

Why this causes problems (see `docs/profit-loss-saturation.md`):

- The `sign` function has zero gradient almost everywhere; only the STE hack keeps
  it trainable.
- The profit loss rewards "harvesting drift" (always-long on a rising stock), which
  pushes `|r̂|` larger and larger.
- Once `|r̂| > ~3`, `tanh(α·r̂)` saturates and the fake backward slope -> 0. The model
  freezes (observed: constant `r̂ = 5.53` for every sample).

The new design keeps the thesis's stated signal (`tanh`) and drops the STE hack,
removing the failure mode at the source.

---

## 3. Proposed design

### 3.1 Training loss — smooth `tanh` position (no STE)

Replace the STE-binary signal with the continuous position:

```
pos_t   = tanh(α · r̂_t)                      # position in (-1, 1)
cost_t  = c · |pos_t - pos_{t-1}|            # cost on any position change
profit  = pos_t · r_t - cost_t
L       = -mean(profit)
```

- **Forward and backward are the same function.** No `autograd.Function`, no fake
  slope. Gradients are alive for every `|r̂|` by construction — saturation is
  eliminated, not patched.
- **Transaction cost still applies** to position changes (now fractional changes).
- `α` (`--alpha`) still controls sharpness: higher `α` = more binary-like position
  sizing.

### 3.2 Evaluation — directional accuracy + discrete buy/sell/hold

The existing `arguments.md` already describes the eval rule. We keep and formalize it:

```
r̂ > +ε   ->  +1  (buy / long)
r̂ < -ε   ->  -1  (sell / short)
|r̂| <= ε ->   0  (hold / no position)
```

- `ε` is a small hold threshold (a knob, default to tune — see Q3 in the decision log).
- Metrics that already exist and need no changes (they operate on a `signal` array):
  - `directional_accuracy` (`utils/metrics.py:8`) — compares `sign(pred)` vs `sign(actual)`.
  - `cumulative_profit`, `cumulative_profit_geometric`, `sharpe_like`, `trade_log`.
- This keeps the comparison "purely about directional accuracy rather than
  prediction magnitude" — the framing already stated in `arguments.md`.

### 3.3 The thesis argument is a correlation, not an identity

Because training optimizes smooth partial positions but evaluation uses discrete
all-in actions, the training objective and the reported profit are **not literally
identical**. That is fine — and intentional:

- The **training loss** is the learning signal (smooth, never saturating).
- The **evaluation** is the measuring instrument: it shows whether better direction
  prediction translates into better simulated P&L.

This is a legitimate, testable thesis claim. It must be presented honestly as an
*empirical relationship*, not as "the loss maximizes the exact profit reported".

---

## 4. Implementation changes (file by file)

### 4.1 `utils/trading.py` — replace STE with smooth signal

- Remove the `StraightThroughSignal` autograd.Function (or leave it unused; deleting
  is cleaner).
- Add a plain function `smooth_signal(pred, alpha) -> tanh(alpha * pred)` (a normal
  torch op, no autograd.Function needed).
- Rewrite `profit_aware_loss` to accept a `position` (the `tanh` output) instead of
  the STE `trading_signal(...)`:
  ```
  profit = -mean( pos_t * r_t - c * |pos_t - pos_{t-1}| )
  ```
- `profit_per_step` / `transaction_cost` stay the same — they take any signal tensor
  (binary or continuous) and already generalize.
- `profit_aware_loss` remains *just* the profit term (`-mean(profit)`); the MSE
  calibration term is combined in `training/train.py` (section 4.2) so the loss
  module stays reusable.

### 4.2 `training/train.py` — wire the smooth loss + MSE calibration + eval hold

- In `run_epoch`, for `loss_type == "profit-aware"`, build the hybrid loss:
  ```
  profit_loss = profit_aware_loss(pred, y_batch, alpha, transaction_cost_rate, previous_signal)
  loss = profit_loss + loss_lambda * F.mse_loss(pred, y_batch)
  ```
  For `loss_type == "mse"`, keep `F.mse_loss(pred, y_batch)` unchanged (lambda not
  applied).
- The eval-signal construction (currently `signal = torch.sign(pred)` at
  `train.py:106`) stays **unchanged** — per Q3 `ε = 0`, raw `sign`, which feeds
  `cumulative_profit`, `trade_log`, etc. (all-in `+1 / -1`).
- Thread `loss_lambda` through `TrainingConfig` and `fit`.
- `directional_accuracy` keeps using raw `sign(pred)` (Q3, unchanged).

### 4.3 `train.py` / `main.py` — CLI additions

- Add `--alpha` (exists) and `--loss-lambda` (new, default `1.0`, applies only to
  the custom profit-aware loss).
- `main.py --compare` still trains MSE and profit-aware and prints the comparison
  table — the table already reports `Directional Accuracy` and `Cumulative Return`,
  which is exactly the accuracy-vs-profit evidence we want to surface.
- Consider adding an explicit per-epoch / final **accuracy-vs-profit** summary or a
  small scatter when `--compare` is used (the evidence plot for the thesis).

### 4.4 Docs — update to match

- `docs/arguments.md`: "Loss Function" section — `--loss profit-aware` now uses the
  smooth `tanh` position loss (not STE `sign`). "Signal Conversion (Evaluation)"
  section — add the hold threshold `ε` and the buy/sell/hold rule with `--loss-eps`.
- `docs/pipeline.md` (~line 60): the STE backpropagation description must be
  rewritten to describe the smooth `tanh` position loss instead of the
  straight-through estimator.

---

## 5. The `λ·MSE` calibration term is REQUIRED, not optional — NEW EVIDENCE

**`--loss-lambda` is now mandatory.** Empirically verified, not speculative:

A pure smooth-`tanh` loss gives the same **dead gradient** as the old STE at large
predictions:

```
# pure profit loss, alpha=1
r̂ = 5.5   ->  pos = tanh(5.5) ≈ 1.0  ->  dL/dr̂ = -1.3e-7   (DEAD)
```

The reason: the profit loss rewards increasing position magnitude toward ±1
unboundedly, so `r̂` still blows up, `tanh` saturates, and the gradient dies —
*exactly* the original bug. Bounding the output head with `tanh` does NOT help
either: the pre-activation can still grow to `∞`, saturating the bound at ±1 and
then saturating `pos = tanh(α·1)`.

Adding a small MSE term fixes it structurally — it pins `r̂` down to realistic
magnitude (~±0.02), far from saturation, so gradients stay alive *because the model
never gets to enter the dead region*:

```
# profit loss + λ·MSE(r̂, r), λ = 1, same inputs
r̂ = 5.5   ->  dL/dr̂ = +2.75   (HEALTHY — large, corrective)
```

This is the same mechanism as option A (the hybrid loss). Without it, reproducing
the original "everything freezes at one constant" bug is not just possible but
**expected**. Verdict: to ship a working build, the MSE term is non-negotiable.

**Wiring (small):**
```
L = profit_aware_loss(...) + loss_lambda * mean((r̂_t - r_t)²)
```
- Add `loss_lambda` to `TrainingConfig` and a `--loss-lambda` CLI flag (used only
  for `loss_type == "profit-aware"`; default `1.0`, tune).
- Add lambda support to the MSE baseline? NO — MSE baseline already *is* the full
  MSE loss; lambda applies only to the custom loss.
- **This overrides Q2** ("no MSE for now"): Q2 was decided before we proved
  saturation recurs without it. Update Q2 in section 7 accordingly.

---

## 6. Evaluation / verification plan

1. **Saturation check:** after training, confirm `mean|r̂|` is bounded and the model
   output is NOT a constant across inputs (the old failure signature was `r̂ = 5.53`
   for all samples).
2. **Gradient health:** with pure `tanh` the slope `α(1 - tanh²(α·r̂))` is always
   `> 0` for finite `|r̂|`; confirm no epoch-1 flatline.
3. **Direction story:** report `directional_accuracy` on the held-out test set for
   MSE vs profit-aware. Expect profit-aware >= MSE if the custom loss genuinely
   improves direction.
4. **Profit story:** for the same test window, simulate buy/sell/hold P&L (net of
   transaction costs) and show the correlation: does the higher-directional-accuracy
   model also show higher `cumulative_profit` / Sharpe? This is the thesis evidence.
5. **Robustness:** sweep `ε` (and optionally `α`) and confirm the accuracy-vs-profit
   relationship holds across settings, not just at one point.
6. **Controls:** report buy-and-hold as a baseline so the drift-harvesting concern
   is transparent (a model that merely rides drift should not be able to claim
   "real forecasting").

---

## 7. Decision log

- **Q1 — Is the thesis claim "directional accuracy correlates with profit"?** -> **YES.**
  That is the headline hypothesis; evaluation simulates P&L to demonstrate it.
- **Q2 — Add the `λ·MSE` calibration term now?** -> **YES (mandatory, was "no for
  now").** Empirical test showed pure smooth-`tanh` reproduces the dead-gradient
  bug (`dL/dr̂ = -1.3e-7` at `r̂ = 5.5`); adding `λ·MSE` restores healthy gradients
  (`+2.75`) because it keeps `r̂` out of the saturation region. Required to ship a
  working build. See section 5.
- **Q3 — Hold threshold `ε` policy?** -> **`ε = 0` (no hold / forced all-in).**
  Evaluation signal stays `sign(pred)` -> `+1 / -1` exactly as today; no `hold`
  state, no thresholding knob. `directional_accuracy` uses raw `sign(pred)`
  (standard). Because `ε = 0`, the "buy / sell / hold" rule becomes the existing
  `sign(pred)` rule already used by the code and described in `arguments.md`.
  **Decided.**
- **Q4 — Status of STE code.** -> **Remove / leave unused.** Cleaner to delete;
  confirm no other file imports `StraightThroughSignal` / `trading_signal` (current
  imports: only `profit_aware_loss` from `utils.trading` in `training/train.py`).

---

## 8. Analysis: is this the best approach?

**Strengths**
- Removes the STE hack and its saturation failure at the source — the most fragile
  part of the current code disappears.
- Aligns the *implementation* with the *thesis narrative* (position-based `tanh`,
  direction-focused evaluation) — no more disconnect between what the document
  claims and what the code does.
- Evaluation already exists and is reused unchanged; change is small and reversible
  (`utils/trading.py` + one threshold line in `run_epoch` + CLI flag).
- The thesis claim is now a clean, falsifiable empirical hypothesis.

**Weaknesses / risks**
- Train-vs-eval mismatch is real but *intentional*; must be presented as a
  correlation study, not as "the loss equals the reported profit".
- Does NOT by itself stop drift-harvesting; if the accuracy-vs-profit relationship
  turns out flat or both models merely ride drift, the claim is unsupported (that
  is a legitimate, reportable outcome).
- `ε` is a free parameter that must be justified and tuned.

**Verdict:** This is the cleanest path that both fixes the saturation bug and makes
the thesis argument honest and testable. It is strictly simpler than the hybrid
loss (A) while remaining open to adding the `λ·MSE` term later if experiments
require it.
