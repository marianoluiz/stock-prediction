# Detailed proposed algorithm with SOP 2

Detailed current-implementation companion to the historical docs/algorithm.md. Step numbers 0–11 are retained. This document describes the saved PSEi experiment and does not change model code.

This text and the editable HTML replacement are generated from `algorithm-with-sop-2.json`. See `figure-alignment-2026-10-28.md` for the live-document synchronization status.

## Step 0 START

## Step 1 Historical Stock Data Input and Preprocessing

**Given:** Pₜ closing PSEi level, Vₜ recorded volume, historical OHLCV observations, and window length L = 30.

Load the fixed TradingView PSEi daily OHLCV cache. Parse dates and numeric columns, remove invalid or incomplete observations, and construct backward-looking features. The implementation uses returns, lagged returns, rolling volume standardization, and rolling return volatility; it does not apply a separate MinMax scaler to every feature.

`xₜ = [rₜ, rₜ₋₁, rₜ₋₅, rₜ₋₁₀, zᵥ,ₜ, σ₁₀,ₜ]`

`zᵥ,ₜ = (Vₜ − mean₂₀(V)ₜ) / (SD₂₀(V)ₜ + 10⁻⁸)`

Here σ₁₀,ₜ is the trailing 10-observation sample standard deviation of returns. Rows without the required lag or rolling history are removed. Recorded volume in the saved PSEi cache is zero, so its volume z-score channel is constant.

`Xₜ = [xₜ₋L, xₜ₋L₊₁, …, xₜ₋₁] ∈ ℝ³⁰ˣ⁶`

The target is rₜ, the next available daily return after the input window. No feature from the target day is included in Xₜ. Four chronological expanding-window folds supply separate training, validation, and held-out test observations.

**SHARED FOUNDATION — OBJECTIVES 1, 2, AND 3**

## Step 2 Market Return Computation

**Given:** Pₜ closing PSEi level for the target day and Pₜ₋₁ preceding closing level.

Compute the simple close-to-close market return:

`rₜ = (Pₜ − Pₜ₋₁) / Pₜ₋₁`

This is the same one-period return as (Pₜ₊₁ − Pₜ) / Pₜ under a one-step shift of the time index. The notation here assigns the return to its realization day, so Xₜ contains observations through t − 1. The realized return is used both as the forecasting target and in the simulated profit calculation.

**SHARED FOUNDATION — OBJECTIVES 1, 2, AND 3**

## Step 3 GRU Forecasting Layer

**Given:** Xₜ input sequence, GRU and output-head parameters, and initial hidden state h₀ = 0 for each sequence.

Initialize the model using the experiment's fixed random seed. Process each sequence with a two-layer GRU containing 64 hidden units per layer and dropout of 0.2 between layers. For an internal sequence step j, the reset gate gᵣ,ⱼ, update gate gᵤ,ⱼ, candidate state h̃ⱼ, and hidden state hⱼ follow the PyTorch GRU formulation:

`gᵣ,ⱼ = sigmoid(Wᵣxⱼ + Uᵣhⱼ₋₁ + bᵣ)`

`gᵤ,ⱼ = sigmoid(Wᵤxⱼ + Uᵤhⱼ₋₁ + bᵤ)`

`h̃ⱼ = tanh(Wₕxⱼ + bₕₓ + gᵣ,ⱼ ⊙ (Uₕhⱼ₋₁ + bₕₕ))`

`hⱼ = (1 − gᵤ,ⱼ) ⊙ h̃ⱼ + gᵤ,ⱼ ⊙ hⱼ₋₁`

W and U are learned input and recurrent matrices, b denotes biases, and ⊙ is element-wise multiplication. Gate biases bᵣ and bᵤ combine the corresponding input and recurrent biases. The reset gate is applied after the recurrent candidate-state transformation, matching nn.GRU.

`qₜ = Wₒ LayerNorm(hfinal, layer 2) + bₒ`

`r̂ₜ = k tanh(qₜ),     k = 5 σtrain`

The last output from the second GRU layer passes through layer normalization and a linear head. The scaled tanh head bounds the return forecast using the training-target return standard deviation. This constrains magnitude but does not guarantee varied positions or prevent saturation.

**SHARED MODEL — OBJECTIVES 1, 2, AND 3**

## Step 4 Differentiable Continuous-Position Generation

**Given:** r̂ₜ predicted return and α scaling factor estimated from the training fold only.

`α = 1 / σtrain`

`sₜ = tanh(α r̂ₜ),     −1 < sₜ < 1`

Positive sₜ represents long-oriented exposure and negative sₜ short-oriented exposure; |sₜ| is the position magnitude. These are synthetic exposures, not directly executable PSEi orders. Both GRU conditions use this same mapping during financial evaluation.

`∂sₜ / ∂r̂ₜ = α (1 − tanh²(α r̂ₜ))`

The same smooth tanh function is used in forward and backward computations. Its derivative becomes small near saturation. There is no straight-through sign estimator or separate incorrect-direction penalty. Position magnitude and persistence are therefore evaluated explicitly.

**OBJECTIVES 1, 2, AND 3 — CONTINUOUS EXPOSURE AND POSITION BEHAVIOR**

## Step 5 Trading Friction Penalty Computation

**Given:** sₜ current position, sₜ₋₁ previous position, cᵦ = 0.00296 increase coefficient, and cₛ = 0.00396 decrease coefficient.

`Δsₜ = sₜ − sₜ₋₁`

`costₜ = cᵦ max(Δsₜ, 0) + cₛ max(−Δsₜ, 0)`

Increasing exposure incurs 0.296% per unit of position increase; decreasing exposure incurs 0.396% per unit of decrease. Changes in magnitude incur cost even when position direction remains unchanged. Each test window starts from s₀ = 0.

During sequential processing, the last position of one batch supplies the previous position for the next batch; the carried position is detached from the preceding batch's computation graph. The rule is a simplified asymmetric position-change cost and does not reconstruct all fees, taxes, spread, slippage, financing, liquidity, or short-selling restrictions.

Objective 3 compares including this cost during profit-aware training with omitting it during training. Saved positions are then evaluated under common stated costs as well as zero costs.

**OBJECTIVE 3 — ASYMMETRIC POSITION-CHANGE COST**

## Step 6 Cost-Adjusted Simulated Trading Profit Computation

**Given:** sₜ continuous position, rₜ realized return, costₜ position-change cost, and r̂ₜ predicted return.

Compute gross and cost-adjusted simulated per-step return:

`grossₜ = sₜ rₜ`

`profitₜ = nₜ = sₜ rₜ − costₜ`

The sign and magnitude of the position determine exposure to the realized market return. Subtracting the cost produces the net-return quantity aggregated by the profit-aware training objective.

As in the initial algorithm, define the per-observation MSE calibration term before combining the components:

`MSEₜ = (r̂ₜ − rₜ)²`

This term penalizes deviation of predicted returns from realized returns. Its weight is varied for Objective 2. Calibration can constrain forecast scale, but its effect on positions and simulated returns is assessed empirically rather than assumed beneficial.

**OBJECTIVES 1, 2, AND 3 — SIMULATED NET RETURN AND MSE CALIBRATION**

## Step 7 Profit-Aware Loss Calculation

**Given:** profitₜ simulated net return, MSEₜ calibration term, N observations in the training batch, and calibration weight λ.

`LossPA = −(1/N) Σₜ profitₜ + λ (1/N) Σₜ MSEₜ`

`LossPA = −mean(sₜrₜ − costₜ) + λ mean((r̂ₜ − rₜ)²)`

Minimizing the first component favors larger simulated net return under the specified position and cost rules. The second component assigns a numerical prediction-error penalty. The matched baseline minimizes MSE alone.

Objective 1 compares MSE-only and profit-aware training at λ = 0.7. Objective 2 compares λ = 0 and λ = 0.7, with 0.3, 1.0, and 2.0 as descriptive sensitivity checks. Objective 3 compares cost inclusion versus omission during training while using common evaluation costs.

The calibration experiment holds the data, folds, architecture, seed, training budget, and cost rule fixed. The held-out sweep does not select an optimal λ or establish that larger calibration weights improve returns.

**OBJECTIVES 1, 2, AND 3 — COMBINED TRAINING OBJECTIVE**

## Step 8 Backpropagation and Parameter Update

**Given:** The condition-specific loss, parameter vector θ, learning rate η, and AdamW decay coefficient d.

Clear accumulated gradients, backpropagate the batch loss through the output head, GRU, position mapping, and relevant loss components, and update model parameters using AdamW. The reported runs use η = 0.001 and d = 0.001.

`gᵤ = ∇θ LossB(θᵤ₋₁)`

`mᵤ = β₁mᵤ₋₁ + (1 − β₁)gᵤ`

`vᵤ = β₂vᵤ₋₁ + (1 − β₂)(gᵤ ⊙ gᵤ)`

`m̂ᵤ = mᵤ / (1 − β₁ᵘ),     v̂ᵤ = vᵤ / (1 − β₂ᵘ)`

`θᵤ = (1 − ηd)θᵤ₋₁ − η m̂ᵤ / (√v̂ᵤ + ε)`

Here u is the optimizer-update index, m and v are first- and second-moment estimates, and division is element-wise. The optimizer uses PyTorch's default β₁ = 0.9, β₂ = 0.999, and ε = 10⁻⁸. These equations retain the parameter-update explanation while reflecting AdamW rather than presenting a plain gradient-descent step as the implemented optimizer.

All matched conditions use the same optimizer settings. Their training losses determine which consequences influence the gradient.

**SHARED TRAINING STEP — OBJECTIVES 1, 2, AND 3**

## Step 9 Iteration

**Given:** Chronological training batches, validation observations, batch size 64, and 50 training epochs.

Repeat Steps 3 to 8 for the training batches in each epoch. Monitor validation loss using the same condition-specific objective. The reported experiments retain the final epoch without early stopping or best-checkpoint selection.

Repeat the matched training procedure separately for each chronological fold and experimental condition. Held-out observations do not update model parameters.

**SHARED TRAINING STEP — OBJECTIVES 1, 2, AND 3**

## Step 10 Output

**Given:** The trained model and the held-out input sequences for its PSEi fold.

Return the trained GRU model and generate next-day return forecasts. Apply the common tanh position mapping and evaluation cost rule. Save or derive aligned predictions, positions, realized returns, simulated costs, and net returns for the specified comparisons.

Forecast error, additive and geometric return, the Sharpe-like ratio, turnover, cost drag, and position persistence are evaluated using these outputs as specified in Section 3.5. These are simulated outcomes and do not establish executable PSEi trading.

**SHARED MODEL OUTPUT — OBJECTIVES 1, 2, AND 3**

## Step 11 END

## Implementation sources

- `utils/preprocessing.py`: cleaning, six features, sequences, chronological folds.
- `models/gru_model.py`: two-layer GRU, layer normalization, bounded output head.
- `utils/trading.py`: training-derived alpha, smooth position, asymmetric position-change cost.
- `training/train.py`: MSE versus combined loss, carried batch position, AdamW, final-epoch training.
- [PyTorch 2.12 GRU](https://docs.pytorch.org/docs/2.12/generated/torch.nn.GRU.html)
- [PyTorch 2.12 AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)
