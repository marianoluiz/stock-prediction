## 3.2.1 Proposed Algorithm

This revision retains Steps 0–11 and the GRU architecture. The learning
criterion has separate profit, downside-risk and asymmetric-cost components.
The implementation and fixed experimental controls are in
[downside-protocol.md](downside-protocol.md).

**Step 0: START**

**Step 1: Historical Stock Data Input and Preprocessing**

Load the fixed TradingView PSEi cache. Build six backward-looking channels:
simple return; return lags 1, 5 and 10; 20-day volume z-score; 10-day return
volatility. The saved index volume is zero. Remove unavailable warm-up rows.
Construct a 30 × 6 window ending before its target return day. Preserve the
existing four expanding-window training/validation/test partitions.

**Step 2: Market Return Computation**

$$r_t = P_t/P_{t-1}-1.$$

The target is the return realized on day t. Its input contains only days
through t−1. This indexing matches `utils.preprocessing.create_sequences`.

**Step 3: GRU Forecasting Layer**

Initialize with seed 42. Use the unchanged two-layer GRU with 64 hidden units,
dropout .2, final-state LayerNorm and a linear output head. The output is
bounded by five training-target standard deviations:

$$\hat r_t = 5\sigma_{train}\tanh(q_t).$$

**Step 4: Differentiable Continuous-Position Generation**

$$s_t=\tanh(\alpha\hat r_t),\qquad \alpha=1/\sigma_{train}.$$

Both GRUs use this mapping during financial evaluation. It is a synthetic
position in (−1,1), not a calibrated confidence probability. The same smooth
function supplies forward and backward computations; its gradient can become
small near saturation. No straight-through sign or direction-mismatch loss.

**Step 5: Asymmetric Position-Change Cost Computation — OBJECTIVE 3**

$$\Delta s_t=s_t-s_{t-1},$$
$$C_t=.00296\max(\Delta s_t,0)+.00396\max(-\Delta s_t,0).$$

Use the shared `utils.trading.position_change_cost`. Each evaluation window
starts flat; there is no forced terminal liquidation. Carry the last position
between chronological batches, detached from the previous computation graph.
Cost inclusion versus omission during profit-aware training is the Objective
Three ablation. Both conditions are scored with the same evaluation costs.

**Step 6: Simulated Net Return, Calibration and Downside Quantities**

$$n_t=s_tr_t-C_t,\qquad MSE_t=(\hat r_t-r_t)^2,$$
$$d_t=\max(0,-n_t)^2.$$

The zero-target downside quantity is zero on nonnegative days. All batch
observations enter its mean; it is not an average restricted to losing days.

**Step 7: Financial Loss Calculation — OBJECTIVES 1 AND 2**

$$L_{PA}=-\operatorname{mean}(n_t)+.7\operatorname{mean}(MSE_t),$$
$$L_{DR}=L_{PA}+\gamma\operatorname{mean}(d_t).$$

**Objective 1:** the negative mean net-return term gives simulated financial
outcomes a direct role in learning. Its base profit-aware model (gamma=0) is
compared with the MSE-only baseline.

**Objective 2:** the additional gamma-weighted squared-negative-net-return term
introduces downside-loss awareness. Compare gamma=0 with the selected positive
gamma while retaining lambda=.7, costs and the forecasting architecture.
Gamma=0 exactly reproduces the base loss; it does not modify MSE training.

For Objective 3, both cost-ablation models retain gamma=0. The fixed cost
coefficients are not the treatment; cost inclusion inside training is.

**Step 8: Backpropagation and Parameter Update**

Backpropagate the condition-specific loss through the GRU, output head,
position map and financial terms. Use the existing AdamW learning rate .001,
weight decay .001, batch size 64 and chronological batches. ReLU squared has
derivative zero at zero; the existing cost function is subdifferentiable.

**Step 9: Iteration and Prior-Validation Selection**

Train for 50 epochs, keeping final-epoch weights. For the new downside
experiment only, train the gamma=0 control and kappa in {.25,1,4}, with
gamma=kappa/sigma_train. Select a positive candidate by the common prior
validation utility `mean(net) − .7*MSE − alpha*mean(downside_squared)`.
The reference coefficient is alpha for every candidate; smaller kappa breaks
exact ties. Save the choice before test evaluation. Do not select weights,
epochs or checkpoints using held-out return. There is no refit on validation.

**Step 10: Output and Held-Out Evaluation**

Save forecasts, realized returns, positions, cost and net return. Report
forecast diagnostics, net additive/geometric return, Sharpe-like ratio,
zero-target downside loss/deviation, turnover and exposure by fold. Compare
cash and validation-fixed static positions in the new downside ablation.
Run separate paired uncertainty analyses for the principal comparison and
the downside comparison. Retain the historical lambda sweep as supplementary.

**Step 11: END**
