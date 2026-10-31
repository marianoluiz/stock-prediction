# Downside-risk ablation protocol (version 1)

SOP 2 introduces an additional **loss mechanism**, not another lambda sweep.
The existing cost-aware profit loss is the matched control. SOP 1 retains
MSE versus profit-aware learning; SOP 3 retains cost-in-loss versus no-cost-in-loss.
These component experiments are separate; the historical SOP 1 and 3 results
must not be relabelled as outcomes of the new downside model.

## Objective and scale

```
R_net[t] = s[t]*r[t] - position_change_cost(s[t], s[t-1])
L_downside = mean(relu(-R_net)**2)
L = -mean(R_net) + 0.7*MSE(predicted_return, actual_return) + gamma*L_downside
gamma = kappa / std(training target returns), ddof=0
```

The downside term is a zero-target lower partial second moment over **all**
observations, including zeros on nonnegative days. Its square root is daily
downside deviation. The gamma scale converts squared-return magnitude into
return magnitude relative to training volatility; it is a design convention,
not a universal optimum. The unchanged MSE term uses the original calibration
convention. Gamma=0 exactly recovers the existing profit-aware implementation.
ReLU squared has derivative zero at zero and is differentiable. The shared
piecewise-linear cost remains subdifferentiable, as in the existing algorithm.

## Fixed experimental controls

PSEi cache SHA256: `35bb13112fc537d62697042bb7dd27803816f51254cadad2822ed0e5913d4972`.
Use the original four expanding windows, sequence length 30, six inputs,
training/validation chronology, 64-unit two-layer GRU, dropout 0.2,
LayerNorm/linear head, output bound five training standard deviations,
continuous tanh exposure, batch 64, AdamW lr .001 and weight decay .001,
50 epochs, final-epoch weights, seed 42, no shuffled batches. Run both new
conditions on CPU, four threads, deterministic algorithms. Costs remain
buy .00296 and sell .00396; each evaluation window begins flat with no forced
terminal liquidation. No architecture, features, extra thresholds or
trading restrictions are added.

## Validation-only selection, fixed before testing

For each fold, train a gamma=0 control and candidates kappa in {.25, 1, 4}.
Each candidate starts with the same initialization and random seed. Evaluate
final-epoch candidates using the **same** validation score:

```
U_validation = mean(R_net) - .7*MSE - alpha*L_downside
alpha = 1 / std(training targets)
```

Select the highest score, with the smaller kappa breaking exact ties. The
reference kappa in this scoring rule is fixed at 1, regardless of candidate
training weight. Costs are charged from a flat start in validation. Save the
candidate scores and selection before evaluating the selected model on its
held-out window. Do not inspect test returns to change candidates, scoring,
epoch count or selection. No refit on validation data is performed. Later
folds may train on earlier test periods, as in the original walk-forward design.

## Reporting and checks

Primary downside endpoint: mean squared negative **net** return (LPM2).
Report daily downside deviation, loss-day share, sum of negative daily net
returns, net additive/geometric return, Sharpe-like ratio, RMSE, turnover,
mean/absolute exposure, near-flat share (|position|<.01, diagnostic only),
sign switches and worst daily loss. Report all four folds and arithmetic
fold means. No claim that downside awareness guarantees lower held-out risk.

Report a zero-return cash reference and constant exposures fixed from each
model's **validation** mean position. These diagnose reduced exposure and
static-side behavior; they do not establish a deployable trading policy.

Two-sided paired daily Newey-West HAC tests assess selected-minus-control
net return and LPM2 (631 days, Bartlett rule as in the existing statistics
script). Report effect sizes, 95% intervals and exact p-values as exploratory,
without selecting one-sided tests or declaring a family-wide discovery.
Four folds and one seed limit inference; paired daily data do not measure
training-seed uncertainty. Existing principal-comparison p-values are never
reused for this new ablation. Lower downside with smaller exposure is an
exposure tradeoff, not automatically improved forecasting.

## Sources

Fishburn's below-target mean-risk framework motivates the measure; the
coupling to this GRU loss and the validation utility are experimental choices.

- Fishburn, P. C. (1977). Mean-risk analysis with risk associated with
  below-target returns. *American Economic Review, 67*(2), 116–126.
  https://www.jstor.org/stable/1807225
- Harlow, W. V. (1991). Asset allocation in a downside-risk framework.
  *Financial Analysts Journal, 47*(5), 28–40. https://doi.org/10.2469/faj.v47.n5.28
- Kidd, D. (2012). *The Sortino ratio: Is downside risk the only risk that matters?*
  CFA Institute. https://rpc.cfainstitute.org/sites/default/files/-/media/documents/code/gips/the-sortino-ratio.pdf

The old lambda experiment remains supplementary and can still
be rebuilt with `scripts/sop2_compare.py`; its filename is retained to preserve
reproducibility of historical outputs.
