# Experiment summary — profit-aware vs MSE on PSE (2026-10-25)

All runs: GRU (2 layers, 64 hidden), seq 30, 50 epochs, data from 2018-01-01,
`benchmark.py` defaults unless stated. "Geo" = compounded return over the test
window. Real PSE costs = 0.296% buy / 0.396% sell.

## TL;DR

Across every setup tested, the profit-aware loss changes **position sizing and
trend exposure**, not **directional skill**. Directional accuracy is at chance
for both models. Under real PSE costs, profit-aware does worse than MSE.
Narrowing to the PSEi index, changing λ or changing seeds doesn't fix this. On
the index, both models collapse to a constant one-sided bet.

---

## 1. Main run — 30 symbols (29 PSEi stocks + PSEi), real costs, λ=0.7

Source: `results/benchmark_summary_ph_20261025_174115*`

| Model | DirAcc | Geo return | Sharpe |
|---|---|---|---|
| MSE | 47.3% | −3.9% | −0.71 |
| Profit-aware | 46.6% | −7.2% | −0.22 |
| Buy & hold | | −16.5% | |

- Profit-aware beats MSE on geo return in **11/30** symbols.
- 23/30 test windows were falling. In those, profit-aware lost to MSE by
  6.4 pp on average. In the 7 rising windows it won by 6.7 pp.
- Profit-aware's return correlates **0.68** with buy & hold (MSE: 0.33). It
  mostly tracks the market.
- Degenerate (never changes side): MSE 3/30, profit-aware 5/30.

## 2. Cost check — same 30 symbols, flat 0.1% cost

Source: `results/cost_check_flat01/`

| | 0.1% cost | Real cost |
|---|---|---|
| Profit-aware mean geo | +0.8% | −7.2% |
| MSE mean geo | +0.2% | −3.9% |
| Profit-aware beats MSE | 15/30 | 11/30 |
| Profit-aware degenerate | 1/30 | 5/30 |

- Cheaper costs help profit-aware about twice as much as MSE (+8.0 pp vs
  +4.1 pp), so it **trades more**. A separate 6-symbol run also measured about
  2x the turnover.
- Even at 0.1%, profit-aware only **ties** MSE: median −1.2%, and −1.8% mean
  without ICT.
- Higher costs push the model toward degenerate "pick a side and hold" policies.

## 3. Comparison with the old September run (18 symbols, 0.1% flat cost)

Source: git history, `benchmark_summary_ph_20260917_183903` (removed in `f3e281e`)

Mean profit-aware geo return on the 16 symbols in both runs:

| Run | Profit-aware | MSE |
|---|---|---|
| Sept run, 0.1% cost | +16.0% | +2.1% |
| Now, 0.1% cost | +7.8% | +1.4% |
| Now, real cost | +0.3% | −2.6% |

- Costs account for about 7.5 pp of the 15.7 pp drop. The other ~8 pp comes
  from the later data end date (test window moved) and code changes since
  September. The model is very sensitive to the setup.
- The old +15% mean was mostly ICT (+106%). The old median was +1.9%.
- In the old run, MSE also "beat" buy & hold (−4%) just by staying near flat.

## 4. PSEi index only — λ × seed grid, real costs, 5 walk-forward folds

Source: this folder (`psei_l<λ>_s<seed>.*`), λ ∈ {0, 0.3, 0.7, 1.0, 2.0},
seeds {42, 43, 44}. That's 15 test windows per λ.

| λ | Profit-aware geo (mean ± std) | DirAcc | Degenerate | PA beats MSE |
|---|---|---|---|---|
| 0.0 | −1.7% ± 8.3% | 49.8% | 15/15 | 4/15 |
| 0.3 | +1.4% ± 5.1% | 52.3% | 15/15 | 7/15 |
| 0.7 | +0.8% ± 2.5% | 52.3% | 15/15 | 12/15 |
| 1.0 | +0.5% ± 1.5% | 52.3% | 15/15 | 12/15 |
| 2.0 | +0.05% ± 0.3% | 51.0% | 14/15 | 13/15 |
| MSE (λ unused) | −0.06% ± 0.8% | 51.5% | 15/15 | — |

Buy & hold per fold: +5.7%, −16.3%, −0.8%, −0.8%, −4.4%.

- Profit-aware holds a **constant short** in every fold and seed. It loses
  when the index rises (fold 0) and gains when it falls (fold 1).
- The directional accuracy values repeat exactly across seeds because they are
  each window's down-day share, not skill.
- Raising λ only shrinks the size of the bet toward zero.
- Seeds barely change anything. The collapse is systematic, not bad luck: the
  model learns the training period's downward drift.
- "Beats MSE 12/15" just means a small constant short beat a near-zero
  position in a mostly falling market.

## 5. Metric note — directional accuracy on PSE data

`utils/metrics.py::directional_accuracy` counts flat days (`actual == 0`) as
wrong for any non-zero prediction. About 6.6% of recent PSE stock days are flat
(up to 12.5%, SMC), so random guessing scores about **46.7%**, not 50%. The
stock-level 46–47% accuracy is therefore **chance**, not worse than chance.
PSEi has no flat days, so this doesn't affect section 4.

---

## Conclusions

1. The profit-aware loss produces **larger, more frequent, trend-following
   positions**. It doesn't improve directional prediction.
2. **Transaction costs decide the outcome.** At 0.1% it ties MSE. At real PSE
   costs it loses.
3. **The results are fragile.** Small changes in costs, test window or code
   move outcomes by several percentage points. Earlier positive results came
   from cheap costs plus one stock (ICT).
4. **The index is no easier.** On the PSEi, both models reduce to a constant
   drift bet at every λ and seed.
5. Possible thesis framing: an evaluation showing that, under realistic PSE
   costs, a profit-aware loss changes behaviour rather than skill, with the
   failure modes diagnosed (tanh saturation / degenerate collapse, cost-driven
   one-sided policies, flat-day accuracy artifact).

## Suggested next steps

- Fix directional accuracy to exclude flat days, and add mean-position,
  |position| and turnover columns to `benchmark.py`.
- Run the 30-symbol benchmark with 3–5 seeds to put error bars on sections 1–2.
- Optional "attempted fix" experiment: trade only when the predicted move is
  larger than the round-trip cost, or use a longer horizon (5–20 days).
