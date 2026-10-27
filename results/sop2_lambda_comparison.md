# SOP 2: PSEi lambda comparison

Same cached PSEi data, four test windows, seed 42, 50 epochs, and asymmetric buy/sell costs (0.296%/0.396%) for every run.
Lambda 0 versus 0.7 is the planned comparison; other weights are exploratory sensitivity checks. Values below are arithmetic means of four fold-level results, not a pooled investment return.

| λ | RMSE | Mean absolute predicted return | Mean absolute position | Turnover/day | Mean additive return | Mean geometric return | Directional accuracy | One-side folds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.06639 | 0.06573 | 1.000 | 0.00634 | +3.2585% | +2.4349% | 50.70% | 4/4 |
| 0.3 | 0.01498 | 0.01053 | 0.659 | 0.00457 | +2.0533% | +1.6940% | 50.70% | 4/4 |
| 0.7 | 0.01203 | 0.00539 | 0.383 | 0.00302 | +1.0945% | +0.9688% | 50.70% | 4/4 |
| 1 | 0.01129 | 0.00333 | 0.244 | 0.00228 | +0.6361% | +0.5825% | 50.70% | 4/4 |
| 2 | 0.01083 | 0.00055 | 0.041 | 0.00086 | -0.0605% | -0.0625% | 49.45% | 4/4 |

## Planned contrast: λ = 0.7 minus λ = 0

- Mean fold RMSE difference: -0.05436; lower RMSE in 4/4 folds.
- Mean absolute predicted-return difference: -0.06034.
- Mean absolute position difference: -0.617.
- Mean fold geometric-return difference: -1.47 percentage points; higher return in 1/4 folds.

## Per-fold results

| λ | Fold | Dates | RMSE | Geometric return | Mean predicted return | Mean position | Side switches |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | 2024-03-25 to 2024-11-13 | 0.06899 | +0.5040% | -0.06837 | -1.000 | 0 |
| 0 | 1 | 2024-11-14 to 2025-07-10 | 0.06717 | +1.2861% | -0.06632 | -1.000 | 0 |
| 0 | 2 | 2025-07-11 to 2026-03-02 | 0.06530 | -1.3190% | -0.06456 | -1.000 | 0 |
| 0 | 3 | 2026-03-03 to 2026-10-23 | 0.06410 | +9.2685% | -0.06365 | -1.000 | 0 |
| 0.3 | 0 | 2024-03-25 to 2024-11-13 | 0.01496 | +0.4825% | -0.01125 | -0.672 | 0 |
| 0.3 | 1 | 2024-11-14 to 2025-07-10 | 0.01627 | +1.1100% | -0.01149 | -0.695 | 0 |
| 0.3 | 2 | 2025-07-11 to 2026-03-02 | 0.01411 | -0.6897% | -0.01017 | -0.653 | 0 |
| 0.3 | 3 | 2026-03-03 to 2026-10-23 | 0.01459 | +5.8733% | -0.00922 | -0.615 | 0 |
| 0.7 | 0 | 2024-03-25 to 2024-11-13 | 0.01147 | +0.3449% | -0.00574 | -0.394 | 0 |
| 0.7 | 1 | 2024-11-14 to 2025-07-10 | 0.01327 | +0.8211% | -0.00647 | -0.449 | 0 |
| 0.7 | 2 | 2025-07-11 to 2026-03-02 | 0.01106 | -0.3150% | -0.00520 | -0.379 | 0 |
| 0.7 | 3 | 2026-03-03 to 2026-10-23 | 0.01231 | +3.0243% | -0.00414 | -0.312 | 0 |
| 1 | 0 | 2024-03-25 to 2024-11-13 | 0.01057 | +0.2278% | -0.00355 | -0.252 | 0 |
| 1 | 1 | 2024-11-14 to 2025-07-10 | 0.01240 | +0.5978% | -0.00433 | -0.312 | 0 |
| 1 | 2 | 2025-07-11 to 2026-03-02 | 0.01028 | -0.1670% | -0.00322 | -0.242 | 0 |
| 1 | 3 | 2026-03-03 to 2026-10-23 | 0.01191 | +1.6713% | -0.00223 | -0.172 | 0 |
| 2 | 0 | 2024-03-25 to 2024-11-13 | 0.01000 | -0.0007% | -0.00037 | -0.027 | 0 |
| 2 | 1 | 2024-11-14 to 2025-07-10 | 0.01172 | +0.1470% | -0.00105 | -0.078 | 0 |
| 2 | 2 | 2025-07-11 to 2026-03-02 | 0.00976 | -0.0072% | -0.00033 | -0.026 | 0 |
| 2 | 3 | 2026-03-03 to 2026-10-23 | 0.01186 | -0.3893% | +0.00044 | +0.034 | 0 |

The one-side flag means the executed position kept the same sign throughout a fold. It does not mean the position's magnitude was constant. This single-seed test sweep is descriptive and should not be used to select a lambda on the held-out test windows.
