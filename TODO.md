# TODO

## High Priority

- [ ] Fix gradient saturation in the profit-aware loss — the model collapses to a constant
      output (verified: `pred` constant 5.53, STE gradient slope 0.00006, always-long, zero trades)
      and the loss/profit curves go flat after epoch 1. See
      `docs/profit-loss-saturation.md` for the full explanation.
  - Plan: hybrid loss `L = -mean(signal*r - cost) + lambda * MSE(pred, r)`
    with a `--loss-lambda` flag. Add to `utils/trading.py` `profit_aware_loss`,
    `TrainingConfig`, and `training/train.py`.

## Done

- [x] Fix cache key to include start/end dates (was reusing old cached data across date ranges)
- [x] Add `--loss` flag to switch between `mse` (baseline) and `profit-aware` (custom) loss functions
      for the MSE-vs-PA thesis comparison
- [x] Name saved models/curves with symbol + date range (e.g.
      `results/profit_aware/AAPL_profit_aware_2018-01-01_2026-08-17.pt`),
      graph titles show symbol, range, and train/val/test split
- [x] Add optional `--metrics-out` to `evaluate.py` to save a metrics summary file

## Nice to Have

- [ ] Early stopping / best-checkpoint selection in `fit()` (currently saves the last epoch only)
