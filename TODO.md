# TODO

## High Priority

- [ ] Directional accuracy is stuck at chance (0.5000) once saturation is fixed, and the
      model loses money net of transaction costs (SPY test: -6.5% additive, -7.2% geometric,
      Sharpe -0.41). The earlier saturated run's "+28,971 PHP" was not skill — it was an
      always-long position riding SPY's uptrend over that window. Single-feature (past
      returns only) input has no signal above noise. Try adding features beyond raw returns
      (rolling volatility, volume z-score, extra lagged-return channels) and see if
      directional accuracy moves off 50%.

## Done

- [x] Fix gradient saturation in the profit-aware loss — the model was collapsing to a
      constant output (verified: `pred` constant ~5.34-5.53, always-long, zero trades) because
      `--loss-lambda` defaulted to `0.0` in `main.py`/`evaluate.py`/`train.py`, so the MSE
      calibration term in `profit_aware_loss` never actually ran (the `TrainingConfig`
      dataclass default in `training/train.py` was overridden by the CLI arg every time).
      Fix: default `--loss-lambda` changed to `0.1` in `main.py`, `evaluate.py`, `train.py`,
      and `training/train.py`'s `TrainingConfig`. Verified on SPY: `pred` now moves across a
      realistic range (-0.008 to +0.004) instead of being pinned at 5.34, signal flips
      throughout the test set instead of always +1, and test MSE dropped from 28.5 to 0.00007.

- [x] Fix cache key to include start/end dates (was reusing old cached data across date ranges)
- [x] Add `--loss` flag to switch between `mse` (baseline) and `profit-aware` (custom) loss functions
      for the MSE-vs-PA thesis comparison
- [x] Name saved models/curves with symbol + date range (e.g.
      `results/profit_aware/AAPL_profit_aware_2018-01-01_2026-08-17.pt`),
      graph titles show symbol, range, and train/val/test split
- [x] Add optional `--metrics-out` to `evaluate.py` to save a metrics summary file
- [x] Early stopping / best-checkpoint selection in `fit()` — opt-in via `--early-stop-patience`
      (0 = disabled, trains the full `--epochs` as before). Monitors `--early-stop-metric`
      (default `val_loss`) on the validation set and restores the best-epoch weights into
      `model` before returning, instead of leaving the last epoch's.
