# TODO

## High Priority

- [ ] Fix gradient saturation in loss function — `tanh(alpha=5.0)` saturates after epoch 1, model stops learning
  - Try lower alpha values or a different activation
  - Investigate if loss landscape has exploitable gradients after saturation

- [ ] Add `--loss` flag to switch between `mse` (baseline) and `profit-aware` (custom) loss functions
  - This is needed to compare the custom loss against a standard MSE baseline for the thesis

## Done

- [x] Fix cache key to include start/end dates (was reusing old cached data across date ranges)

## Nice to Have

- [ ] Separate train and test scripts (currently runs both in one go)
