# Plan: Separate Train / Evaluate / Predict Scripts

Splits the current all-in-one `main.py` into focused entry points so a trained
model can be saved, reloaded, and evaluated without retraining.

## Why

Currently `python main.py` does everything in one run: fetch -> split -> train
-> evaluate -> save. Problems for a thesis:

- A saved model cannot be re-evaluated on the test set without retraining.
- Multiple experiments on the same test set force redundant training.
- A live prediction demo at the defense requires waiting through training.
- `TODO.md` already lists this as "Nice to Have."

## Current Files (unchanged)

- `models/gru_model.py` — `GRUReturnPredictor`
- `training/train.py` — `fit()`, `run_epoch()`, `to_loader()`, `TrainingConfig`
- `utils/preprocessing.py` — data loading, returns, windows, split
- `utils/metrics.py`, `utils/trading.py`

## New Files

### `train.py`
Entry point: training only.

1. Parse args (subset of today's CLI: data, model arch, training, loss).
2. Load data, compute returns, create sequences, chronological 70/15/15 split.
3. Train and evaluate on the **validation** set (for progress curves).
4. Save the best/last model to `results/<loss>/gru_<loss>.pt` (checkpoint).
5. Save `loss_curve.png` / `profit_curve.png`.

Does **not** touch the test set — test data stays untouched for `evaluate.py`.

### `evaluate.py`
Entry point: evaluation of a saved model on the held-out test set.

1. Args: `--model <path.pt>`, `--loss`, plus the same data args as `train.py`.
2. Load data, compute returns, sequences, identical split (uses the same
   `create_sequences` / `train_val_test_split` logic so the test slice is the
   *exact same dates*).
3. Load model weights, run `run_epoch(model, test_loader, ...)`.
4. Print full test metrics table + optional `--trade-log`.

### `predict.py` (optional, for the defense demo)
Forecast the next N days after the latest data.

1. Args: `--model <path.pt>`, `--symbol`, `--days N` (default 5), `--start`.
2. Load model, seed with the last `sequence_length` returns, iterate N
   steps feeding predictions back as inputs.
3. Print the predicted daily returns / signals for each upcoming day.

## `main.py` After Refactor

Keeps working unchanged (fetch -> split -> train -> evaluate all-in-one) by
delegating to the shared pipeline, so old commands still run. Its test-set
metrics remain intact. Used for the `--compare` MSE vs profit-aware table.

## Shared Pipeline

To avoid duplicated logic between the three scripts, extract the data prep
steps into a small helper (e.g. `utils/pipeline.py`):

```python
def prepare_data(symbol, start, end, sequence_length):
    df = load_stock_data(...)
    returns = compute_returns(df)
    x, y, dates = create_sequences(...)
    split = train_val_test_split(...)
    return split, to_loader(split.x_train, ...), val_loader, test_loader
```

Both `train.py`, `evaluate.py`, and `main.py` call this — guaranteeing the
test set is always produced by the identical code path.

## CLI Sketch

```bash
# Train only (validation for monitoring, test untouched)
python train.py --symbol AAPL --loss profit-aware --epochs 50 \
    --save results/profit_aware/gru_profit_aware.pt

# Evaluate a saved model on the exact same test window
python evaluate.py --symbol AAPL --loss profit-aware \
    --model results/profit_aware/gru_profit_aware.pt

# Live forecast for a defense demo (no fixed train/test split)
python predict.py --symbol AAPL --model results/profit_aware/gru_profit_aware.pt --days 5
```

## Docs Updates

- Update `docs/arguments.md` with the new flags (`--save`, `--model`, `--days`).
- Update `docs/pipeline.md` to describe the split workflow.
- Update README usage examples.
- Update `TODO.md` (mark "separate train and test scripts" done).

## Open Questions

- Keep `run_single`'s "save last epoch" behavior, or add early stopping /
  best-checkpoint selection? (Recommended: save last epoch for simplicity,
  note it in the thesis.)
- Should `--compare` still exist? (Recommend: keep in `main.py` only.)