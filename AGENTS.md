# Agent notes for this repo

Thesis project: GRU return predictor with a profit-aware trading loss vs. an MSE
baseline. See `README.md` for architecture/usage. This file is about pitfalls
that aren't obvious from the code alone.

## Environment

- Python deps live in `.venv/` (Windows). The bare `python` / `python3` on
  PATH is a *different* interpreter without `torch` etc. installed. Always
  invoke `.venv/Scripts/python.exe main.py ...` (or activate the venv first).
- Training runs on CUDA if available (`torch.cuda.is_available()`), else CPU.
  A `--compare` run trains two full 50-epoch models sequentially — budget a
  few minutes per symbol.

## `docs/` freshness — read the code before trusting a doc

The loss implementation was rewritten mid-project (commit `0da444f`, "loss =
netprofit tanh smooth return"), and some docs describe the pre-rewrite
mechanism. Check `git log -1 -- docs/<file>` if unsure; as of this writing:

- **Stale mechanism description**: `docs/profit-loss-saturation.md`,
  `docs/plan-directional-smooth.md` (committed at `8ef95a2`, before the
  rewrite). They describe a Straight-Through Estimator (`sign()` forward +
  fake `tanh` gradient backward) that **no longer exists** —
  `utils/trading.py` now uses real `tanh` for both forward and backward, no
  STE, no `torch.sign` in the loss. Their *diagnosis of the symptom*
  (prediction collapses to a saturating constant) and their *proposed fix*
  (`--loss-lambda` MSE calibration term) are still correct and verified
  working — only the "why" (STE-specific framing) is outdated. The real why:
  `tanh` itself saturates for large `|alpha * pred|` regardless of STE, so
  the gradient vanishes once the model's raw prediction runs away.
- **Current**: `docs/sesh.md`, `docs/run-result-no-plusmse.md` (untracked —
  newest analysis), `docs/arguments.md`, `docs/pipeline.md` (uncommitted
  edits), `docs/run-result.md`, `docs/run-result-w-llambda.md` (same commit
  as `sesh.md`).
- **Historical/completed plan**: `docs/sep-scripts.md` proposed splitting
  `main.py` into `train.py` / `evaluate.py` / `predict.py`. That split is
  already done — treat it as a record, not a TODO.

When in doubt, verify a doc's claim against `training/train.py::run_epoch`
and `utils/trading.py` directly rather than trusting the doc's prose.

## Known open issue (as of `6f8d85e`, "save return back but not good bi accuracy")

`--loss-lambda` defaults to `0.0` in both `main.py` and `evaluate.py`. At that
default, the profit-aware loss reliably collapses within ~2 epochs: the raw
prediction runs away to a large constant (e.g. `+5.7`), `tanh(alpha * pred)`
saturates to `±1`, gradient vanishes, and the model freezes there for the
rest of training. Symptoms: `signal = sign(pred)` is the same value every
single test day, MSE metric explodes (~20-32 instead of ~0.0004), and any
"profit-aware beats MSE" result is really just both models betting on the
test window's buy-and-hold drift direction — not learned forecasting.
Verified live across MSFT/GOOG/NVDA on `feat/tanh-signal`.

Setting `--loss-lambda 1.0` does fix the saturation (predictions stay
realistic, MSE stays healthy, signal actually flips day-to-day) — but the
resulting directional accuracy has tested *below* coin-flip (47.2% on MSFT).
So: saturation-fix and forecasting-skill are currently two separate unsolved
problems, not one. Don't assume enabling `--loss-lambda` alone makes the
profit-aware model good — verify with `--trade-log` before citing a result.
