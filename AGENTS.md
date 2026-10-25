# Agent notes for this repo

Thesis project: GRU return predictor with a profit-aware trading loss vs. an MSE
baseline. See `README.md` for architecture/usage and `docs/algorithm.md` for
the original thesis algorithm (Steps 0-11) the code is meant to implement —
treat it as the spec to check code against, not a doc to edit when code
drifts. This file is about pitfalls that aren't obvious from the code alone.

## Environment

- Python deps live in `.venv/` (Windows). The bare `python` / `python3` on
  PATH is a *different* interpreter without `torch` etc. installed. Always
  invoke `.venv/Scripts/python.exe main.py ...` (or activate the venv first).
- On this machine (torch 2.12+cu126, RTX 3050) any process that runs a cuDNN
  GRU — even forward only — crashes at interpreter exit with 0xC0000409
  (bash shows exit 127). All output is already written; the run is valid.
  `torch.backends.cudnn.enabled = False` avoids it but is ~3x slower and
  changes results, so `run_thesis.py` tolerates that one exit code instead.
- Training runs on CUDA if available (`torch.cuda.is_available()`), else CPU.
  A `--compare` run trains two full 50-epoch models sequentially — budget a
  few minutes per symbol.

## Scope: PH market only, asymmetric costs

- The study universe is the PSEi constituents (minus MYNLD, too little
  history) plus `PSE:PSEI` as reference — `benchmark.py::PSEI_STOCKS` /
  `INDEX_SYMBOL`. Stocks are reported as one group (no tiers); the index is
  always reported separately and kept out of stock means and the t-test.
  **The study is now index-first**: every entry point defaults to `PSE:PSEI`,
  `run_thesis.py` evaluates it with 4-fold walk-forward, and `--stocks`
  (on `benchmark.py`, `cost_comparison.py`, `run_thesis.py`) brings back the
  29-stock universe. Index-only CSVs are `benchmark_summary_psei_*.csv`, full
  ones `_ph_*`; `latest_csv()` picks the newest of either by mtime.
  `significance.py`, the charts and the summary all branch on "are there
  stock rows?" -- keep that branch when changing any of them.
  `docs/system-and-results.md` is the current full write-up.
  US tickers were dropped; older `results/` and `docs/run-result*.md` may
  still describe US runs.
- Symbols are `PSE:TICKER` (routed to tvDatafeed). The `:` is illegal in
  Windows file names — anything that builds a path from a symbol must
  replace it (`PSE-ICT_...`), as `cache_path_for` and `main.py` do.
- Costs are asymmetric: 0.296% per unit of position increase, 0.396% per unit
  of decrease (`utils.trading.position_change_cost`). Never reintroduce a
  local `rate * |delta|` — every simulator must call that one function.

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
