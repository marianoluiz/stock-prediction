"""Produce every thesis result in one go, with the default settings.

The study's primary series is the PSEi index (PSE:PSEI). Runs, in order:
  1. benchmark.py --walk-forward --folds 4 --charts
                                 MSE vs profit-aware on the PSEi, on 4
                                 expanding-window out-of-time test folds,
                                 plus the comparison figures and equity curve
  2. scripts/significance.py     daily paired test (Newey-West) and fold-level
                                 sign test on that benchmark
  3. scripts/cost_comparison.py --walk-forward --folds 4
                                 with-cost vs without-cost comparison on the
                                 same folds
  4. scripts/sop2_sweep.py       SOP 2: profit-aware benchmark retrained at
                                 lambda 0, 0.3, 0.7, 1.0, 2.0 (CPU), compared
                                 in results/sop2/
  5. main.py --compare           one worked example (PSE:PSEI, single 70/15/15
                                 split): loss/profit curves and a full trade log

With --stocks, steps 1-3 instead run the PSEi constituents (29 stocks plus
the index, single split; the cost comparison uses six of the stocks), as the
study did before it was narrowed to the index; the SOP 2 sweep is index-only
and is skipped.

Outputs land in results/ (see README.md, "Thesis run"). Each step uses the
defaults: 50 epochs, loss lambda 0.7 (the SOP 2 sweep varies it), buy cost
0.296%, sell cost 0.396%.

Usage:
    .venv/Scripts/python.exe run_thesis.py
    .venv/Scripts/python.exe run_thesis.py --start-step 2   # reuse the latest benchmark CSV
    .venv/Scripts/python.exe run_thesis.py --stocks         # the 29-stock universe instead
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WALK_FORWARD = ["--walk-forward", "--folds", "4"]
STEPS = [
    ("Benchmark + comparison charts", ["benchmark.py", *WALK_FORWARD, "--charts"]),
    ("Significance test", ["scripts/significance.py"]),
    ("Cost comparison", ["scripts/cost_comparison.py", *WALK_FORWARD]),
    ("SOP 2 lambda sweep", ["scripts/sop2_sweep.py"]),
    ("Worked example (PSE:PSEI)", ["main.py", "--compare", "--trade-log", "--symbol", "PSE:PSEI"]),
]
STOCK_STEPS = [
    ("Benchmark + comparison charts (stocks)", ["benchmark.py", "--stocks", "--charts"]),
    ("Significance test (stocks)", ["scripts/significance.py"]),
    ("Cost comparison (six stocks)", ["scripts/cost_comparison.py", "--stocks"]),
    ("Worked example (PSE:ICT)", ["main.py", "--compare", "--trade-log", "--symbol", "PSE:ICT"]),
]

# 0xC0000409: on this machine (torch 2.12 + CUDA 12.6, RTX 3050) any process
# that ran a cuDNN GRU crashes during interpreter shutdown, *after* all output
# is written. Disabling cuDNN avoids it but trains ~3x slower and changes the
# numbers, so this one exit code is tolerated; any other failure stops the run.
CUDNN_EXIT_CRASH = 0xC0000409


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every thesis step with the default settings")
    parser.add_argument("--start-step", type=int, default=1, choices=range(1, len(STEPS) + 1),
                        help="Skip the steps before this one (e.g. 2 to reuse an existing benchmark CSV)")
    parser.add_argument("--stocks", action="store_true", help="Run the 29-stock universe instead of the PSEi index")
    args = parser.parse_args()
    steps = STOCK_STEPS if args.stocks else STEPS

    t_all = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        if i < args.start_step:
            continue
        print(f"\n{'#' * 78}\n# [{i}/{len(steps)}] {name}: python {' '.join(cmd)}\n{'#' * 78}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        if code == CUDNN_EXIT_CRASH:
            print(f"# note: known cuDNN shutdown crash (0x{code:X}) after the step finished; its output is complete",
                  flush=True)
        elif code != 0:
            raise SystemExit(f"Step {i} ({name}) failed with exit code {code}")
        print(f"# [{i}/{len(steps)}] {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} min. Results are in {ROOT / 'results'}")


if __name__ == "__main__":
    main()
