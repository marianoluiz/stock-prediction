"""Produce every thesis result in one go, with the default settings.

Runs, in order:
  1. benchmark.py --charts       MSE vs profit-aware on all 30 PSE symbols,
                                 plus the comparison figures
  2. scripts/significance.py     paired t-test on that benchmark (29 stocks)
  3. scripts/cost_comparison.py  with-cost vs without-cost comparison
  4. main.py --compare           one worked example (PSE:ICT): loss/profit
                                 curves and a full trade log

Outputs land in results/ (see README.md, "Thesis run"). Each step uses the
defaults: 50 epochs, loss lambda 0.7, buy cost 0.296%, sell cost 0.396%.

Usage:
    .venv/Scripts/python.exe run_thesis.py
    .venv/Scripts/python.exe run_thesis.py --start-step 2   # reuse the latest benchmark CSV
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Benchmark + comparison charts", ["benchmark.py", "--charts"]),
    ("Significance test", ["scripts/significance.py"]),
    ("Cost comparison", ["scripts/cost_comparison.py"]),
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
    args = parser.parse_args()

    t_all = time.time()
    for i, (name, cmd) in enumerate(STEPS, 1):
        if i < args.start_step:
            continue
        print(f"\n{'#' * 78}\n# [{i}/{len(STEPS)}] {name}: python {' '.join(cmd)}\n{'#' * 78}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode
        if code == CUDNN_EXIT_CRASH:
            print(f"# note: known cuDNN shutdown crash (0x{code:X}) after the step finished; its output is complete",
                  flush=True)
        elif code != 0:
            raise SystemExit(f"Step {i} ({name}) failed with exit code {code}")
        print(f"# [{i}/{len(STEPS)}] {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} min. Results are in {ROOT / 'results'}")


if __name__ == "__main__":
    main()
