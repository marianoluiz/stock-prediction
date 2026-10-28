"""SOP 2: retrain the profit-aware PSEi benchmark at several MSE calibration
weights (lambda) and compare them.

For every lambda this runs benchmark.py (4-fold walk-forward, PSEi only) and
then scripts/sop2_compare.py. Everything lands in one folder:

    results/sop2/
      manifest.json              settings, git revision, data hash, timings
      comparison.csv / .md       per-fold and mean metrics across lambdas
      sensitivity.png            RMSE / prediction / position / return vs lambda
      lambda_0p0/                one benchmark run per lambda:
        benchmark.csv              fold-level metrics (MSE and profit-aware)
        benchmark_daily.csv        every test day's prediction, position, net return
        benchmark_summary.txt      benchmark.py's text summary
        run.log                    full console output of that run
      lambda_0p3/ ...

The committed runs used --device cpu; the default keeps that so a rerun on the
same cached data reproduces them. Rerunning overwrites results/sop2/.

Usage:
    .venv/Scripts/python.exe scripts/sop2_sweep.py
    .venv/Scripts/python.exe scripts/sop2_sweep.py --lambdas 0 0.7 --epochs 2 --out results/sop2_smoke
    .venv/Scripts/python.exe scripts/sop2_sweep.py --compare-only               # no training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_thesis import CUDNN_EXIT_CRASH  # noqa: E402
from scripts.sop2_compare import DEFAULT_LAMBDAS, SWEEP_DIR, compare, run_dir  # noqa: E402
from utils.pipeline import cache_path_for  # noqa: E402

SYMBOL = "PSE:PSEI"
START = "2018-01-01"


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() or None


def run_benchmark(weight: float, args: argparse.Namespace, out: Path) -> float:
    """Run benchmark.py for one lambda, echoing its output into out/run.log."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "benchmark.py",
        "--symbols", SYMBOL, "--start", START,
        "--walk-forward", "--folds", "4", "--eval-fraction", "0.30",
        "--epochs", str(args.epochs), "--seed", str(args.seed), "--device", args.device,
        "--buy-cost", "0.00296", "--sell-cost", "0.00396",
        "--loss-lambda", f"{weight:g}",
        "--output", str(out / "benchmark.csv"),
    ]
    t0 = time.time()
    with (out / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            log.write(line)
            print(f"  [lambda {weight:g}] {line}", end="", flush=True)
        code = proc.wait()
    if code not in (0, CUDNN_EXIT_CRASH):
        raise SystemExit(f"benchmark.py failed for lambda={weight:g} (exit {code}); see {out / 'run.log'}")
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="SOP 2: profit-aware PSEi benchmark across MSE calibration weights")
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="Default cpu matches the committed SOP 2 runs")
    parser.add_argument("--out", type=Path, default=SWEEP_DIR,
                        help="Sweep folder, relative to the repo root (default: results/sop2)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training; rebuild comparison.* and sensitivity.png from the existing runs")
    args = parser.parse_args()
    sweep = ROOT / args.out

    timings: dict[str, float] = {}
    if not args.compare_only:
        for i, weight in enumerate(args.lambdas, 1):
            print(f"\n=== SOP 2 [{i}/{len(args.lambdas)}] lambda = {weight:g} ===", flush=True)
            timings[f"{weight:g}"] = round(run_benchmark(weight, args, run_dir(sweep, weight)), 1)

    print("\n=== SOP 2 comparison ===", flush=True)
    compare(sweep, args.lambdas)

    if args.compare_only:
        return
    data_file = ROOT / cache_path_for(SYMBOL, START, None)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "symbol": SYMBOL,
        "lambdas": args.lambdas,
        "planned_contrast": "lambda 0 vs 0.7; other weights are descriptive sensitivity checks",
        "settings": {"start": START, "folds": 4, "eval_fraction": 0.30, "epochs": args.epochs,
                     "seed": args.seed, "device": args.device, "buy_cost": 0.00296, "sell_cost": 0.00396},
        "data_file": data_file.relative_to(ROOT).as_posix(),
        "data_sha256": sha256(data_file),
        "versions": {"python": platform.python_version(),
                     **{pkg: version(pkg) for pkg in ("torch", "numpy", "pandas", "scikit-learn")}},
        "seconds_per_lambda": timings,
        "runs": {f"{w:g}": run_dir(args.out, w).as_posix() for w in args.lambdas},
    }
    (sweep / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {sweep / 'manifest.json'}")


if __name__ == "__main__":
    main()
