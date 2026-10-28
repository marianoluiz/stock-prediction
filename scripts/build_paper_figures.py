"""Rebuild the figure gallery for docs/with-sop-2.docx without training models.

Figures 1.2 and 1.3 use observed PSEi data and hypothetical forecasts.
Figures 1.1 and 1.4 retain the original illustrative calculations. Chapter 3 diagrams are preserved paper assets. Chapter 4
charts are recalculated from the committed PSEi result CSVs with the same
plotting functions that produced the repository's original charts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cost_comparison import plot as plot_cost_comparison  # noqa: E402
from scripts.plot_loss_comparison import plot_comparisons  # noqa: E402
from scripts.significance import index_tests, plot as plot_significance  # noqa: E402
from scripts.build_chapter1_loss_figures import build as build_loss_figures
from sop2_compare import main as compare_lambdas  # noqa: E402

BENCHMARK = Path("results/benchmark_summary_psei_20261025_203842.csv")
INPUT_HASHES = {
    BENCHMARK: "1f5904d9febf36dd0c1aff1dc9d40e0d5eb6cd2e1b5101de5d51db30f9affc31",
    Path("results/benchmark_summary_psei_20261025_203842_daily.csv"):
        "54eafd5c59ba59d50dd9e7bbec0a94b7c885312e60e49de344251d8d85febe09",
    Path("results/cost_comparison.csv"):
        "c44a12f56137980eea16c2bf422bbe085b4545fb04aa04dad3024f7522a26a1e",
    Path("results/sop2_lambda_00.csv"):
        "638a79d8fc1dc72879e7c528dfdeeb698c6203aea53625428429edaf6c8f1950",
    Path("results/sop2_lambda_00_daily.csv"):
        "5a5e2e0237acaa8e6d3e7c2ae41193280fbffa90da773893d8af69ad7af8d6b5",
    Path("results/sop2_lambda_03.csv"):
        "558e1fe9d3e9486c5cb65429df43b0fe3088c71fbd85d11a8ae8f371797d4407",
    Path("results/sop2_lambda_03_daily.csv"):
        "00b35aa0fef5e844829c526b71e8cb5028fd2a5e5580967bb97dc63b6c24dac7",
    Path("results/sop2_lambda_07.csv"):
        "f0cc1dc657b7fa7180b025314896f40ca845ff13d815c1c85431177c6fbcf9eb",
    Path("results/sop2_lambda_07_daily.csv"):
        "69b6d8e9e432a0e43764cb24b02687f941977b984f326e64242bab21bc79fe7b",
    Path("results/sop2_lambda_01.csv"):
        "88d600eeb225b9e9ed69aa96f9aeda0b100afedec4165756ce220c3e3e4cae8c",
    Path("results/sop2_lambda_01_daily.csv"):
        "72ed087bc599f68fd81944ec68fdf309fe6aa450269d6b9016613cefaa0aa3c4",
    Path("results/sop2_lambda_02.csv"):
        "3e9010e335a60657bc2369fe75e9f27763632980675be358e0a01bf8554fc286",
    Path("results/sop2_lambda_02_daily.csv"):
        "8b3defbb2027c0d8e14adf2f0b482a5660c377394d901f28934578d5d9528b32",
}
CHAPTER_1 = {
    "figure_1_1.png": "MSE equation; generated version is typeset on white for visibility",
    "figure_1_2.png": "One observed pre-test PSEi return: MSE versus profit-aware loss over hypothetical forecasts",
    "figure_1_3.png": "Same observed return: illustrative loss sensitivity at lambda 0, 0.3, 0.7, 1, 2",
    "figure_1_4.png": "Original Colab's fixed 0.5-unit-per-row cost illustration",
}
CHAPTER_3 = ("figure_3_1.png", "figure_3_2.png")
CHAPTER_4 = {
    "figure_4_1.png": "per_fold_geo_diff.png",
    "figure_4_2.png": "strategy_comparison.png",
    "figure_4_3.png": "sop2_lambda_sensitivity.png",
    "figure_4_4.png": "significance.png",
    "figure_4_5.png": "cost_comparison.png",
}



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    for relative, expected in INPUT_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing committed result: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Result changed: {relative}; review before replotting")
    print(f"Verified {len(INPUT_HASHES)} committed result CSVs")


def verify_paper_images(paper: Path) -> None:
    with ZipFile(ROOT / "docs/with-sop-2.docx") as document:
        hashes = {hashlib.sha256(document.read(name)).hexdigest()
                  for name in document.namelist() if name.startswith("word/media/")}
        expected = [*CHAPTER_1, *CHAPTER_3, *CHAPTER_4]
        for name in expected:
            if sha256(paper / name) not in hashes:
                raise ValueError(f"Current paper image missing from the DOCX: {name}")
    print(f"Verified {len(expected)} exact images against the current paper DOCX")


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_chapter_1(output: Path) -> dict[str, float]:
    """Retain the legacy simulations for audit; replace 1.2/1.3 below."""
    # The original DOCX image is transparent black text. Typeset a white
    # rendition here so the formula stays visible in dark GitHub themes.
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.5, r"$\mathrm{MSE}=\frac{1}{N}\sum_{t=1}^{N}(r_t-\hat r_t)^2$",
            ha="center", va="center", fontsize=24, transform=ax.transAxes)
    save(fig, output / "figure_1_1.png")

    rng = np.random.RandomState(42)
    days = 200
    price = 100 + np.cumsum(rng.normal(0, 1, days))
    predicted = price + rng.normal(0, 0.5, days)
    frame = pd.DataFrame({"actual": price, "predicted": predicted})
    mse = float(np.mean((predicted - price) ** 2))
    frame["actual_change"] = frame.actual.diff()
    frame["pred_change"] = frame.predicted.diff()
    frame["position"] = np.where(frame.pred_change > 0, 1, -1)
    frame["profit"] = frame.position * frame.actual_change
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.actual, label="Actual Price")
    ax.plot(frame.predicted, label="Predicted Price")
    ax.set_title("Synthetic actual and noisy predicted prices (legacy)")
    ax.legend()
    save(fig, output / "figure_1_2.png")

    squared_errors = [(prediction - 100) ** 2 for prediction in (105, 95)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Overprediction (105)", "Underprediction (95)"], squared_errors)
    ax.set_title("Symmetric MSE Penalty")
    ax.set_ylabel("MSE Loss")
    save(fig, output / "figure_1_3.png")

    # This is the group's original illustration, NOT the thesis's asymmetric
    # position-change cost or a held-out strategy return.
    gross = float(frame.profit.sum())
    net = float((frame.profit - 0.5).sum(skipna=True))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No Transaction Cost", "With Transaction Cost"], [gross, net])
    ax.set_title("Trading Friction Impact")
    ax.set_ylabel("Total Profit")
    save(fig, output / "figure_1_4.png")
    return {"illustrative_mse": mse, "illustrative_gross_profit": gross,
            "illustrative_net_profit": net}


def build_chapter_4(output: Path) -> None:
    summary_path = ROOT / BENCHMARK
    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(summary_path.with_name(summary_path.stem + "_daily.csv"))
    if set(summary.symbol) != {"PSE:PSEI"} or set(summary.fold) != {0, 1, 2, 3}:
        raise ValueError("Expected four PSEi-only benchmark folds")

    with tempfile.TemporaryDirectory(prefix="paper-figures-") as temp_name:
        temp = Path(temp_name)
        charts = temp / "benchmark_charts"
        plot_comparisons(summary, charts, daily=daily)
        shutil.copyfile(charts / "per_fold_geo_diff.png", output / "figure_4_1.png")
        shutil.copyfile(charts / "strategy_comparison.png", output / "figure_4_2.png")

        # sop2_compare.py verifies matching test dates and reconstructs its
        # metrics from daily rows. Run it on a temporary copy of the inputs.
        sweep = temp / "sweep"
        sweep.mkdir()
        for source in sorted((ROOT / "results").glob("sop2_lambda_*.csv")):
            if source.name.startswith("sop2_lambda_comparison"):
                continue
            shutil.copyfile(source, sweep / source.name)
        argv = sys.argv
        try:
            sys.argv = ["sop2_compare.py", "--results", str(sweep)]
            compare_lambdas()
        finally:
            sys.argv = argv
        shutil.copyfile(sweep / "sop2_lambda_sensitivity.png", output / "figure_4_3.png")

        tests, unit = index_tests(summary_path)
        plot_significance(tests, unit, summary_path.name, output / "figure_4_4.png")
        cost = pd.read_csv(ROOT / "results/cost_comparison.csv")
        if set(cost.symbol) != {"PSE:PSEI"} or set(cost.fold) != {0, 1, 2, 3}:
            raise ValueError("Expected four PSEi-only cost-ablation folds")
        plot_cost_comparison(cost, 0.00296, 0.00396, output / "figure_4_5.png")


def main() -> None:
    verify_inputs()
    paper = ROOT / "figures/paper"
    output = ROOT / "figures/generated"
    expected = [*CHAPTER_1, *CHAPTER_3, *CHAPTER_4]
    missing = [name for name in expected if not (paper / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images extracted from the paper: {missing}")
    verify_paper_images(paper)
    output.mkdir(parents=True, exist_ok=True)
    illustrative = build_chapter_1(output)
    illustrative["psei_loss_illustration"] = build_loss_figures(
        output, ROOT / "data/PSE-PSEI_2018-01-01_latest.csv")
    for name, source in {
        "figure_1_2.png": "Figure_1_2_SOP1_data_based_loss_comparison.png",
        "figure_1_3.png": "Figure_1_3_SOP2_data_based_lambda_curves.png",
    }.items():
        shutil.copyfile(output / source, output / name)
    for name in CHAPTER_3:
        shutil.copyfile(paper / name, output / name)
    build_chapter_4(output)
    manifest = {
        "paper": "docs/with-sop-2.docx",
        "paper_sha256": sha256(ROOT / "docs/with-sop-2.docx"),
        "note": "Gallery matches the 2026-10-28 Google Docs export. Historical synthetic figures 1.2 and 1.3 are retained separately in figures/legacy-paper. See docs/figure-alignment-2026-10-28.md. Chapter 1 is illustrative; Chapter 3 diagrams are saved assets; Chapter 4 replots committed CSVs without retraining.",
        "illustration_input_sha256": "35bb13112fc537d62697042bb7dd27803816f51254cadad2822ed0e5913d4972",
        "inputs": {str(k).replace("\\", "/"): v for k, v in INPUT_HASHES.items()},
        "illustrative_values": illustrative,
        "figures": [
            {"number": name[7:10].replace("_", "."), "paper": f"figures/paper/{name}",
             "paper_sha256": sha256(paper / name),
             "current": f"figures/paper/{name}",
             "current_sha256": sha256(paper / name),
             "generated": f"figures/generated/{name}",
             "generated_sha256": sha256(output / name),
             "source_detail": CHAPTER_1.get(name, CHAPTER_4.get(name, "Exact paper diagram")),
             "status": ("illustrative" if name in CHAPTER_1 else
                        "diagram copy" if name in CHAPTER_3 else "CSV replot")}
            for name in expected
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(expected)} figures in {output}")


if __name__ == "__main__":
    main()
