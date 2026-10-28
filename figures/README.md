# Figures in the PSEi thesis

The [full paper](../docs/with-sop-2.docx) is the 28 October 2026 export of the Google Docs tab **OCTOBER 26 PSEi REVISION**. `paper/` contains all 11 numbered images embedded in that export. The gallery builder verifies each image's SHA-256 against the DOCX media, independently of Google's changing media names.

| Figure | Current content | Reproduction |
|---|---|---|
| 1.1 | MSE equation | Paper image retained; generated formula has a white background. |
| 1.2 | MSE versus profit-aware loss on one observed PSEi return | `scripts/build_chapter1_loss_figures.py`, fixed cached PSEi data and hypothetical forecasts. |
| 1.3 | Effect of lambda on the same illustrative loss | Same script, weights 0, 0.3, 0.7, 1, 2. |
| 1.4 | Fixed 0.5-unit cost in the original simulation | Original Colab arithmetic retained; not the implemented asymmetric cost. |
| 3.1 | Principal PSEi workflow | Exact saved diagram copied into the gallery. The calibration experiment is explained in the objective-mapping paragraph and Section 3.5.4. |
| 3.2 | Input-process-output architecture | Exact saved diagram copied into the gallery. |
| 4.1 | Per-fold profit-aware minus MSE geometric return | Committed benchmark CSV and `scripts/plot_loss_comparison.py`. |
| 4.2 | GRUs and fixed reference strategies | Same benchmark CSV and plotting module. |
| 4.3 | Calibration-weight sensitivity | Five committed lambda runs, reconstructed by `sop2_compare.py`. |
| 4.4 | Paired daily differences and intervals | Benchmark daily CSV and `scripts/significance.py`. |
| 4.5 | Cost-in-loss ablation | `results/cost_comparison.csv` and `scripts/cost_comparison.py`. |

Figures 1.2 and 1.3 illustrate a mathematical mechanism using an observed pre-test return. Their grid points are **hypothetical forecasts, not trained GRU predictions or test results**. The example selection, formulas, checksum, and plotting-point table are documented in [the calculation basis](CHAPTER_1_LOSS_BASIS.md).

Figures 4.1-4.5 show saved PSEi experiment outputs; replotting does not retrain the models. The Chapter 3 diagrams are preserved assets, not plots recreated from model results. Generated plots may differ in typography or compression from Google Docs' embedded PNGs while preserving the source values.

## Google Colab

Open [PSEi_Thesis_Figures_Colab.ipynb](../notebooks/PSEi_Thesis_Figures_Colab.ipynb) on the `codex/with-sop-2` branch and select **Runtime → Run all**. It verifies 13 result CSVs, checks the PSEi illustration CSV separately, verifies all 11 embedded images, rebuilds the generated plots, displays the current paper images, and offers a ZIP download.

The separate [SOP 1 and 2 notebook](../notebooks/SOP1_SOP2_PSEi_Loss_Illustrations_Colab.ipynb) recreates Figures 1.2 and 1.3 and every plotted point without training a model.

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/build_paper_figures.py
```

`generated/manifest.json` records image and input hashes. The original synthetic Figures 1.2 and 1.3 remain in `legacy-paper/`, with their original [Colab source](../notebooks/Chapter_1_original_Colab_source.ipynb). The prior paper is recoverable from Git revision `00996ca`; it is not the current copy.

See the [alignment audit](../docs/figure-alignment-2026-10-28.md) for the full paper, algorithm, and figure synchronization record.
