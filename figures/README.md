# Figures for the PSEi thesis copy

This folder corresponds to [`docs/with-sop-2.docx`](../docs/with-sop-2.docx).
`paper/` contains all **11 images actually embedded** as Figures 1.1–1.4,
3.1–3.2, and 4.1–4.5. The exporter checked each PNG byte-for-byte against
the DOCX's media part. `generated/` contains the output of
[`scripts/build_paper_figures.py`](../scripts/build_paper_figures.py), plus a
manifest with source and SHA-256 information.

| Paper figure | Meaning | Source and rebuild method |
|---|---|---|
| 1.1 | MSE equation | Exact paper equation image retained. |
| 1.2 | Low error, poor simulated profit | Group's original [Chapter 1 Colab](../notebooks/Chapter_1_original_Colab_source.ipynb): 200-day synthetic price series, seed 42, predicted prices constructed from actual prices plus noise. |
| 1.3 | Symmetric MSE | Same original Colab: actual 100, predictions 105 and 95, squared error 25 in both cases. |
| 1.4 | Trading friction illustration | Same original Colab: a fixed 0.5-unit deduction per simulated observation. |
| 3.1–3.2 | PSEi workflow and architecture | Exact diagrams from the current paper, carried through unchanged. The gallery copies these images; it does not redraw their shapes. |
| 4.1–4.2 | Per-fold geometric difference and reference strategies | Replotted from the committed PSEi benchmark CSV using `scripts/plot_loss_comparison.py`. |
| 4.3 | MSE calibration-weight sensitivity | Recalculated from the five committed lambda runs in `results/sop2/` using `scripts/sop2_compare.py`. |
| 4.4 | Paired differences and intervals | Recomputed from the benchmark daily CSV using `scripts/significance.py`. |
| 4.5 | Cost-in-loss ablation | Replotted from `results/cost_comparison.csv` using `scripts/cost_comparison.py`. |

**Interpretation boundary:** Figures 1.2–1.4 are illustrations, not evidence
from the held-out PSEi test folds. The original Figure 1.2 simulation uses
actual prices to construct noisy predicted prices, so it must not be called an
out-of-sample forecast. The corrected paper caption identifies Figure 1.2 as
synthetic actual and noisy predicted prices. The Colab reports **positive
aggregate gross profit of about 115 units** on its synthetic path; the plot
itself displays prices, not a profit series. Its inherited in-image title
mentions "poor trading profit," but the chart alone cannot establish that
claim. Figure 1.4's cost is a fixed example, not the
0.296%/0.396% asymmetric position-change cost used by the current model.
Figures 4.1–4.5 show the saved PSEi experiment results. Replotting those CSVs
does not retrain the models. Differences in font, resolution, or cropping
between `paper/` and `generated/` do not change the plotted source values.

## Run all in Google Colab

Open [`notebooks/PSEi_Thesis_Figures_Colab.ipynb`](../notebooks/PSEi_Thesis_Figures_Colab.ipynb)
from the `codex/with-sop-2` branch in Google Colab, then choose **Runtime →
Run all**. It clones this branch when needed, verifies 13 saved CSV input
hashes and all 11 paper images, rebuilds the generated set, displays the
exact paper figures in order, and offers a ZIP download.

To run locally from the repository root with the project's installed Python:

```powershell
.venv/Scripts/python.exe scripts/build_paper_figures.py
```

The existing unnumbered benchmark and training charts remain in `results/`.
They are supporting outputs; the numbered gallery maps specifically to the
figures used in the thesis copy.
