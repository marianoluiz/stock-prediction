# Paper figure and algorithm alignment

## Source and synchronization

The current `with-sop-2.docx` is a native export of the existing Google Docs
tab **OCTOBER 26 PSEi REVISION**, downloaded on 28 October 2026 after the edits
below. It is not a newly reconstructed manuscript.

[Live Google document](https://docs.google.com/document/d/18dI9A6nictxKtwUsZi-vYEzwHdwTcIToBo5tAluDUa0/edit?tab=t.v4y691jc21z1)

The pre-edit repository version was `00996ca`. That version did not contain
the latest live Chapter 1–3 revisions. The source code, model weights,
experimental CSVs, and Chapter 4–5 findings were not changed in this sync.
The captured text of Chapters 4 and 5 was compared before and after the edits
and was identical.

## Figures

All 11 numbered paper images are present under `figures/paper/`, and the
gallery checks their bytes against the current DOCX. Figure 4.3 is the λ
sensitivity plot; Figure 4.4 is the significance plot; Figure 4.5 is the cost
ablation. The front-matter titles were corrected and the missing Figure 4.5
entry was added. Front-matter page numbers still require a final pagination
refresh after the manuscript's formatting is settled.

The live document already contained the new PSEi loss images for Figures 1.2
and 1.3, but their captions and surrounding explanations still described the
old synthetic examples. This sync corrected those descriptions and sources,
updated the List of Figures titles, added the matching Colab notebook and
generator, and preserved the old images separately.

The new examples use the observed PSEi return on 31 January 2019 and a grid of
hypothetical forecasts. They illustrate the loss mechanism; they do not show
trained-model performance. The example was chosen from pre-test history to
make the trade-off visible. Full calculation details are in
`figures/CHAPTER_1_LOSS_BASIS.md`.

Figures 3.1 and 3.2 are preserved summary diagrams. Figure 3.1 does not draw
the λ sweep as a separate branch. The objective-mapping paragraph and
Section 3.5.4 describe that comparison using the same pipeline with λ varied.
The gallery displays and copies these static diagram assets; it does not
redraw them from code. Figures 4.1–4.5 are regenerated from committed CSVs.

## Restored proposed algorithm

The previous Chapter 3 rewrite condensed the algorithm too far. It removed
equations and explanatory material even though the corresponding operations
remained in the implementation. The current paper restores Steps 0–11,
the Given lines, mathematical operations, and objective labels.

The detailed companion is `algorithm-with-sop-2.md`, generated with the
editable HTML from `algorithm-with-sop-2.json`. It covers:

- Cleaning, six feature channels, 30-observation windows, and chronological folds.
- The daily-return formula and its time-index convention.
- GRU reset/update/candidate/hidden-state operations and the bounded output head.
- The smooth tanh position and its derivative.
- Asymmetric position-change costs and carried batch position.
- Gross return, net return, and the per-observation MSE calibration term.
- The combined loss and the three controlled comparisons.
- Backpropagation, AdamW moment estimates and parameter update.
- Training iteration, validation monitoring, final-epoch evaluation, and outputs.

Descriptions were aligned with the current code rather than copying stale
claims from the initial specification: there is no straight-through estimator,
separate direction penalty, all-feature MinMax scaler, symmetric single cost
rate, or guarantee that MSE calibration prevents constant-side positions.
The historical `docs/algorithm.md` remains unchanged as required by `AGENTS.md`.

## Verification

`scripts/build_paper_figures.py` checks the saved result hashes, verifies all
11 embedded images, recreates the two PSEi loss illustrations, reconstructs
the λ comparison, and redraws the Chapter 4 plots without retraining. The
manifest records the current DOCX, figure, and source-data hashes. Training
code and numerical result inputs remain unchanged.

The Word export was rendered for inspection. Google Docs and LibreOffice
paginate the thesis differently; final page references should follow the
submission version. This synchronization is a content and source alignment,
not a claim that the thesis has completed final typesetting review.
