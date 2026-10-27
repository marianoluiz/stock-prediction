# With SOP 2 change record

Base: Mariano's `main` commit `645807a` (`feat: psei only fold`) and the latest
locally available full working-paper DOCX export dated 26 October 2026. The
repository's `main` branch did not contain that full thesis DOCX; this branch
adds it as `docs/with-sop-2.docx`. Reconcile any later live Google Doc edits
before replacing the shared paper.

## Paper

- **Chapter 1:** Replaced the former SOP 2 question about the tanh position
  mapping with a question about how the MSE calibration weight changes PSEi
  forecast error, position behavior, and after-cost simulated return. Updated
  Objective 2, its rationale, scope, and the profit-aware-loss definition.
- **Chapter 2:** Revised the literature link to the second problem so it
  describes the implemented balance between the profit term and MSE
  calibration, without implying a separate directional-mismatch penalty.
- **Chapter 3:** Kept the existing GRU, tanh position function, asymmetric
  costs, and main `lambda = 0.7` comparison. Added the matched SOP 2
  calibration experiment at `lambda = 0, 0.3, 0.7, 1.0, 2.0`, including its
  held-out diagnostics and the rule against selecting a weight from test
  performance. Restored the original inline equation drawings after editing
  their surrounding text.
- **Chapter 4:** Kept the existing primary comparison, reference-policy
  visuals, significance analysis, and cost ablation. Added Table 4.5 and the
  sensitivity chart as Figure 4.3 beside the other results. Renumbered the
  earlier significance and cost figures to Figures 4.4 and 4.5 and their
  tables to Tables 4.6 and 4.7. Updated the discussion to report that
  calibration reduced RMSE and exposure but did not improve mean geometric
  return in the planned `0` versus `0.7` contrast.
- **Chapter 5 and front matter:** Updated the summary, Objective 2 conclusion,
  recommendations, contents, list of figures, and list of tables. New or
  revised paper text is highlighted in yellow.

The revised DOCX retains all 42 original media assets and all 82 original
drawing instances from the local paper export, then adds one drawing for the
SOP 2 chart. These are preservation checks on the paper copy; they do not
assert that embedded images are byte-identical to standalone PNGs in `main`.

## Code and evidence added on this branch

- `benchmark.py` now saves predicted returns in its daily CSV and accepts
  `--device cpu` for a reproducible CPU sweep. Its default device selection
  and Mariano's primary experiment are unchanged.
- `sop2_compare.py` checks the matching PSEi fold dates and reconstructs
  reported RMSE and returns from daily rows before writing the comparison
  CSV, Markdown table, and sensitivity chart.
- `results/sop2_lambda_*` contains the five completed fold-level and daily
  result sets plus run logs. `SOP2_REPRODUCTION.md` gives the exact rerun
  command and cached-data hash.

The `lambda = 0.7` primary benchmark and the separate CPU sensitivity run
must be read as different runs. The five weights were examined on held-out
windows for description, not selected as a validated optimum.

## Figure package added after the paper revision

The `figures/paper/` gallery exports the eleven numbered images embedded in
this DOCX; `scripts/build_paper_figures.py` checks those images against the
DOCX and rebuilds a separate `figures/generated/` set from the archived
Chapter 1 illustration code and committed PSEi result CSVs. The Colab entry
point is `notebooks/PSEi_Thesis_Figures_Colab.ipynb`. This packaging does not
change model results. The group's original Chapter 1 notebook is archived
with outputs and account metadata removed.

## Chapter 1 source correction

Figure 1.2's caption and explanation now identify the image as synthetic
actual and noisy predicted prices. The original Colab constructs its
predicted prices from realized prices plus noise and reports positive
aggregate simulated profit; the paper no longer claims this visual proves
negative aggregate profit or out-of-sample forecasting. Figure 1.4's caption
and explanation now identify its fixed 0.5-unit-per-observation deduction as
an illustration, separate from the PSEi model's asymmetric position-change
cost. The List of Figures titles were updated. All figure images and model
results remain unchanged; the revised prose is highlighted yellow.
