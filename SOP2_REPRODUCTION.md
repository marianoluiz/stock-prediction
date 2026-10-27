# PSEi SOP 2 lambda experiment

The SOP 2 comparison extends the PSEi-only repository state at
`645807a4acb1f8c35df9ac164ee7d4771d93f319`. The benchmark records
`predicted_return` in its daily CSV, and `sop2_compare.py` calculates forecast
magnitude and position behavior. The separate thesis copy is
`docs/with-sop-2.docx`; its newly revised wording is highlighted in yellow.
The sensitivity chart is embedded as Figure 4.3 alongside the paper's
original Chapter Four visuals; the standalone PNG remains in `results/`.
The thesis copy was built from the latest locally available full DOCX export
of the working paper (26 October 2026), preserving its figures and main
tables. Compare any edits made later in the live Google Doc before replacing
that document with this copy.
For a section-by-section record, see `docs/with-sop-2-changes.md`.
For the exact paper images, current PSEi chart sources, and a Colab run-all
gallery, see `figures/README.md` and
`notebooks/PSEi_Thesis_Figures_Colab.ipynb`.

## Data and environment used here

- PSEi cached file: `data/PSE-PSEI_2018-01-01_latest.csv`
- SHA-256: `35BB13112FC537D62697042BB7DD27803816F51254CADAD2822ED0E5913D4972`
- Last cached observation: 2026-10-23
- Python 3.12.14, PyTorch 2.12.0+cpu, NumPy 2.4.6, pandas 3.0.3,
  scikit-learn 1.8.0
- Four expanding-window test folds, seed 42, 50 epochs; PSEi index only
- Buy cost 0.00296 and sell cost 0.00396 per unit of position change
- Main comparison: lambda 0 versus 0.7. Lambda 0.3, 1.0, and 2.0 are
  exploratory sensitivity checks.

## Rerun from the repository root in PowerShell

The commands below overwrite files with the same names in `results/`.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$runs = @(
    @{ value = '0'; tag = '00' },
    @{ value = '0.3'; tag = '03' },
    @{ value = '0.7'; tag = '07' },
    @{ value = '1.0'; tag = '01' },
    @{ value = '2.0'; tag = '02' }
)
foreach ($run in $runs) {
    & '.\.venv\Scripts\python.exe' benchmark.py `
      --symbols PSE:PSEI --start 2018-01-01 `
      --walk-forward --folds 4 --eval-fraction 0.30 `
      --epochs 50 --seed 42 --device cpu `
      --buy-cost 0.00296 --sell-cost 0.00396 `
      --loss-lambda $run.value `
      --output "results/sop2_lambda_$($run.tag).csv" `
      *> "results/sop2_lambda_$($run.tag)_run.log"
    if ($LASTEXITCODE -ne 0) { throw "Lambda $($run.value) failed" }
}
& '.\.venv\Scripts\python.exe' sop2_compare.py
```

Each benchmark produces a fold-level CSV, a daily CSV, a text summary, and
a run log. The saved reference runs were CPU runs; `--device cpu` avoids
hardware-dependent differences from the primary GPU benchmark. The same
cached data file is needed for numerical reproduction. `sop2_compare.py`
verifies that the four test date sequences match
across weights and recomputes RMSE and returns from the daily rows. It writes
`results/sop2_lambda_comparison.csv`, `.md`, and a sensitivity `.png`.
The reported returns are means of fold-level results, not pooled or
annualized portfolio returns.

Use the training/validation data to choose lambda in a later experiment.
Reporting the full test-window sweep here does not establish that its
best-returning value will generalize. A single seed also cannot establish
stability across initialization.
