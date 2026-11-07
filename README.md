# NeuroToolbox

A modular, **dataset-agnostic** toolkit for brain–behavior analyses (EEG/fMRI ROI features + behavioral outcomes).  
It refactors a long, lab-specific script into a reusable package with clear modules, CLI entry points, logging, and reproducible outputs.
CSV spreadsheet input for easy integration across labs and datasets.

## Features

- **Asymmetry indices** from ROI left/right pairs
- **Normative deviations** via linear residuals or **Gaussian Process Regression**
- **Regression families** with standardization and multiple-comparisons correction (FDR, Bonferroni)
- **Multivariate**: PLS / optional CCA (with permutation p for LV/canonical correlations)
- **Structural covariance networks** with FDR-pruned edges and graph metrics
- **Robustness**: permutation tests, bootstrap CI for β, Huber/Theil-Sen/RANSAC fits
- **Publication-ready plots**: heatmaps, forest plots, score scatters, network heatmaps/graphs
- **Logging** of every step for reproducibility

## Install

```bash
# recommended: in a fresh virtual environment
python -m pip install .
# For development and the regression/CLI checks:
python -m pip install pytest
python -m pytest -q
```

## Data expectations

Your CSV can use **any column names**. You specify:
- Behavioral outcomes: e.g., `--outcomes tepanticip tepsconsum`
- ROI left/right pairs: e.g., `--roi-pairs lNACC:rNACC lcaud:rcaud`
- Covariates: e.g., `--covars age gender estICV`

Inputs must be numeric (encode categorical covariates before analysis). The
GPR option currently requires complete data in its selected ROI/covariate
columns. The other examples drop incomplete cases in the selected analysis
columns; plan missing-data handling before comparing models or cohorts.

## Quick starts

### 1) Asymmetry + Normative + Regressions

```bash
python scripts/run_asymmetry.py   --data TEPSData.csv   --outdir outputs_asym   --roi-pairs lcaud:rcaud lput:rput lNACC:rNACC lamyg:ramyg   --outcomes tepanticip tepsconsum   --covars age gender estICV   --normative linear   --composite
```

Outputs (CSV + figures) land in `outputs_asym/` and an `analysis.log` captures all steps.

### 2) Multivariate (PLS / CCA) + Network

```bash
python scripts/run_multivariate.py   --data TEPSData.csv   --outdir outputs_multivariate   --rois lcaud rcaud lput rput lNACC rNACC lamyg ramyg   --items teps1 teps2 teps3 teps4 teps5 teps6   --covars age gender estICV   --residualize   --do-cca
```

Generates weights, score plots, correlation heatmap, FDR-pruned network, node/global metrics.

### 3) Robustness follow-ups

```bash
python scripts/run_followups.py   --data TEPSData.csv   --outdir outputs_followups   --x AI_nacc   --y tepsconsum   --group gender
```

Produces permutation p-values, bootstrap CI for β, robust model slopes, and a stratified scatter.

## Config via YAML (optional)

See `examples/config_example.yaml` for how you might store your analysis spec and invoke scripts accordingly.  
(You can also add your own wrapper that parses YAML and calls the scripts).

## Repository layout

```
neurotoolbox/
├─ neurotoolbox/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ asymmetry.py
│  ├─ normative.py
│  ├─ regression.py
│  ├─ visuals.py
│  ├─ multivariate.py
│  ├─ networks.py
│  └─ robustness.py
├─ scripts/
│  ├─ run_asymmetry.py
│  ├─ run_multivariate.py
│  └─ run_followups.py
├─ examples/
│  └─ config_example.yaml
├─ README.md
├─ requirements.txt
└─ pyproject.toml
```

## Notes

This is a standalone Python package for tabular data, not an EEGLAB plugin.
Export subject-level features to CSV before using it. Python 3.12 is the
tested environment. Normative deviations currently fit and evaluate on the
same input cohort; they are descriptive residuals, not out-of-sample clinical
reference scores. Validate the design and covariates for your own study.

Network significance requires participant counts. Matrices returned by
`structural_covariance(data)` retain pairwise counts; for a complete-case
matrix loaded from CSV use `fdr_edges(R, n_samples=number_of_participants)`.

PLS/CCA permutation tests refit the latent model for each permutation of the
behavioral rows. They assume exchangeable, independent participants and are
not a substitute for held-out prediction or a study-specific permutation
scheme. Signed graph summaries are exploratory; modularity can be undefined.

- All regressions are **standardized** by default (X and y), and we report standardized β.
- FDR/BH and Bonferroni are applied **within each (Outcome × Model)** family.
- Network modularity uses greedy modularity; if it fails (small graphs), the code falls back gracefully.
- Figures are generated with matplotlib (no style dependencies).

## License

MIT

## Downloadable synthetic example

[Run the bundled synthetic example](examples/synthetic/README.md) to check your setup against
included reference outputs. It contains actual input files, a one-command runner,
expected tables/plots, and numerical comparisons that report PASS or a mismatch.
The example guide links to the main-branch ZIP, including the readiness fixes.
