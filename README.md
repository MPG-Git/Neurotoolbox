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
pip install -r requirements.txt
# (optional) install the package in editable mode
pip install -e .
```

## Data expectations

Your CSV can use **any column names**. You specify:
- Behavioral outcomes: e.g., `--outcomes tepanticip tepsconsum`
- ROI left/right pairs: e.g., `--roi-pairs lNACC:rNACC lcaud:rcaud`
- Covariates: e.g., `--covars age gender estICV`

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

- All regressions are **standardized** by default (X and y), and we report standardized β.
- FDR/BH and Bonferroni are applied **within each (Outcome × Model)** family.
- Network modularity uses greedy modularity; if it fails (small graphs), the code falls back gracefully.
- Figures are generated with matplotlib (no style dependencies).

## License

MIT
