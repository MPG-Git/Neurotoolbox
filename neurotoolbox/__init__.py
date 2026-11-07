"""NeuroToolbox: generalizable brain–behavior analysis utilities.

This package provides modular functions for:
- Asymmetry indices
- Normative modeling (linear residuals, GPR)
- Regression with correction
- Multivariate analyses (PLS/CCA)
- Structural covariance networks
- Robustness checks (permutation, bootstrap, robust regression)
- Visualizations (heatmap, forest, scatter, network)

Author: You
License: MIT
"""
from . import asymmetry, normative, regression, visuals, multivariate, networks, robustness, utils

__all__ = [
    "asymmetry",
    "normative",
    "regression",
    "visuals",
    "multivariate",
    "networks",
    "robustness",
    "utils",
]

__version__ = "0.1.0"
