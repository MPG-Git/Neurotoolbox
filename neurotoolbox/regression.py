from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from .utils import apply_mcc

@dataclass
class RegressionResult:
    table: pd.DataFrame

def run_regression(df: pd.DataFrame,
                   outcome: str,
                   predictors: Sequence[str],
                   covars: Optional[Sequence[str]] = None,
                   standardize: bool = True,
                   add_constant: bool = True,
                   dropna: bool = True) -> pd.DataFrame:
    """Fit OLS and return tidy table per predictor (excluding intercept).
    Columns: Predictor, Beta, SE, CI_low, CI_high, p, Adj_R2, F, F_p, n
    """
    covars = list(covars) if covars else []
    cols = [c for c in predictors + covars if c in df.columns]
    use = df[[outcome] + cols].dropna() if dropna else df[[outcome] + cols]
    if use.empty or len(use) < 8:
        return pd.DataFrame()

    X = use[cols].copy()
    y = use[outcome].astype(float).copy()

    if standardize:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols, index=use.index)
        y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).ravel(), index=use.index, name=outcome)

    if add_constant:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    ci = model.conf_int()
    rows = []
    for name in predictors:
        if name in model.params.index:
            rows.append({
                "Predictor": name,
                "Beta": float(model.params[name]),
                "SE": float(model.bse[name]),
                "CI_low": float(ci.loc[name, 0]),
                "CI_high": float(ci.loc[name, 1]),
                "p": float(model.pvalues[name])
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Adj_R2"] = float(model.rsquared_adj)
    out["F"] = float(model.fvalue) if model.fvalue is not None else np.nan
    out["F_p"] = float(model.f_pvalue) if model.f_pvalue is not None else np.nan
    out["n"] = int(model.nobs)
    out["y"] = outcome
    out["predictors"] = ",".join(predictors)
    out["covars"] = ",".join(covars) if covars else ""
    return out

def run_family(df: pd.DataFrame,
               outcomes: Sequence[str],
               predictor_sets: Dict[str, Sequence[str]],
               covars: Optional[Sequence[str]] = None,
               standardize: bool = True,
               model_label: str = "") -> pd.DataFrame:
    """Run a family of regressions: for each outcome and each predictor in each set.
    predictor_sets maps a display name (e.g., 'Asymmetry', 'Deviation') to a list of predictors.
    Returns long-form table with Outcome, Model, Target (= predictor), and tidy stats.
    """
    rows = []
    for y in outcomes:
        for set_name, preds in predictor_sets.items():
            for p in preds:
                tab = run_regression(df, outcome=y, predictors=[p], covars=covars, standardize=standardize)
                if tab.empty:
                    continue
                row = tab.iloc[0].to_dict()
                row["Outcome"] = y
                row["Model"] = f"{model_label}{set_name}".strip()
                row["Target"] = p
                rows.append(row)
    return pd.DataFrame(rows)

def add_corrections(summary: pd.DataFrame,
                    group_cols: Sequence[str] = ("Outcome", "Model"),
                    alpha: float = 0.05) -> pd.DataFrame:
    if summary.empty:
        return summary
    corrected = apply_mcc(summary, list(group_cols), pcol="p", methods=("fdr_bh", "bonferroni"), alpha=alpha)
    return corrected
