from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

@dataclass
class NormativeOutputs:
    deviations: pd.DataFrame
    y_pred: Optional[pd.DataFrame] = None
    y_std: Optional[pd.DataFrame] = None
    model_info: Optional[dict] = None

def residual_deviation(df: pd.DataFrame,
                       rois: Sequence[str],
                       covars: Sequence[str]) -> NormativeOutputs:
    """Compute normative deviations as standardized residuals:
       For each ROI: ROI ~ covars; deviation := z(residual).
    """
    dev = pd.DataFrame(index=df.index)
    for col in rois:
        cols = [c for c in covars if c in df.columns]
        if col not in df.columns:
            continue
        use = df[cols + [col]].dropna()
        s = pd.Series(index=df.index, dtype=float)
        if len(use) < 8:
            dev[col + "_dev"] = s
            continue
        X = sm.add_constant(use[cols]) if cols else np.ones((len(use), 1))
        y = use[col]
        m = sm.OLS(y, X).fit()
        resid = y - m.predict(X)
        z = (resid - resid.mean()) / resid.std(ddof=1)
        s.loc[use.index] = z
        dev[col + "_dev"] = s
    return NormativeOutputs(deviations=dev)

def gpr_deviation(df: pd.DataFrame,
                  rois: Sequence[str],
                  covars: Sequence[str],
                  kernel: Optional[object] = None,
                  alpha: float = 1e-6,
                  normalize_y: bool = True,
                  random_state: Optional[int] = 1) -> NormativeOutputs:
    """Gaussian Process Regression normative model.
    Predict ROI ~ covars, then deviation = (y - y_pred) / y_std.

    Parameters
    ----------
    rois : columns to model
    covars : predictor columns
    kernel : None for default (Constant * RBF + WhiteKernel)
    """
    cols = [c for c in covars if c in df.columns]
    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0) + WhiteKernel()
    Xraw = df[cols].values if cols else np.ones((len(df), 1))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xraw)

    dev = pd.DataFrame(index=df.index)
    ypred_df = pd.DataFrame(index=df.index)
    ystd_df = pd.DataFrame(index=df.index)

    for col in rois:
        if col not in df.columns:
            continue
        y = df[col].values
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, random_state=random_state)
        gpr.fit(Xs, y)
        y_pred, y_std = gpr.predict(Xs, return_std=True)
        ystd_safe = np.where(y_std <= np.finfo(float).eps, np.nan, y_std)
        dev[col + "_dev"] = (y - y_pred) / ystd_safe
        ypred_df[col] = y_pred
        ystd_df[col] = y_std

    return NormativeOutputs(deviations=dev, y_pred=ypred_df, y_std=ystd_df,
                            model_info={"kernel": str(kernel)})
