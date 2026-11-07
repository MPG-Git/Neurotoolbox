from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.base import clone
import matplotlib.pyplot as plt

from .visuals import scatter_with_fit

def residualize_columns(df: pd.DataFrame, cols: Sequence[str], covars: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if not covars:
        return df[cols].copy()
    tmp = df[cols + list(covars)].dropna()
    for col in cols:
        sub = tmp[[col] + list(covars)].dropna()
        if sub.empty:
            out[col] = np.nan
            continue
        X = sm.add_constant(sub[list(covars)])
        y = sub[col]
        m = sm.OLS(y, X).fit()
        resid = y - m.predict(X)
        out.loc[sub.index, col] = resid
    return out

def _perm_corr(x: np.ndarray, y: np.ndarray, n_perm: int = 2000, seed: int = 1) -> Tuple[float, float]:
    rng = check_random_state(seed)
    obs = np.corrcoef(x, y)[0, 1]
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = np.corrcoef(x, yp)[0, 1]
        if abs(r) >= abs(obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return obs, p

def _model_permutation(model, X, Y, observed, n_perm=2000, seed=1):
    """Refit latent components for every row permutation of the Y block."""
    rng = check_random_state(seed)
    count = 0
    for _ in range(n_perm):
        yp = Y[rng.permutation(len(Y))]
        fitted = clone(model).fit(X, yp)
        if isinstance(fitted, PLSRegression):
            xs, ys = fitted.x_scores_[:, 0], fitted.y_scores_[:, 0]
        else:
            a, b = fitted.transform(X, yp)
            xs, ys = a[:, 0], b[:, 0]
        if abs(np.corrcoef(xs, ys)[0, 1]) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)


def run_pls(X: pd.DataFrame, Y: pd.DataFrame, n_components: int = 1,
            outdir: Optional[str] = None, prefix: str = "PLS", n_perm: int = 2000, seed: int = 1) -> dict:
    Xs = StandardScaler().fit_transform(X.values)
    Ys = StandardScaler().fit_transform(Y.values)
    pls = PLSRegression(n_components=min(n_components, Xs.shape[1], Ys.shape[1]))
    pls.fit(Xs, Ys)
    xs = pls.x_scores_[:, 0]
    ys = pls.y_scores_[:, 0]
    r = np.corrcoef(xs, ys)[0, 1]
    p = _model_permutation(pls, Xs, Ys, r, n_perm, seed)
    out = {
        "x_weights": pd.Series(pls.x_weights_[:,0], index=X.columns, name=f"{prefix}_xw_c1"),
        "y_weights": pd.Series(pls.y_weights_[:,0], index=Y.columns, name=f"{prefix}_yw_c1"),
        "r": float(r), "p": float(p)
    }
    if outdir:
        out["x_weights"].to_csv(f"{outdir}/{prefix}_Xweights_c1.csv")
        out["y_weights"].to_csv(f"{outdir}/{prefix}_Yweights_c1.csv")
        scatter_with_fit(xs, ys, "Brain LV1", "Behavior LV1",
                         title=f"{prefix} scores: r={r:.2f}, p={p:.3f}",
                         outpath=f"{outdir}/{prefix}_Scores_c1.png")
    return out

def run_cca(X: pd.DataFrame, Y: pd.DataFrame, n_components: int = 1,
            outdir: Optional[str] = None, prefix: str = "CCA", n_perm: int = 2000, seed: int = 1) -> dict:
    Xs = StandardScaler().fit_transform(X.values)
    Ys = StandardScaler().fit_transform(Y.values)
    ncomp = min(n_components, Xs.shape[1], Ys.shape[1])
    cca = CCA(n_components=ncomp, max_iter=5000)
    cca.fit(Xs, Ys)
    Xc, Yc = cca.transform(Xs, Ys)
    r = np.corrcoef(Xc[:,0], Yc[:,0])[0, 1]
    p = _model_permutation(cca, Xs, Ys, r, n_perm, seed)
    out = {
        "x_weights": pd.Series(cca.x_weights_[:,0], index=X.columns, name=f"{prefix}_xw_c1"),
        "y_weights": pd.Series(cca.y_weights_[:,0], index=Y.columns, name=f"{prefix}_yw_c1"),
        "r": float(r), "p": float(p)
    }
    if outdir:
        out["x_weights"].to_csv(f"{outdir}/{prefix}_Xweights_c1.csv")
        out["y_weights"].to_csv(f"{outdir}/{prefix}_Yweights_c1.csv")
        scatter_with_fit(Xc[:,0], Yc[:,0], "Brain can1", "Behavior can1",
                         title=f"{prefix} scores: r={r:.2f}, p={p:.3f}",
                         outpath=f"{outdir}/{prefix}_Scores_c1.png")
    return out
