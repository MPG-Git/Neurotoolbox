from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.utils import check_random_state
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kendalltau

def permutation_test_corr(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 1) -> Tuple[float, float]:
    rng = check_random_state(seed)
    r_obs, _ = pearsonr(x, y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r, _ = pearsonr(x, yp)
        if abs(r) >= abs(r_obs):
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)
    return float(r_obs), float(p_perm)

def bootstrap_beta(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 2) -> Tuple[float, float, float]:
    rng = check_random_state(seed)
    betas = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        xb, yb = x[idx], y[idx]
        X = sm.add_constant(xb)
        m = sm.OLS(yb, X).fit()
        betas.append(m.params[1])
    ci_low, ci_high = np.percentile(betas, [2.5, 97.5])
    return float(np.mean(betas)), float(ci_low), float(ci_high)

def robust_regressions(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    X = x.reshape(-1,1)
    Y = y
    out = {}
    out["OLS"] = LinearRegression().fit(X, Y).coef_[0]
    out["Huber"] = HuberRegressor().fit(X, Y).coef_[0]
    out["TheilSen"] = TheilSenRegressor().fit(X, Y).coef_[0]
    try:
        ransac = RANSACRegressor(estimator=LinearRegression(), random_state=1).fit(X, Y)
    except TypeError:
        ransac = RANSACRegressor(base_estimator=LinearRegression(), random_state=1).fit(X, Y)
    out["RANSAC"] = getattr(ransac.estimator_, "coef_", [np.nan])[0]
    return {k: float(v) for k, v in out.items()}

def nonparametric_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    rho, p_s = spearmanr(x, y)
    tau, p_k = kendalltau(x, y)
    return {"spearman_rho": float(rho), "spearman_p": float(p_s), "kendall_tau": float(tau), "kendall_p": float(p_k)}

def moderation(df: pd.DataFrame, y: str, x: str, moderator: str, covars: Optional[Sequence[str]] = None) -> pd.DataFrame:
    covars = list(covars) if covars else []
    cols = [c for c in [y, x, moderator] + covars if c in df.columns]
    use = df[cols].dropna()
    if len(use) < 20:
        return pd.DataFrame()
    d = use.copy()
    if moderator != "gender":
        d[moderator] = d[moderator] - d[moderator].mean()
    if x != "gender":
        d[x] = d[x] - d[x].mean()
    d["interaction"] = d[x] * d[moderator]
    X = sm.add_constant(d[[x, moderator, "interaction"] + covars])
    m = sm.OLS(d[y], X).fit()
    ci = m.conf_int()
    rows = []
    for name in ["interaction", x, moderator]:
        if name in m.params.index:
            rows.append({
                "Term": name,
                "Beta": float(m.params[name]),
                "SE": float(m.bse[name]),
                "CI_low": float(ci.loc[name,0]),
                "CI_high": float(ci.loc[name,1]),
                "p": float(m.pvalues[name])
            })
    out = pd.DataFrame(rows)
    out["Adj_R2"] = float(m.rsquared_adj)
    out["F"] = float(m.fvalue) if m.fvalue is not None else np.nan
    out["F_p"] = float(m.f_pvalue) if m.f_pvalue is not None else np.nan
    out["n"] = int(m.nobs)
    return out

def mediation_bootstrap(df: pd.DataFrame, x: str, mvar: str, y: str, covars: Optional[Sequence[str]] = None,
                        n_boot: int = 5000, seed: int = 1) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    covars = list(covars) if covars else []
    cols = [c for c in [x, mvar, y] + covars if c in df.columns]
    use = df[cols].dropna()
    if len(use) < 20:
        return {"n": len(use), "ab": np.nan, "ab_CI_low": np.nan, "ab_CI_high": np.nan}
    Xm = sm.add_constant(use[[x] + covars])
    Mm = sm.OLS(use[mvar], Xm).fit()
    a_hat = float(Mm.params.get(x, np.nan))
    Xy = sm.add_constant(use[[x, mvar] + covars])
    My = sm.OLS(use[y], Xy).fit()
    b_hat = float(My.params.get(mvar, np.nan))
    cprime = float(My.params.get(x, np.nan))

    ab_boot = []
    n = len(use)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        db = use.iloc[idx]
        Xm_b = sm.add_constant(db[[x] + covars])
        Mm_b = sm.OLS(db[mvar], Xm_b).fit()
        a_b = float(Mm_b.params.get(x, np.nan))
        Xy_b = sm.add_constant(db[[x, mvar] + covars])
        My_b = sm.OLS(db[y], Xy_b).fit()
        b_b = float(My_b.params.get(mvar, np.nan))
        ab_boot.append(a_b * b_b)
    ab_boot = np.array(ab_boot, dtype=float)
    ci_low, ci_high = np.nanpercentile(ab_boot, [2.5, 97.5])
    return {
        "n": n, "a": a_hat, "b": b_hat, "ab": float(a_hat*b_hat),
        "ab_CI_low": float(ci_low), "ab_CI_high": float(ci_high), "c_prime": cprime,
        "model_M_R2": float(Mm.rsquared), "model_Y_R2": float(My.rsquared_adj)
    }
