from __future__ import annotations
import logging
import os
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # avoid duplicate handlers
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    den = np.where(np.abs(den) < eps, np.nan, den)
    return num / den

def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR control. Returns (reject, p_fdr)."""
    rej, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return rej, p_fdr

def apply_mcc(df: pd.DataFrame,
              group_cols: List[str],
              pcol: str = "p",
              methods: Tuple[str, ...] = ("fdr_bh", "bonferroni"),
              alpha: float = 0.05) -> pd.DataFrame:
    """Apply multiple-comparisons corrections within groups defined by group_cols.
    Adds columns: p_fdr, p_bonf, sig_fdr, sig_bonf.
    """
    df = df.copy()
    if "p_fdr" not in df.columns:
        df["p_fdr"] = np.nan
    if "p_bonf" not in df.columns:
        df["p_bonf"] = np.nan
    df["sig_fdr"] = False
    df["sig_bonf"] = False

    if not group_cols:
        groups = [((), df.index)]
    else:
        groups = list(df.groupby(group_cols).groups.items())

    for _, idx in groups:
        p = df.loc[idx, pcol].values.astype(float)
        n = np.isfinite(p).sum()
        if n == 0:
            continue
        if "fdr_bh" in methods:
            rej, pfdr = fdr_bh(p, alpha=alpha)
            df.loc[idx, "p_fdr"] = pfdr
            df.loc[idx, "sig_fdr"] = rej
        if "bonferroni" in methods:
            pbon = np.minimum(p * n, 1.0)
            df.loc[idx, "p_bonf"] = pbon
            df.loc[idx, "sig_bonf"] = pbon < alpha
    return df
