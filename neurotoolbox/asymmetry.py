from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .utils import safe_divide

@dataclass
class AsymmetryResult:
    ai_df: pd.DataFrame
    composite: Optional[pd.Series] = None

def compute_ai(df: pd.DataFrame,
               roi_pairs: Sequence[Tuple[str, str]],
               method: str = "frac",
               prefix: str = "AI_",
               composite: bool = False,
               composite_name: str = "AI_composite") -> AsymmetryResult:
    """Compute asymmetry indices for given (L, R) ROI pairs.

    Parameters
    ----------
    df : DataFrame
        Input dataframe with ROI columns.
    roi_pairs : list of (left, right)
        Column names for left/right ROIs.
    method : {'frac', 'halfnorm'}
        'frac': (L - R) / (L + R)
        'halfnorm': (L - R) / ((L + R)/2)
    prefix : str
        Prefix for output AI columns.
    composite : bool
        If True, compute mean of all AI columns per row.
    composite_name : str
        Column name for composite.

    Returns
    -------
    AsymmetryResult
        ai_df: dataframe with AI columns
        composite: optional composite series
    """
    out = pd.DataFrame(index=df.index)
    for L, R in roi_pairs:
        if L not in df.columns or R not in df.columns:
            continue
        num = df[L] - df[R]
        den = (df[L] + df[R]) if method == "frac" else (df[L] + df[R]) / 2.0
        ai = safe_divide(num.to_numpy(dtype=float), den.to_numpy(dtype=float))
        col = f"{prefix}{_short_name(L, R)}"
        out[col] = ai

    comp = None
    if composite and not out.empty:
        comp = out.mean(axis=1, skipna=True)
        out[composite_name] = comp
    return AsymmetryResult(ai_df=out, composite=comp)

def _short_name(L: str, R: str) -> str:
    # common base name from L/R like 'lNACC' / 'rNACC' -> 'nacc'
    def strip_lr(s: str) -> str:
        s2 = s.lower()
        if s2.startswith("l") or s2.startswith("r"):
            return s2[1:]
        return s2
    baseL, baseR = strip_lr(L), strip_lr(R)
    return baseL if baseL == baseR else f"{baseL}_{baseR}"
