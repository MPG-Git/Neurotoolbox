from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def heatmap_betas(summary: pd.DataFrame,
                  row: str,
                  col: str,
                  beta_col: str = "Beta",
                  p_col: str = "p",
                  title: str = "",
                  outpath: Optional[str] = None) -> None:
    """Draw a heatmap of betas annotated with p-values."""
    if summary.empty:
        return
    mat = summary.pivot_table(index=row, columns=col, values=beta_col, aggfunc="mean")
    pmat = summary.pivot_table(index=row, columns=col, values=p_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.35*len(mat.index)+1)))
    vmax = np.nanmax(np.abs(mat.values)) or 1.0
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Standardized β")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            b = mat.values[i, j]
            p = pmat.values[i, j]
            if not (np.isnan(b) or np.isnan(p)):
                ax.text(j, i, f"{b:.2f}\np={p:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

def forest_plot(summary: pd.DataFrame,
                title: str,
                outpath: Optional[str] = None,
                group_by: Optional[str] = None) -> None:
    """Forest plot of betas with 95% CI. Optionally grouped/sorted by a column."""
    if summary.empty:
        return
    df = summary.copy()
    if group_by and group_by in df.columns:
        df.sort_values([group_by, df["Beta"].abs().name], ascending=[True, False], inplace=True)
    else:
        df.sort_values(df["Beta"].abs().name, ascending=False, inplace=True)
    df["row"] = range(len(df))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35*len(df))), dpi=300)
    for _, r in df.iterrows():
        y = r["row"]
        ax.plot([r["CI_low"], r["CI_high"]], [y, y])
        ax.scatter(r["Beta"], y, zorder=3)
        ax.text(r["CI_high"] + 0.05, y, f"p={r['p']:.3f}", va="center", fontsize=8)
        label = f"{r.get('Target', r.get('Predictor', ''))} ({r.get('Outcome','')})"
        ax.text(min(r["CI_low"], r["CI_high"]) - 0.35, y, label, va="center", fontsize=8)
    ax.axvline(0, ls="--", color="gray")
    ax.set_yticks([])
    ax.set_xlabel("Standardized β (95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

def scatter_with_fit(x: np.ndarray, y: np.ndarray,
                     xlabel: str, ylabel: str,
                     title: str = "",
                     outpath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(4.5,4), dpi=300)
    ax.scatter(x, y, alpha=0.8, edgecolors="white", linewidths=0.5)
    if len(x) > 1:
        m = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(xs, m[0]*xs + m[1], lw=1)
    ax.axvline(0, ls="--", color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300)
        plt.close(fig)
