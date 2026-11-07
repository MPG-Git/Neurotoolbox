from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import t as tdist

def structural_covariance(roi_df: pd.DataFrame) -> pd.DataFrame:
    result = roi_df.corr(method="pearson")
    valid = roi_df.notna().astype(int)
    result.attrs["pairwise_n"] = valid.T.dot(valid)
    return result

def fdr_edges(R: pd.DataFrame, alpha: float = 0.05,
              n_samples: Optional[int] = None) -> pd.DataFrame:
    """Test correlations using participant counts, with BH correction.

    Pass a matrix from structural_covariance, or supply n_samples for a
    complete-case correlation matrix loaded from disk.
    """
    rois = list(R.columns)
    counts = R.attrs.get("pairwise_n")
    if counts is None and (n_samples is None or n_samples < 3):
        raise ValueError("Supply n_samples >= 3 or use structural_covariance(data).")
    rows = []
    for i, a in enumerate(rois):
        for j, b in enumerate(rois):
            if j <= i:
                continue
            r = float(R.loc[a, b])
            n = int(counts.loc[a, b]) if counts is not None else n_samples
            if n < 3 or not np.isfinite(r):
                p = np.nan
            else:
                t = r * np.sqrt((n - 2) / max(1e-15, 1 - r**2))
                p = 2 * tdist.sf(abs(t), df=n-2)
            rows.append((a, b, r, p))
    df = pd.DataFrame(rows, columns=["roi_i","roi_j","r","p"])
    df["p_fdr"] = np.nan
    df["keep"] = False
    finite = df["p"].notna()
    if finite.any():
        rej, pfdr, _, _ = multipletests(df.loc[finite, "p"], method="fdr_bh", alpha=alpha)
        df.loc[finite, "p_fdr"] = pfdr
        df.loc[finite, "keep"] = rej
    return df

def build_graph(edges: pd.DataFrame, nodes: Optional[Sequence[str]] = None) -> nx.Graph:
    G = nx.Graph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    for _, r in edges.iterrows():
        if r.get("keep", True):
            G.add_edge(r["roi_i"], r["roi_j"], weight=r["r"])
    return G

def node_metrics(G: nx.Graph) -> pd.DataFrame:
    deg = dict(G.degree())
    strength = {n: sum(abs(d["weight"]) for _,_,d in G.edges(n, data=True)) for n in G.nodes()}
    clust = nx.clustering(G, weight="weight")
    return pd.DataFrame({
        "ROI": list(G.nodes()),
        "degree": [deg.get(n,0) for n in G.nodes()],
        "strength": [strength.get(n,0.0) for n in G.nodes()],
        "clustering": [clust.get(n,0.0) for n in G.nodes()],
    })

def global_metrics(G: nx.Graph) -> pd.DataFrame:
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        comms = greedy_modularity_communities(G, weight="weight")
        Q = modularity(G, comms, weight="weight")
    except Exception:
        Q = np.nan
    avgC = np.mean(list(nx.clustering(G, weight="weight").values())) if len(G) else np.nan
    return pd.DataFrame([{
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_clustering": avgC,
        "modularity_Q": Q
    }])

def plot_corr_heatmap(R: pd.DataFrame, outpath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    im = ax.imshow(R.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(R.columns))); ax.set_xticklabels(R.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(R.index))); ax.set_yticklabels(R.index)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("r")
    ax.set_title("Structural covariance (Pearson r)")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300); plt.close(fig)

def plot_graph(G: nx.Graph, outpath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(6,5), dpi=300)
    pos = nx.spring_layout(G, seed=1, weight="weight")
    ew = [abs(d.get("weight", 1.0))*3 for *_, d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=ew, alpha=0.7)
    ax.set_title("Structural covariance network (FDR-kept edges)")
    ax.axis("off")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=300); plt.close(fig)

def subject_projection(roi_df: pd.DataFrame,
                       weights: pd.Series) -> pd.Series:
    w = weights.reindex(roi_df.columns).fillna(0.0).values
    denom = np.linalg.norm(w) + 1e-9
    return pd.Series((roi_df.values * w).sum(axis=1) / denom, index=roi_df.index, name="projection")
