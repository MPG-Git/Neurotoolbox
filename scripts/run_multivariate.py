#!/usr/bin/env python3
"""Run multivariate (PLS/CCA) and network covariance analysis.

Example:
python scripts/run_multivariate.py \
  --data TEPSData.csv \
  --outdir outputs_multivariate \
  --rois lcaud rcaud lput rput lNACC rNACC lamyg ramyg \
  --items teps1 teps2 teps3 teps4 teps5 teps6 \
  --covars age gender estICV \
  --residualize
"""
import argparse, os
import pandas as pd
from neurotoolbox.utils import ensure_dir, get_logger
from neurotoolbox.multivariate import residualize_columns, run_pls, run_cca
from neurotoolbox.networks import structural_covariance, fdr_edges, build_graph, node_metrics, global_metrics, plot_corr_heatmap, plot_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rois", nargs="+", required=True, help="ROI columns to use")
    ap.add_argument("--items", nargs="+", required=True, help="Behavioral items/variables for multivariate")
    ap.add_argument("--covars", nargs="*", default=[])
    ap.add_argument("--residualize", action="store_true", help="Residualize ROIs for covariates before PLS/CCA")
    ap.add_argument("--do-cca", action="store_true", help="Run CCA in addition to PLS")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    log = get_logger("run_multivariate", log_file=os.path.join(args.outdir, "analysis.log"))
    df = pd.read_csv(args.data)

    # ROI matrix (optionally residualized)
    if args.residualize and args.covars:
        log.info("Residualizing %d ROIs by covars=%s", len(args.rois), args.covars)
        roi_df = residualize_columns(df, args.rois, args.covars)
    else:
        roi_df = df[args.rois].copy()
    roi_df = roi_df.dropna()

    # Items matrix (z-scored inside run_pls/cca)
    items_df = df.loc[roi_df.index, args.items].dropna()
    common_idx = roi_df.index.intersection(items_df.index)
    roi_df = roi_df.loc[common_idx]
    items_df = items_df.loc[common_idx]

    # PLS
    log.info("Running PLS...")
    pls_out = run_pls(roi_df, items_df, n_components=1, outdir=args.outdir, prefix="PLS")
    log.info("PLS r=%.3f, p=%.3f", pls_out["r"], pls_out["p"])

    # CCA (optional)
    if args.do-cca:
        log.info("Running CCA...")
        cca_out = run_cca(roi_df, items_df, n_components=1, outdir=args.outdir, prefix="CCA")
        log.info("CCA r=%.3f, p=%.3f", cca_out["r"], cca_out["p"])

    # Structural covariance network
    log.info("Computing structural covariance matrix and FDR-pruned edges.")
    R = structural_covariance(roi_df)
    R.to_csv(os.path.join(args.outdir, "StructuralCovariance_ROI_corr.csv"), index=True)
    plot_corr_heatmap(R, outpath=os.path.join(args.outdir, "Network_Heatmap_ROIcorr.png"))

    edges = fdr_edges(R)
    edges.to_csv(os.path.join(args.outdir, "StructuralCovariance_edges_FDR.csv"), index=False)

    G = build_graph(edges[edges["keep"]])
    plot_graph(G, outpath=os.path.join(args.outdir, "Network_Graph.png"))
    nm = node_metrics(G); nm.to_csv(os.path.join(args.outdir, "Network_NodeMetrics.csv"), index=False)
    gm = global_metrics(G); gm.to_csv(os.path.join(args.outdir, "Network_GlobalMetrics.csv"), index=False)

    log.info("Done. Outputs in %s", args.outdir)

if __name__ == "__main__":
    main()
