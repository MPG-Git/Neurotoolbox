#!/usr/bin/env python3
"""Run robustness follow-ups for a given x predictor and y outcome.

Example:
python scripts/run_followups.py \
  --data TEPSData.csv \
  --outdir outputs_followups \
  --x AI_nacc \
  --y tepsconsum \
  --group gender
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neurotoolbox.utils import ensure_dir, get_logger
from neurotoolbox.robustness import permutation_test_corr, bootstrap_beta, robust_regressions, nonparametric_corr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x", required=True, help="Predictor column")
    ap.add_argument("--y", required=True, help="Outcome column")
    ap.add_argument("--group", default=None, help="Optional grouping column for stratified scatter")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    log = get_logger("run_followups", log_file=os.path.join(args.outdir, "analysis.log"))
    df = pd.read_csv(args.data)

    use = df[[args.x, args.y] + ([args.group] if args.group else [])].dropna()
    if use.empty:
        log.error("No complete cases for x=%s, y=%s", args.x, args.y)
        return

    x = use[args.x].values.astype(float)
    y = use[args.y].values.astype(float)

    # 1) Permutation test
    r_obs, p_perm = permutation_test_corr(x, y)
    # 2) Bootstrap CI for beta
    _, ci_low, ci_high = bootstrap_beta(x, y)
    # 3) Robust regressions
    fits = robust_regressions(x, y)
    # 4) Nonparametric correlations
    nonpar = nonparametric_corr(x, y)

    summary = {
        "r_obs": r_obs,
        "p_perm": p_perm,
        "boot_CI_low": ci_low,
        "boot_CI_high": ci_high,
        **{f"beta_{k}": v for k,v in fits.items()},
        **nonpar
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "Followup_Summary.csv"), index=False)

    # Scatter with multiple fits per group (if any)
    plt.figure(figsize=(6,5), dpi=300)
    if args.group and args.group in use.columns:
        for g, sub in use.groupby(args.group):
            xs = sub[args.x].values
            ys = sub[args.y].values
            plt.scatter(xs, ys, alpha=0.7, edgecolor="white", linewidth=0.5, label=f"{args.group}={g}")
    else:
        plt.scatter(x, y, alpha=0.7, edgecolor="white", linewidth=0.5, label="All")
    # global OLS fit
    coef = np.polyfit(x, y, 1)
    xs_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs_line, coef[0]*xs_line + coef[1], lw=2, label=f"OLS (β={coef[0]:.2f})")
    plt.axvline(0, ls="--", color="gray")
    plt.xlabel(args.x); plt.ylabel(args.y)
    plt.title(f"{args.x} → {args.y}\nr={r_obs:.2f}, perm p={p_perm:.3f}; Spearman ρ={nonpar['spearman_rho']:.2f}")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "Scatter_with_OLS.png"), dpi=300)
    plt.close()

    log.info("Done. Outputs in %s", args.outdir)

if __name__ == "__main__":
    main()
