#!/usr/bin/env python3
"""Run asymmetry and normative analyses on a generic dataset.

Example:
python scripts/run_asymmetry.py \
  --data TEPSData.csv \
  --outdir outputs_asym \
  --roi-pairs lcaud:rcaud lput:rput lNACC:rNACC lamyg:ramyg \
  --outcomes tepanticip tepsconsum \
  --covars age gender estICV \
  --normative linear \
  --composite
"""
import argparse, os, json
import pandas as pd
from neurotoolbox.utils import ensure_dir, get_logger
from neurotoolbox.asymmetry import compute_ai
from neurotoolbox.normative import residual_deviation, gpr_deviation
from neurotoolbox.regression import run_family, add_corrections
from neurotoolbox.visuals import heatmap_betas, forest_plot

def parse_roi_pairs(pairs):
    out = []
    for p in pairs:
        if ":" in p:
            L, R = p.split(":", 1)
            out.append((L, R))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV dataset")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--roi-pairs", nargs="+", required=True, help="Space-separated entries like L1:R1 L2:R2 ...")
    ap.add_argument("--outcomes", nargs="+", required=True, help="Behavioral outcomes (dependent variables)")
    ap.add_argument("--covars", nargs="*", default=[], help="Covariate columns")
    ap.add_argument("--normative", choices=["none","linear","gpr"], default="linear", help="Normative model type")
    ap.add_argument("--composite", action="store_true", help="Compute composite AI (mean of all AIs)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for corrections")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    log = get_logger("run_asymmetry", log_file=os.path.join(args.outdir, "analysis.log"))
    log.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)

    # 1) Asymmetry
    roi_pairs = parse_roi_pairs(args.roi_pairs)
    log.info("Computing asymmetry for %d pairs", len(roi_pairs))
    ai_res = compute_ai(df, roi_pairs, method="halfnorm", composite=args.composite, composite_name="AI_composite")
    for c in ai_res.ai_df.columns:
        df[c] = ai_res.ai_df[c]

    # 2) Normative deviations
    dev_cols = []
    if args.normative != "none":
        rois = sorted(set(sum(([L,R] for L,R in roi_pairs), [])))
        if args.normative == "linear":
            log.info("Computing residual deviations for %d ROIs (covars=%s)", len(rois), args.covars)
            dev = residual_deviation(df, rois=rois, covars=args.covars)
        else:
            log.info("Computing GPR deviations for %d ROIs (covars=%s)", len(rois), args.covars)
            dev = gpr_deviation(df, rois=rois, covars=args.covars)
        for c in dev.deviations.columns:
            df[c] = dev.deviations[c]
            dev_cols.append(c)
        pd.DataFrame(dev.deviations).to_csv(os.path.join(args.outdir, "Normative_Deviations.csv"), index=True)

    # 3) Regressions: Asymmetry & Deviations
    asym_vars = list(ai_res.ai_df.columns)
    predictor_sets = {}
    if asym_vars:
        predictor_sets["Asymmetry"] = asym_vars
    if dev_cols:
        predictor_sets["Deviation"] = dev_cols

    log.info("Running regression families on outcomes=%s", args.outcomes)
    summary = run_family(df, outcomes=args.outcomes, predictor_sets=predictor_sets, covars=args.covars,
                         standardize=True, model_label="")
    summary = add_corrections(summary, group_cols=("Outcome","Model"), alpha=args.alpha)
    summary.to_csv(os.path.join(args.outdir, "Regression_Summary_with_MCC.csv"), index=False)

    # 4) Visualizations
    # Heatmaps by Model
    for model in summary["Model"].unique():
        sub = summary[summary["Model"] == model]
        heatmap_betas(sub, row="Target", col="Outcome",
                      title=f"{model} → Outcomes (β with p)",
                      outpath=os.path.join(args.outdir, f"Heatmap_{model.replace(' ','_')}.png"))
        forest_plot(sub[["Outcome","Target","Beta","CI_low","CI_high","p"]].copy(),
                    title=f"{model} (forest plot)",
                    outpath=os.path.join(args.outdir, f"Forest_{model.replace(' ','_')}.png"),
                    group_by="Outcome")

    log.info("Done. Outputs in %s", args.outdir)

if __name__ == "__main__":
    main()
