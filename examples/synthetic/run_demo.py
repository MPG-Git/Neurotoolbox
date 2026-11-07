"""Exercise the installed package through all three real analysis CLIs."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Check all columns except the deliberately unseeded Theil-Sen estimator.
REFERENCES = [
    ("asymmetry/Regression_Summary_with_MCC.csv", []),
    ("asymmetry/Normative_Deviations.csv", []),
    ("multivariate/StructuralCovariance_ROI_corr.csv", []),
    ("multivariate/StructuralCovariance_edges_FDR.csv", []),
    ("multivariate/PLS_Xweights_c1.csv", []),
    ("multivariate/CCA_Xweights_c1.csv", []),
    ("followups/Followup_Summary.csv", ["beta_TheilSen"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=HERE / "output")
    out = ap.parse_args().outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = HERE / "data/cohort.csv"
    env = dict(os.environ, MPLBACKEND="Agg")
    commands = [
        ("asymmetry", ["--roi-pairs", "L:R", "--outcomes", "y", "--covars", "age", "--normative", "linear", "--composite"]),
        ("multivariate", ["--rois", "L", "R", "Z", "--items", "y", "z", "--covars", "age", "--residualize", "--do-cca"]),
        ("followups", ["--x", "L", "--y", "y", "--group", "gender"]),
    ]
    for name, args in commands:
        subprocess.run([sys.executable, str(REPO / f"scripts/run_{name}.py"), "--data", str(data),
                        "--outdir", str(out / name), *args], check=True, env=env, cwd=out)
    versions = {p: importlib.metadata.version(p) for p in
                ["neurotoolbox", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib"]}
    (out / "versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    rows = []
    for name, excluded in REFERENCES:
        actual = pd.read_csv(out / name).drop(columns=excluded)
        reference = pd.read_csv(HERE / "expected" / name).drop(columns=excluded)
        if list(actual.columns) != list(reference.columns) or actual.shape != reference.shape:
            raise AssertionError(f"Table shape/schema differs: {name}")
        for column in reference:
            a, b = actual[column], reference[column]
            if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
                # A whole component can flip sign in PLS/CCA without changing its meaning.
                if "weights_c1" in name and np.dot(a, b) < 0:
                    a = -a
                passed = np.isfinite(a).all() and np.allclose(a, b, rtol=1e-5, atol=1e-6)
                delta = float(np.max(np.abs(a - b)))
            else:
                passed = a.equals(b)
                delta = None
            rows.append({"file": name, "column": column, "max_absolute_difference": delta, "pass": bool(passed)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "comparison.csv", index=False)
    if not comparison["pass"].all():
        raise AssertionError(f"Reference mismatch; see {out / 'comparison.csv'} and README version notes")
    assert len(list(out.rglob("*.png"))) >= 8, "Expected analysis plots are missing"
    (out / "PASS.txt").write_text("PASS: all three CLI examples matched the bundled numerical references.\n")
    print(f"PASS: synthetic cohort example. Results: {out}")


if __name__ == "__main__":
    main()
