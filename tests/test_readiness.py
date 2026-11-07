import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from neurotoolbox.networks import structural_covariance, fdr_edges, build_graph
from neurotoolbox.robustness import bootstrap_beta
from neurotoolbox.multivariate import run_pls, run_cca

ROOT = Path(__file__).resolve().parents[1]

def test_network_uses_subjects_and_pairwise_missingness():
    rng = np.random.default_rng(7)
    x = rng.normal(size=80)
    df = pd.DataFrame({'a': x, 'b': .5*x + rng.normal(size=80), 'c': rng.normal(size=80)})
    df.loc[:9, 'b'] = np.nan
    edges = fdr_edges(structural_covariance(df))
    complete = df[['a', 'b']].dropna()
    assert edges.iloc[0]['p'] == pytest.approx(pearsonr(complete.a, complete.b).pvalue)
    with pytest.raises(ValueError, match='n_samples'):
        fdr_edges(pd.DataFrame(np.eye(2)))
    assert len(build_graph(edges.iloc[:0], nodes=df.columns)) == 3
    assert fdr_edges(structural_covariance(df[['a']])).empty

def test_bootstrap_recovers_known_slope():
    x = np.arange(20, dtype=float)
    mean, low, high = bootstrap_beta(x, 3*x+2, n_boot=50)
    np.testing.assert_allclose([mean, low, high], 3)

@pytest.mark.parametrize('method,expected', [(run_pls, .48), (run_cca, .735)])
def test_latent_null_refits_do_not_treat_training_scores_as_fixed(method, expected):
    # Independent high-dimensional blocks have inflated fitted correlations.
    # A fixed-score shuffle would report a spuriously tiny p-value here.
    rng=np.random.default_rng(10)
    x=pd.DataFrame(rng.normal(size=(60,12))); y=pd.DataFrame(rng.normal(size=(60,12)))
    result=method(x,y,n_perm=199)
    assert result['p']==pytest.approx(expected)

@pytest.mark.parametrize('script,args,expected', [
 ('run_asymmetry.py', ['--roi-pairs', 'L:R', '--outcomes', 'y', '--covars', 'age', '--normative', 'linear', '--composite'], 'Regression_Summary_with_MCC.csv'),
 ('run_asymmetry.py', ['--roi-pairs', 'L:R', '--outcomes', 'y', '--covars', 'age', '--normative', 'gpr'], 'Normative_Deviations.csv'),
 ('run_multivariate.py', ['--rois', 'L', 'R', 'Z', '--items', 'y', 'z', '--covars', 'age', '--residualize', '--do-cca'], 'StructuralCovariance_edges_FDR.csv'),
 ('run_followups.py', ['--x', 'L', '--y', 'y', '--group', 'gender'], 'Followup_Summary.csv'),
])
def test_installed_cli(tmp_path, script, args, expected):
    rng = np.random.default_rng(23)
    n = 80
    age = rng.normal(30, 5, n); latent = rng.normal(size=n)
    df = pd.DataFrame({'age': age, 'gender': np.arange(n)%2,
        'L': 100+latent+age*.1+rng.normal(size=n),
        'R': 100+latent+age*.15+rng.normal(size=n), 'Z': rng.normal(size=n),
        'y': latent+rng.normal(size=n), 'z': rng.normal(size=n)})
    data = tmp_path/'data.csv'; df.to_csv(data, index=False)
    out = tmp_path/'outputs'
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/script), '--data', str(data), '--outdir', str(out), *args],
        cwd=tmp_path, env={**os.environ, 'MPLBACKEND': 'Agg'}, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    table = pd.read_csv(out/expected)
    assert len(table) > 0
    assert list(out.glob('*.png'))
