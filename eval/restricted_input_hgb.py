"""Matched Gaia-anchored feature ablation; historical inputs are immutable."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import full_physical_rebuild as original
from hgb_reference_repair import sha256, write_json, per_star

ROOT = Path('eval/restricted_input_hgb_20261013')
BASE = Path('eval/full_rebuild_20261012')
REMOVED = {
    'target variability / upstream PDC diagnostics': ['cdpp1_0', 'pdcvar', 'pdc_noi', 'pr_wght2', 'pdc_corp', 'pdc_totp', 'pdctotp_x_crowdsap', 'dev_pdcvar_loo'],
    'target flux-weighted centroids and deviations': ['mom_centr1_rms_default', 'mom_centr2_rms_default', 'dev_mom_centr1_rms_default', 'dev_mom_centr2_rms_default'],
    'target position errors with uncertain provenance': ['perr_col', 'perr_row'],
    'target quality/availability summaries': ['quality_zero_fraction', 'default_quality_fraction', 'n_cadences', 'scatter_flag_frac'],
}

def freeze():
    ROOT.mkdir(exist_ok=True)
    features = json.loads((BASE/'physical/complete.json').read_text())['features']
    removed = sum(REMOVED.values(), [])
    assert set(removed) <= set(features)
    inputs = [BASE/'features.parquet', BASE/'freeze.json', BASE/'preparation.json',
              BASE/'cohort_manifest.parquet', BASE/'reference_bank.parquet', original.SOURCE,
              Path('research_defaults.json')]
    inputs = [p for p in inputs if p.exists()]
    protocol = dict(seed=original.SEED, params=original.PARAMS, rounds=5, target='gaia_rp',
        features={'full': features, 'restricted': [x for x in features if x not in removed]},
        removed=REMOVED, input_sha256={str(p): sha256(p) for p in inputs},
        anchor='Training-only existing physical-group-safe Gaia RP/color transfer; no per-star target centering at any round. Report missing-anchor fallbacks.',
        retained='Catalog/detector/aperture/PRF geometry, ensemble pointing, backgrounds, thermal context, source-excluded references and their existing derived features. Geometry-centered 0..13 use geometry/color only; slot 2 is the original constant-zero baseline-strength input.',
        limits='Backgrounds, aperture choice, pipeline masks and catalog crowding may still respond indirectly to astrophysics. This restricted set is not a causal instrumental-only certificate. All-sector context is transductive.',
        training='Same exact qualified fit_row population, ordering, seed, capacity and per-column training medians as original; full control is independently refit.',
        selection='Five rounds fixed in advance. Select lower median per-star CV on qualified validation evaluation_group=calibration only; report both candidates on confirmation/development test. No promotion based on science examples.',
        science='Unchanged existing ten external-ground sources and 30 detached binaries, both product families. Preserve early calibration cutoffs, truth, periods, templates, matching and eligibility. Report all and qualified with attrition. These are examined development examples.',
        injections='Existing six-variable and nine-quiet conditional scenarios: recompute centroid and PDCVAR proxy paths from actual cadences. Full-cadence amplitude projection and added-offset distortion. Restricted invariance on removed paths is structural, not end-to-end evidence; upstream PDC not rerun.',
        created_utc=pd.Timestamp.now(tz='UTC').isoformat())
    path=ROOT/'freeze.json'
    if path.exists():
        old=json.loads(path.read_text())
        for key in ['features','input_sha256','seed','params']: assert old[key]==protocol[key], key
        return
    write_json(path,protocol)
    (ROOT/'features.parquet').symlink_to((BASE/'features.parquet').resolve())
    snap=ROOT/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(original.__file__),Path('eval/calibration_bank_study.py'),Path('eval/fit_geometry_training_transfer.py'),Path('eval/test_variable_injection.py'),Path('research_defaults.json')]: shutil.copy2(p,snap/p.name)

def fit(case):
    freeze()
    original.ROOT=ROOT
    original.fit(case)

class Model:
    def __init__(self,case):
        self.root=ROOT/case
        self.columns=json.loads((self.root/'complete.json').read_text())['features']
        self.impute=joblib.load(self.root/'imputation.pkl')
        self.models=[joblib.load(self.root/f'round_{i}.pkl') for i in range(5)]
    def predict_log(self,f):
        x=f[self.columns].replace([np.inf,-np.inf],np.nan).fillna(self.impute).to_numpy(np.float32)
        return np.log(f.pred_knn.to_numpy())+sum(m.predict(x) for m in self.models)

def score():
    original.ROOT=BASE
    f=original.load_frame();f=f[f.role.ne('train')].copy()
    for case in ['full','restricted']:
        p=pd.read_parquet(ROOT/case/'predictions.parquet').set_index('source_row')
        f[case]=p.log_correction_r5.reindex(f.source_row).to_numpy()
    records=[];individual=[]
    def measure(group):
        g=f[f.evaluation_group.eq(group)] if group!='test' else f[f.role.eq('test')]
        for view in ['all','qualified']:
            z=g if view=='all' else g[g.systematic_score.gt(.95)]
            for case in ['full','restricted']:
                s=per_star(z,np.exp(z[case].to_numpy()))
                records.append(dict(group=group,view=view,model=case,rows=len(z),stars=z.tic_id.nunique(),metric_stars=len(s),missing_anchors=int((~z.has_gaia_anchor).sum()),median_cv_pct=float(s.median())))
                individual.extend(dict(group=group,view=view,model=case,tic_id=int(t),cv_pct=float(v)) for t,v in s.items())
    measure('calibration')
    eligible=[r for r in records if r['view']=='qualified']
    write_json(ROOT/'selection.json',dict(selected=min(eligible,key=lambda r:r['median_cv_pct'])['model'],calibration=records.copy(),basis='Qualified validation calibration only; five rounds fixed; no production promotion'))
    measure('confirmation');measure('test')
    pd.DataFrame(records).to_csv(ROOT/'quiet_metrics.csv',index=False)
    pd.DataFrame(individual).to_parquet(ROOT/'quiet_per_star.parquet',index=False)
    old=pd.read_parquet(BASE/'physical/predictions.parquet').set_index('source_row')
    error=float(np.max(abs(f.full.to_numpy()-old.log_correction_r5.reindex(f.source_row).to_numpy())))
    assert error<1e-10, error
    write_json(ROOT/'training_verification.json',dict(full_control_max_log_prediction_difference=error,
        full_features=120,restricted_features=len(Model('restricted').columns),
        historical_features_sha256=sha256(BASE/'features.parquet'),anchor_retained=True))
    print(pd.DataFrame(records).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','full','restricted','score']);a=p.parse_args()
    if a.action=='freeze': freeze()
    elif a.action=='score': score()
    else: fit(a.action)
