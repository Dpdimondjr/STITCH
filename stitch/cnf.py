"""Frozen pdcsap-branch CNF102 application with a per-sector sigma (added 2026-11-02).

The point estimate is the analytic conditional median of the flow (the quantity every protocol
metric used). The sigma is the standard deviation of the log correction over N_SAMPLES seeded flow
samples; on qualified quiet confirmation stars it is calibrated in scale (|z|<1: 0.80, |z|<2: 0.97,
heavy tails) and ranks per-star residuals (Spearman 0.65). Two quality flags are frozen from the
qualified quiet confirmation population (eval/sigma_cut_20261023): `sigma_keep80` and
`sigma_keep50` mark sectors whose sigma is below the 80 % and 50 % population quantiles. They
trade coverage for precision, like TARS qualification; report them with attrition.

Nothing here fits, selects or promotes a model. The pdcsap reference bank, colour folds and
imputer are the ones frozen with the October 15 SAP sequence; new sources must be absent from the
historical cohort (checked upstream by `FrozenPhysicalModel.check_new_identity`).
"""
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import pandas as pd

from .frozen import REPO, legacy_imports, sha256

CNF102 = REPO / 'eval/sap_gaia_cnf_20261017/pdcsap'
SAPREF = REPO / 'eval/sap_ordered_validation_20261015'
SIGMA_POPULATION = REPO / 'eval/sigma_cut_20261023/sectors.parquet'
THRESHOLDS = REPO / 'eval/cnf_sigma_release_20261102/thresholds.json'
N_SAMPLES = 256
SEED = 20261102


def frozen_thresholds():
    """Population sigma quantiles of CNF102 on qualified quiet confirmation star-sectors."""
    if THRESHOLDS.exists():
        return json.loads(THRESHOLDS.read_text())
    s = pd.read_parquet(SIGMA_POPULATION, columns=['qualified', 'cnf102_sd', 'tic_id'])
    s = s[s.qualified & s.cnf102_sd.notna()]
    t = dict(source=str(SIGMA_POPULATION.relative_to(REPO)), source_sha256=sha256(SIGMA_POPULATION),
             population='qualified quiet confirmation star-sectors', rows=int(len(s)), stars=int(s.tic_id.nunique()),
             sigma_units='per cent of flux (100 x std of log correction over %d samples)' % N_SAMPLES,
             keep80_sigma_pct=float(s.cnf102_sd.quantile(.8)), keep50_sigma_pct=float(s.cnf102_sd.quantile(.5)),
             median_sigma_pct=float(s.cnf102_sd.median()),
             created_utc=pd.Timestamp.now(tz='UTC').isoformat())
    THRESHOLDS.parent.mkdir(exist_ok=True)
    THRESHOLDS.write_text(json.dumps(t, indent=2) + '\n')
    return t


class PdcsapCNF:
    def __init__(self, model_dir=CNF102, reference_root=SAPREF, n_samples=N_SAMPLES):
        import torch
        legacy_imports()
        import evaluate_matched_sap_hgb as sap
        from train_restricted_gaia_cnf import Model
        sap.ROOT = Path(reference_root)
        self.sap = sap
        self.root = Path(model_dir)
        self.model = Model(self.root)
        self.n_samples = n_samples
        self.thresholds = frozen_thresholds()
        self.features = list(self.model.prep['features'])
        self.hashes = {n: sha256(self.root / n) for n in ['best_cv.pt', 'selection.json', 'preprocessing.pkl']}
        self.hashes.update({f'pdcsap/{n}': sha256(Path(reference_root) / 'pdcsap' / n)
                            for n in ['reference_bank.parquet', 'imputation.pkl'] + [f'color_fold_{i}.pkl' for i in range(5)]})
        support = pd.read_parquet(Path(reference_root) / 'pdcsap/features_and_targets.parquet',
                                  columns=['fit_row', 'tmag', 'bp_rp', 'sector'])
        support = support[support.fit_row.astype(bool)]
        self.ranges = {n: [float(support[n].min()), float(support[n].max())] for n in ['tmag', 'bp_rp', 'sector']}
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    def prepare(self, frame):
        """Attach pdcsap-branch references, Gaia anchor and the pdcsap HGB factor. No query flux is read."""
        import full_physical_rebuild as rebuild
        f = frame.copy()
        if 'reference_fold' not in f:
            f['reference_fold'] = [rebuild.hash_bucket(str(t), 5, 'reference') for t in f.tic_id]
        if 'physical_group_id' not in f:
            f['physical_group_id'] = f.tic_id.astype(str)
        q = self.sap.scientific_predict('pdcsap', f).rename(columns={'correction': 'hgb_pdcsap'})
        q['has_gaia_anchor'] = np.isfinite(q.gaiarp) & np.isfinite(q.bp_rp)
        q.loc[~q.has_gaia_anchor, 'gaia_log_anchor'] = np.nan
        return q.sort_values(['tic_id', 'sector']).reset_index(drop=True)

    def _standardised(self, q):
        p = self.model.prep
        return ((q[self.features].replace([np.inf, -np.inf], np.nan).fillna(p['medians']) - p['means']) / p['stds']).to_numpy(np.float32)

    def predict(self, q):
        return self.model.predict(q)

    def sample(self, q, seed=SEED):
        """Log correction samples, shape (n_samples, rows)."""
        import torch
        p = self.model.prep
        x = self._standardised(q)
        torch.manual_seed(seed)
        self.model.flow.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                dist = self.model.flow(torch.from_numpy(x[start:start + 2048]))
                out.append(dist.sample((self.n_samples,)).numpy().reshape(self.n_samples, -1))
        y = np.concatenate(out, axis=1)
        return np.log(q.pred_knn.to_numpy(float))[None, :] + y * p['y_std'] + p['y_median']

    def apply(self, frame, seed=SEED):
        """Per-sector factors with sigma and quality flags. `frame`: completed physical feature rows."""
        q = self.prepare(frame)
        factor = self.predict(q)
        logs = self.sample(q, seed)
        pct = np.percentile(logs, [2.5, 16, 50, 84, 97.5], axis=0)
        t = self.thresholds
        p = q[['tic_id', 'sector', 'cam', 'ccd', 'col', 'row', 'tmag', 'bp_rp', 'crowdsap', 'flfrcsap', 'ref_count',
               'knn50_radius', 'reference_fold', 'has_gaia_anchor', 'gaia_log_anchor', 'pred_knn', 'hgb_pdcsap']].copy()
        p['cnf102'] = factor
        p['cnf102_log_sd'] = logs.std(axis=0)
        p['cnf102_sigma_pct'] = 100 * p.cnf102_log_sd
        p['cnf102_p16'] = np.exp(pct[1])
        p['cnf102_p84'] = np.exp(pct[3])
        p['cnf102_p2_5'] = np.exp(pct[0])
        p['cnf102_p97_5'] = np.exp(pct[4])
        p['cnf102_sample_median'] = np.exp(pct[2])
        p['sigma_keep80'] = p.cnf102_sigma_pct <= t['keep80_sigma_pct']
        p['sigma_keep50'] = p.cnf102_sigma_pct <= t['keep50_sigma_pct']
        p['reference_supported'] = p.ref_count.fillna(0).gt(0)
        for name, (lo, hi) in self.ranges.items():
            p[name + '_outside_training_range'] = ~p[name].between(lo, hi)
        missing = ~np.isfinite(q[self.features].to_numpy(float))
        p['imputed_feature_count'] = missing.sum(axis=1)
        p['imputed_features'] = [';'.join(np.asarray(self.features)[row]) for row in missing]
        p['science_supported'] = (p.reference_supported & p.has_gaia_anchor & ~p.tmag_outside_training_range &
                                  ~p.bp_rp_outside_training_range & ~p.sector_outside_training_range)
        return p, q

    def describe(self):
        return dict(model_dir=str(self.root.relative_to(REPO)), reference_root=str(SAPREF.relative_to(REPO)),
                    point_estimate='analytic conditional median of the flow', n_samples=self.n_samples,
                    sigma='std of log correction over the samples; calibrated scale on quiet confirmation stars, heavy tails',
                    thresholds=self.thresholds, features=len(self.features), artifact_sha256=self.hashes,
                    support_ranges=self.ranges, no_training=True)
