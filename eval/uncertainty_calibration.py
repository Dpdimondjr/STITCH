"""Empirical coverage audit for an NSF correction distribution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import zuko
from scipy.stats import kstest
from sklearn.model_selection import train_test_split
from checkpoint_value_benchmark import historical_features, make_context

DATA = 'training_data_v4_fixed.parquet'
MODEL = 'eval/ablation_legacy_mean.pt'
ck = torch.load(MODEL, map_location='cpu', weights_only=False)
d = pd.read_parquet(DATA)
d['perstar_gaia_offset'] = d.groupby('tic_id').flux_offset.transform('mean')
d = d.dropna(subset=['col', 'row', 'flux_offset_loo'])
d = d[d.flux_offset_loo.between(.85, .15 + 1, inclusive='neither') &
      d.n_sectors_total.ge(4) & d.tmag.le(13)].copy()
for c in ck['continuous_cols']:
    if c not in ['sector_ccd_mean_loo', 'spatial_knn_mean_loo']:
        d[c] = d[c].fillna(d[c].median())
d, _ = historical_features(d, 'flux_offset_loo')
stars = d.groupby('tic_id').cam.agg(lambda x: x.mode().iloc[0]).reset_index()
train, tmp = train_test_split(stars.tic_id, test_size=.2,
                              stratify=stars.cam, random_state=42)
val, test = train_test_split(tmp, test_size=.5,
    stratify=stars[stars.tic_id.isin(tmp)].cam.to_numpy(), random_state=42)
t = d[d.tic_id.isin(test)].reset_index(drop=True)
cfg = ck['flow_config']
flow = zuko.flows.NSF(features=cfg['features'], context=cfg['context'],
    transforms=cfg['transforms'], hidden_features=cfg['hidden_features'], bins=cfg['bins'])
flow.load_state_dict(ck['model_state']); flow.eval(); torch.manual_seed(20260916)
levels = [.5, .68, .8, .9, .95]
hits = {x: [] for x in levels}; widths = {x: [] for x in levels}; pits = []
for start in range(0, len(t), 512):
    f = t.iloc[start:start+512]
    context = torch.tensor(make_context(f, ck)); y = f.flux_offset_loo.to_numpy()
    with torch.no_grad():
        samples = (flow(context).sample((1000,)).squeeze(-1).numpy() *
                   ck['y_std'] + ck['y_mean'])
    pits.extend((samples < y[None, :]).mean(axis=0))
    for level in levels:
        lo, hi = np.quantile(samples, [(1-level)/2, (1+level)/2], axis=0)
        hits[level].extend((y >= lo) & (y <= hi)); widths[level].extend(hi-lo)
pits = np.asarray(pits); ks = kstest(pits, 'uniform')
result = {'model': MODEL, 'data': DATA, 'test_rows': len(t),
  'coverage': {str(k): {'nominal': k, 'empirical': float(np.mean(hits[k])),
    'median_width': float(np.median(widths[k]))} for k in levels},
  'pit': {'mean': float(pits.mean()), 'std': float(pits.std()),
    'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue),
    'edge_fraction': float(np.mean((pits == 0) | (pits == 1)))},
  'note': ('Observation-label predictive coverage on the historically reconstructed '
           'held-out-star split; not coverage of a latent true instrumental correction.')}
out = Path('eval/uncertainty_calibration_results'); out.mkdir(exist_ok=True)
(out/'results.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
