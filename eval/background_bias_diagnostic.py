"""Exploratory additive-bias scaling on an existing cache-native feature cohort.

No target flux enters the susceptibility denominator. Out-of-fold predictions
exclude each subject TIC. This tests association, not a causal background estimate
or an irreducible floor. Input parquet is built by spoc_cohort_experiment.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

CORE = ['sector', 'camera', 'ccd', 'tmag', 'crowdsap', 'flfrcsap',
        'mom_col', 'mom_row', 'ap_npix', 'cadence_days', 'data_rel']


def center(values, ids):
    s = pd.Series(np.asarray(values))
    return s.to_numpy() - s.groupby(np.asarray(ids)).transform('median').to_numpy()


def susceptibility(frame):
    """Fractional response to 1 electron/s/pixel residual background bias."""
    expected = 15000 * 10 ** ((10 - frame.tmag.to_numpy()) / 2.5)
    return (frame.ap_npix.to_numpy() * frame.crowdsap.to_numpy()
            / frame.flfrcsap.to_numpy() / expected)


def global_predict(train, query, columns=CORE, rounds=4):
    x = train[columns].replace([np.inf, -np.inf], np.nan)
    q = query[columns].replace([np.inf, -np.inf], np.nan)
    y = np.log(train.pdcsap_median.to_numpy())
    intercept = y - center(y, train.tic_id)
    for _ in range(rounds):
        model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=120,
            max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1,
            early_stopping=False, random_state=20260928)
        model.fit(x, y - intercept)
        residual = y - model.predict(x)
        intercept = residual - center(residual, train.tic_id)
    return model.predict(q)


def demean_groups(values, codes):
    a = np.asarray(values, float)
    return a - pd.DataFrame(a).groupby(codes).transform('mean').to_numpy()


def partial_slope(frame, draws=1000, seed=20260928):
    """Within sector/camera/CCD partial slope, with TIC-cluster bootstrap.

    Refit nuisance regression in each bootstrap via cluster cross-products.
    Group demeaning is fixed to the diagnostic sample; intervals are conditional
    on the out-of-fold models/reference population, not full pipeline uncertainty.
    """
    f = frame.copy()
    keys = ['sector', 'camera', 'ccd']
    counts = f.groupby(keys).tic_id.transform('nunique')
    f = f[counts >= 8].copy()
    if len(f) < 30:
        return {'status': 'insufficient within-group support', 'rows': len(f)}
    codes = pd.MultiIndex.from_frame(f[keys]).factorize()[0]
    # Scale susceptibility to 1e-3 for numerical conditioning. Do not log it:
    # the linear area/flux relation is the proposed additive signature.
    x = f.susceptibility.to_numpy() / .001
    controls = np.column_stack([f.tmag, f.tmag**2, f.crowdsap, f.flfrcsap,
                               f.mom_col/2048, f.mom_row/2048])
    matrix = np.column_stack([x, controls, f.signed_residual])
    star_codes = pd.factorize(f.tic_id)[0]
    # Absorb both star and observing-group fixed effects. Signed residuals have
    # a star normalization; center predictors consistently rather than regressing
    # those residuals against an uncentered, mostly star-specific susceptibility.
    for iteration in range(1000):
        previous = matrix.copy()
        matrix = demean_groups(demean_groups(matrix, star_codes), codes)
        if np.max(np.abs(matrix-previous)) < 1e-10:
            break
    else:
        raise RuntimeError('Two-way fixed-effect projection did not converge')
    y = matrix[:, -1]
    # Star-constant controls vanish under fixed effects. Remove those and use
    # an orthogonal nuisance basis so redundant magnitude terms cannot make
    # cluster-bootstrap normal equations numerically singular.
    controls = matrix[:, 1:-1]
    scales = controls.std(axis=0)
    controls = controls[:, scales > 1e-8] / scales[scales > 1e-8]
    u, singular, _ = np.linalg.svd(controls, full_matrices=False)
    rank = int(np.sum(singular > (singular[0]*1e-8 if len(singular) else 1e-8)))
    design = np.column_stack([matrix[:, 0], u[:, :rank]*np.sqrt(len(f))])
    if np.std(design[:, 0]) < 1e-10:
        return {'status': 'no within-group susceptibility variation', 'rows': len(f)}
    gram = design.T @ design
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0][0]
    ids, code = np.unique(f.tic_id.to_numpy(), return_inverse=True)
    xx = np.zeros((len(ids), design.shape[1], design.shape[1]))
    xy = np.zeros((len(ids), design.shape[1]))
    np.add.at(xx, code, design[:, :, None]*design[:, None, :])
    np.add.at(xy, code, design*y[:, None])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        take = rng.integers(0, len(ids), len(ids))
        samples.append(np.linalg.lstsq(xx[take].sum(0), xy[take].sum(0), rcond=None)[0][0])
    return {'status': 'ok', 'rows': len(f), 'stars': len(ids),
            'groups': int(len(np.unique(codes))),
            'fixed_effects': 'TIC and sector/camera/CCD',
            'slope_fraction_per_1e_minus3_susceptibility': float(coefficient),
            'tic_bootstrap_ci95': np.quantile(samples, [.025, .975]).tolist(),
            'design_condition_number': float(np.linalg.cond(gram)),
            'interpretation': 'association only; not an independently measured background bias'}


def coherence(frame, draws=500, seed=20260928):
    """Equal-cell mean cross-star residual product; within-TIC permutation null.

    Permuting residuals across each star's retained visits preserves its marginal
    residual distribution. It does not preserve time correlation or heteroscedasticity.
    """
    f = frame.copy()
    f['xb'] = np.floor(f.mom_col/512).astype(int)
    f['yb'] = np.floor(f.mom_row/512).astype(int)
    keys = ['sector', 'camera', 'ccd', 'xb', 'yb']
    codes = pd.MultiIndex.from_frame(f[keys]).factorize()[0]
    n = np.bincount(codes); keep = n >= 3
    if not keep.any():
        return {'status': 'insufficient spatial support'}
    def score(v):
        total = np.bincount(codes, weights=v)
        squares = np.bincount(codes, weights=v*v)
        return float(np.mean((total[keep]**2-squares[keep])/(n[keep]*(n[keep]-1))))
    y = f.signed_residual.to_numpy()
    indices = list(f.reset_index(drop=True).groupby('tic_id').indices.values())
    rng = np.random.default_rng(seed); null = []
    for _ in range(draws):
        v = y.copy()
        for ix in indices:
            v[ix] = rng.permutation(y[ix])
        null.append(score(v))
    observed = score(y)
    return {'status': 'ok', 'cells': int(keep.sum()),
            'mean_cross_star_product': observed,
            'permutation_null95': np.quantile(null, [.025, .975]).tolist(),
            'one_sided_permutation_p': float((1+np.sum(np.asarray(null)>=observed))/(draws+1)),
            'limitations': 'shared model error, time dependence and heteroscedasticity can mimic coherence'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', default='eval/spoc_cohort_results/cohort_features.parquet')
    p.add_argument('--quiet-catalog', default='tars_quiet_tics_v2.csv')
    p.add_argument('--out', default='eval/background_bias_results')
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(args.features)
    quiet = pd.read_csv(args.quiet_catalog, usecols=['tic_id'])
    original_stars = d.tic_id.nunique()
    d = d[d.error.fillna('').eq('') & d.tic_id.isin(quiet.tic_id)].copy()
    required = ['pdcsap_median', 'ap_npix', 'tmag', 'crowdsap', 'flfrcsap',
                'mom_col', 'mom_row', 'sector', 'camera', 'ccd']
    d = d.replace([np.inf,-np.inf],np.nan).dropna(subset=required)
    # Require genuine quality-zero coverage; extractor's low-coverage fallback
    # otherwise includes flagged cadences. No target-offset or residual cut.
    d = d[(d.pdcsap_median>0)&(d.ap_npix>0)&(d.flfrcsap>0)&(d.flfrcsap<=1)
          &(d.crowdsap>0)&(d.crowdsap<=1)&(d.tmag<=13)&(d.n_good>=50)
          &d.sector.between(1,100)&d.camera.isin([1,2,3,4])&d.ccd.isin([1,2,3,4])]
    if d.duplicated(['tic_id','sector']).any():
        raise ValueError('Duplicate TIC-sector products must be resolved by provenance')
    count = d.groupby('tic_id').sector.transform('nunique')
    d = d[count>=5].reset_index(drop=True)
    ids = np.sort(d.tic_id.unique())
    if len(ids)<30:
        raise ValueError('Fewer than 30 eligible quiet stars')
    d['prediction'] = np.nan; d['fold'] = -1
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=20260928).split(ids)):
        train = d.tic_id.isin(ids[tr]); test = d.tic_id.isin(ids[te])
        print(f'fold {fold+1}: {len(tr)} train / {len(te)} diagnostic stars',flush=True)
        d.loc[test,'prediction'] = global_predict(d[train],d[test])
        d.loc[test,'fold'] = fold
    log_corrected = np.log(d.pdcsap_median.to_numpy())-d.prediction.to_numpy()
    # Subject medians define the diagnostic response only, never model features.
    d['signed_residual'] = np.expm1(center(log_corrected,d.tic_id))
    d['susceptibility'] = susceptibility(d)
    d['raw_signed_residual'] = np.expm1(center(np.log(d.pdcsap_median),d.tic_id))
    d['corrected_flux'] = np.exp(log_corrected)
    def cv(s): return float(s.std(ddof=0)/s.mean())
    per = d.groupby('tic_id').agg(raw_cv=('pdcsap_median',cv),corrected_cv=('corrected_flux',cv))
    results = {'seed':20260928,'source_sha256':hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
        'quiet_catalog_sha256':hashlib.sha256(Path(args.quiet_catalog).read_bytes()).hexdigest(),
        'source_stars':int(original_stars),'eligible_stars':len(ids),'rows':len(d),
        'raw_median_cv_pct':float(100*per.raw_cv.median()),
        'corrected_median_cv_pct':float(100*per.corrected_cv.median()),
        'controlled_scaling':partial_slope(d), 'spatial_coherence':coherence(d),
        'fold_scaling':{str(k):partial_slope(g,draws=300) for k,g in d.groupby('fold')},
        'limitations':['Exploratory re-split of an already examined cache cohort; not a fresh final holdout.',
         'No independent residual-background measurement is available; SAP_BKG is not bgbias.',
         'Bootstrap conditions on trained models and fixed within-group demeaning.',
         'No causal mechanism, signal-preservation guarantee or irreducible noise floor established.']}
    d.drop(columns=['path','error'],errors='ignore').to_parquet(out/'diagnostic_rows.parquet',index=False)
    per.to_csv(out/'per_star_cv.csv')
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__': main()
