"""Matched legacy versus cross-fitted reference construction for current HGB.

Run prepare, then four train cases, then summarize. All new output stays in
--out. Original TIC roles, eligibility, features, and HGB capacity are retained.
Clean queries use nine of ten TRAINING TIC folds, with the query fold excluded.
Validation/test labels never define clean reference features. The legacy cases
deliberately reproduce the old leakage for a controlled diagnostic comparison.
"""
import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor

KEYS = ['sector', 'cam', 'ccd']
KS = [5, 10, 20, 50]
SPATIAL = [f'knn{k}_{s}' for k in KS for s in ['mean', 'median', 'mad', 'radius']] + ['ref_count']
SECTOR = ['sector_mean_loo', 'sector_std_loo', 'sector_frac_extreme']
STRIPE = ['col_destripe_loo', 'row_destripe_loo']
DEV_SOURCE = ['crowdsap', 'jitter_rms', 'pdcvar', 'focal_plane_temp']
DEV = ['dev_crowdsap_loo', 'dev_jitter_rms_loo', 'dev_pdcvar_loo', 'dev_fptemp_loo']
PARAMS = dict(loss='absolute_error', learning_rate=.06, max_iter=250,
              max_leaf_nodes=1023, min_samples_leaf=40, l2_regularization=.1,
              early_stopping=False, verbose=0)
CASES = ['legacy_base', 'legacy_deviation', 'clean_base', 'clean_deviation']
ROUNDS = 5


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stripe_profiles(ref, axis):
    profiles = {}
    for key, g in ref.groupby(['cam', 'ccd']):
        bins = np.linspace(g[axis].min(), g[axis].max(), 129)
        bucket = np.clip(np.digitize(g[axis], bins) - 1, 0, 127)
        p = g.assign(_bin=bucket).groupby('_bin').flux_offset_loo.agg(['median', 'count']).reindex(range(128))
        count = p['count'].fillna(0)
        shrunk = (p['median'].fillna(1) * count + 30) / (count + 30)
        profiles[key] = (bins, shrunk.rolling(3, center=True, min_periods=1).median().to_numpy())
    return profiles


def reference_features(ref, query, legacy=False, sector_stats=None):
    """No query flux is read. Clean ref/query TIC sets must be disjoint."""
    if not legacy and set(ref.tic_id) & set(query.tic_id):
        raise ValueError('Clean reference bank contains a query TIC')
    q = query.reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=q.index, columns=SECTOR + SPATIAL + STRIPE + ['pred_knn'])
    out['pred_knn'] = 1.
    if sector_stats is None:
        sector_stats = ref.groupby(KEYS).flux_offset_loo.agg(
            sector_mean_loo='mean', sector_std_loo='std',
            sector_frac_extreme=lambda x: (np.abs(x-1) > .02).mean()).reset_index()
    merged = q[KEYS].merge(sector_stats, on=KEYS, how='left', validate='many_to_one')
    for col, fill in zip(SECTOR, [1., 0., 0.]):
        out[col] = merged[col].fillna(fill).to_numpy()
    groups = dict(tuple(ref.groupby(KEYS)))
    for key, g in q.groupby(KEYS):
        r = groups.get(key)
        if r is None or not len(r):
            continue
        labels = r.flux_offset_loo.to_numpy()
        rids = r.tic_id.to_numpy()
        out.loc[g.index, 'ref_count'] = len(r)
        # Match historical float32 spatial search/statistics and mean absolute
        # deviation (the historical 'mad' is not a median absolute deviation).
        k = min(max(KS), len(r))
        distance, ix = cKDTree(r[['col', 'row']].to_numpy(np.float32)).query(
            g[['col', 'row']].to_numpy(np.float32), k=k)
        ix = np.asarray(ix).reshape(len(g), -1)
        distance = np.asarray(distance).reshape(len(g), -1)
        values = labels.astype(np.float32)
        for kk in KS:
            v = values[ix[:, :min(kk, k)]]
            med = np.median(v, axis=1)
            for suffix, x in [('mean', v.mean(axis=1)), ('median', med),
                              ('mad', np.abs(v-med[:, None]).mean(axis=1)),
                              ('radius', distance[:, :min(kk, k)].max(axis=1))]:
                out.loc[g.index, f'knn{kk}_{suffix}'] = x.astype(np.float32)
        if len(r) < 2:
            continue
        take = min(100 + int(r.tic_id.value_counts().max()), len(r))
        _, ix = cKDTree(r[['col', 'row']].to_numpy(np.float64)).query(
            g[['col', 'row']].to_numpy(np.float64), k=take)
        ix = np.asarray(ix).reshape(len(g), -1)
        if legacy:
            for i, tic in enumerate(g.tic_id):
                neighbors = ix[i][rids[ix[i]] != tic][:100]
                if len(neighbors):
                    out.loc[g.index[i], 'pred_knn'] = np.median(labels[neighbors])
        else:
            out.loc[g.index, 'pred_knn'] = np.median(labels[ix[:, :100]], axis=1)
    for col in SPATIAL:
        out[col] = out[col].fillna(0 if col == 'ref_count' else out.sector_mean_loo)
    for axis, col in zip(['col', 'row'], STRIPE):
        values = np.ones(len(q), dtype=np.float32)
        for (cam, ccd), (bins, profile) in stripe_profiles(ref, axis).items():
            mask = ((q.cam == cam) & (q.ccd == ccd)).to_numpy()
            idx = np.clip(np.digitize(q.loc[mask, axis], bins)-1, 0, len(profile)-1)
            values[mask] = profile[idx]
        out[col] = values
    return out


def add_context(frame):
    out = frame.copy()
    for axis in ['col', 'row']:
        out['temp_x_' + axis] = out.focal_plane_temp * (out[axis] / 2200.)
    out['temp_x_crowdsap'] = out.focal_plane_temp * out.crowdsap
    out['pdctotp_x_crowdsap'] = out.pdc_totp * out.crowdsap
    out['n_sectors_obs'] = out.groupby('tic_id').sector.transform('nunique').astype(float)
    arrays = {c: np.zeros(len(out), dtype=np.float32) for c in DEV}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _, g in out.groupby('tic_id'):
            n = len(g)
            if n < 3:
                continue
            for src, dst in zip(DEV_SOURCE, DEV):
                v = g[src].to_numpy(float)
                others = np.broadcast_to(v, (n, n)).copy()
                np.fill_diagonal(others, np.nan)
                med = np.nanmedian(others, axis=1)
                width = 1.4826*np.nanmedian(np.abs(others-med[:, None]), axis=1)
                sd = np.nanstd(others, axis=1)
                width = np.where(width < 1e-9, np.maximum(sd, 1e-9), width)
                z = (v-med)/width
                z[np.isfinite(others).sum(axis=1) < 2] = 0
                arrays[dst][g.index] = np.nan_to_num(z, nan=0).astype(np.float32)
    for c, values in arrays.items():
        out[c] = values
    return out


def assign_folds(frame, folds, seed):
    stars = frame.groupby('tic_id').agg(role=('role', 'first'),
                                      cam=('cam', lambda s: s.mode().iloc[0]))
    stars['reference_fold'] = -1
    rng = np.random.default_rng(seed)
    for _, g in stars.groupby(['role', 'cam'], sort=True):
        ids = rng.permutation(g.index.to_numpy())
        stars.loc[ids, 'reference_fold'] = np.arange(len(ids)) % folds
    return stars.reset_index()


def crossfit_reference_features(frame, folds):
    clean = pd.DataFrame(index=frame.index, columns=SECTOR+SPATIAL+STRIPE+['pred_knn'], dtype=float)
    audit = []
    for fold in range(folds):
        ref = frame[(frame.role=='train') & (frame.reference_fold != fold)]
        query = frame[frame.reference_fold==fold]
        block = reference_features(ref, query)
        clean.loc[query.index, block.columns] = block.to_numpy()
        audit.append({'fold':fold, 'reference_stars':int(ref.tic_id.nunique()),
                      'reference_rows':len(ref), 'query_rows':len(query),
                      'subject_reference_overlap':len(set(ref.tic_id)&set(query.tic_id)),
                      'nontraining_reference_rows':int((ref.role!='train').sum())})
        log(f'Clean reference fold {fold+1}/{folds}: {len(ref):,} reference rows')
    return clean, audit


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'prepared.json').exists():
        raise ValueError('Prepared output already exists; use a new directory to rebuild')
    log('Loading original cohort and TIC roles')
    source = pd.read_parquet(args.data)
    source['source_row'] = np.arange(len(source))
    d = source.dropna(subset=['col', 'row', 'flux_offset', 'flux_offset_loo', 'sector_median'])
    d = d[(d.flux_offset>.85)&(d.flux_offset<1.15)&(d.flux_offset_loo>.85)&
          (d.flux_offset_loo<1.15)&(d.tmag<=13)&(d.n_sectors_total>=4)].copy().reset_index(drop=True)
    assert not d.duplicated(['tic_id']+KEYS).any()
    d['role'] = ''
    ids = {}
    for name, file_role in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        ids[name] = np.load(Path(args.split_dir)/f'{file_role}_tic_ids.npy')
        d.loc[d.tic_id.isin(ids[name]), 'role'] = name
    assert d.role.ne('').all()
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not set(ids[a]) & set(ids[b])
    manifest = assign_folds(d, args.folds, args.fold_seed)
    d['reference_fold'] = d.tic_id.map(manifest.set_index('tic_id').reference_fold)
    manifest.to_csv(out/'tic_roles_and_folds.csv', index=False)
    log('Building metadata-only within-star features')
    d = add_context(d)
    feature_meta = json.loads(Path('models/hgb_bprp_only_ensemble/meta.json').read_text())
    features = feature_meta['features']
    assert len(features) == 54
    clean, audit = crossfit_reference_features(d, args.folds)
    # Save training references so future feature builds use the same bank.
    ref_cols = ['tic_id']+KEYS+['col', 'row', 'flux_offset_loo', 'reference_fold']
    d.loc[d.role=='train', ref_cols].to_parquet(out/'training_reference_rows.parquet', index=False)
    clean_frame = d.drop(columns=clean.columns, errors='ignore').join(clean)
    clean_frame.to_parquet(out/'clean_features.parquet', index=False)
    log('Building historical reference policy for matched controls')
    legacy_ref = d[d.role != 'test']
    legacy_stats = pd.read_parquet(Path(args.split_dir)/'sec_stats.parquet')
    legacy = reference_features(legacy_ref, d, legacy=True, sector_stats=legacy_stats)
    d.drop(columns=legacy.columns, errors='ignore').join(legacy).to_parquet(out/'legacy_features.parquet', index=False)
    config = {'data':str(Path(args.data).resolve()), 'dataset_sha256':sha256(args.data),
              'script_sha256':sha256(__file__), 'split_dir':str(Path(args.split_dir).resolve()),
              'folds':args.folds, 'fold_seed':args.fold_seed, 'features_base':features,
              'features_deviation':features+DEV, 'rounds':ROUNDS, 'hgb_params':PARAMS,
              'roles':{r:{'stars':int(g.tic_id.nunique()), 'rows':len(g),
                         'metric_stars':int((g.groupby('tic_id').sector.nunique()>=2).sum())}
                       for r,g in d.groupby('role')},
              'reference_audit':audit, 'legacy_reference_stars':int(legacy_ref.tic_id.nunique()),
              'limitations':['Existing repeatedly examined test set; diagnostic, not pristine.',
                             'Legacy outcome cuts and pre-filter sector count retained for comparability.',
                             'Clean reference banks are smaller than historical train+validation banks.',
                             'One member per case initially; not a five-member ensemble result.']}
    write_json(out/'prepared.json', config)
    log('Preparation complete')


def per_star(frame, prediction):
    z = frame[['tic_id', 'sector', 'sector_median']].copy()
    z['flux'] = z.sector_median.to_numpy()/prediction
    g = z.groupby('tic_id')
    cv = 100*g.flux.std(ddof=0)/g.flux.mean()
    return cv[g.sector.nunique()>=2]


def group_median(values, ids):
    return pd.Series(values).groupby(pd.Series(ids)).transform('median').to_numpy()


def train(args):
    root = Path(args.out)
    prep = json.loads((root/'prepared.json').read_text())
    policy, variant = args.case.split('_')
    run = root/args.case/f'seed_{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    feats = prep['features_'+('base' if variant=='base' else 'deviation')]
    config = {'case':args.case, 'seed':args.seed, 'features':feats, 'params':PARAMS,
              'rounds':ROUNDS, 'prepared_sha256':sha256(root/'prepared.json'),
              'imputation':'per_split_historical' if policy=='legacy' else 'training_only',
              'script_sha256':sha256(__file__)}
    if (run/'config.json').exists():
        if json.loads((run/'config.json').read_text()) != config:
            raise ValueError('Run configuration changed; use a new output directory')
    else:
        write_json(run/'config.json', config)
    if (run/'results.json').exists():
        log(f'{args.case} already complete'); return
    df = pd.read_parquet(root/f'{policy}_features.parquet')
    frames = {r:g.reset_index(drop=True) for r,g in df.groupby('role')}
    del df
    train_medians = frames['train'][feats].median().fillna(0)
    x = {}; imputers = {}
    for role, frame in frames.items():
        med = frame[feats].median() if policy=='legacy' else train_medians
        # HGB accepts remaining all-missing columns in the legacy policy.
        imputers[role] = med
        x[role] = frame[feats].fillna(med).to_numpy(np.float32)
    joblib.dump(imputers, run/'imputation.pkl')
    tr = frames['train']; ids = tr.tic_id.to_numpy()
    residual = np.log(tr.sector_median.to_numpy())-np.log(tr.pred_knn.to_numpy())
    delta = {r:np.zeros(len(frames[r])) for r in ['validation', 'test']}
    history = []; start = time.monotonic()
    for rnd in range(ROUNDS):
        residual -= group_median(residual, ids)
        path = run/f'round_{rnd}.pkl'
        before = time.monotonic()
        if path.exists():
            log(f'{args.case} loading round {rnd+1}/{ROUNDS}')
            model = joblib.load(path)
        else:
            log(f'{args.case} training round {rnd+1}/{ROUNDS}')
            model = HistGradientBoostingRegressor(random_state=args.seed+rnd, **PARAMS)
            model.fit(x['train'], residual)
            temp = path.with_suffix('.pkl.tmp')
            joblib.dump(model, temp); temp.replace(path)
        for role in delta:
            delta[role] += model.predict(x[role])
        residual -= model.predict(x['train'])
        history.append({'round':rnd+1, 'seconds':time.monotonic()-before,
                        'centered_train_mae':float(np.median(np.abs(residual-group_median(residual, ids))))})
        write_json(run/'progress.json', {'status':'training', 'completed_rounds':rnd+1,
                                        'total_rounds':ROUNDS, 'history':history})
        log(f'{args.case} round {rnd+1} complete in {history[-1]["seconds"]:.1f}s')
    curve = []
    va = frames['validation']
    for beta in np.linspace(.5, 2, 31):
        pred = va.pred_knn.to_numpy()*np.exp(beta*delta['validation'])
        curve.append({'beta':float(beta), 'validation_cv_pct':float(per_star(va, pred).median())})
    best = min(curve, key=lambda r:r['validation_cv_pct'])
    metrics = {}; tables = {}
    for role in ['validation', 'test']:
        frame = frames[role]
        pred = frame.pred_knn.to_numpy()*np.exp(best['beta']*delta[role])
        table = pd.DataFrame({'raw':per_star(frame, np.ones(len(frame))),
                              'knn':per_star(frame, frame.pred_knn.to_numpy()),
                              'hgb':per_star(frame, pred)})
        tables[role] = table
        table.to_csv(run/f'{role}_per_star_cv.csv')
        saved = frame[['source_row', 'tic_id']+KEYS+['sector_median', 'reference_fold', 'pred_knn']].copy()
        saved['log_delta'] = delta[role]; saved['correction'] = pred
        saved.to_parquet(run/f'{role}_predictions.parquet', index=False)
        metrics[role] = {'stars':len(table), 'raw_cv_pct':float(table.raw.median()),
                         'knn_cv_pct':float(table.knn.median()), 'hgb_cv_pct':float(table.hgb.median()),
                         'hgb_beta1_cv_pct':float(per_star(frame,frame.pred_knn.to_numpy()*np.exp(delta[role])).median()),
                         'hgb_p90_cv_pct':float(table.hgb.quantile(.9)),
                         'stars_worse_than_raw_pct':float((table.hgb>table.raw).mean()*100)}
    # Reload actual checkpoint files and compare their predictions on a fixed
    # set of validation rows; no test-dependent acceptance or model selection.
    ix = np.linspace(0, len(va)-1, min(512, len(va)), dtype=int)
    reloaded = sum(joblib.load(run/f'round_{r}.pkl').predict(x['validation'][ix]) for r in range(ROUNDS))
    np.testing.assert_allclose(reloaded, delta['validation'][ix], rtol=1e-12, atol=1e-12)
    result = {'case':args.case, 'seed':args.seed, 'ensemble_members':1, 'best_beta':best['beta'],
              'metrics':metrics, 'beta_curve':curve, 'training_history':history,
              'elapsed_seconds':time.monotonic()-start, 'reload_prediction_check':True,
              'limitations':prep['limitations']}
    write_json(run/'results.json', result)
    write_json(run/'progress.json', {'status':'complete', 'completed_rounds':ROUNDS, 'metrics':metrics})
    log(json.dumps({'case':args.case, 'beta':best['beta'], 'metrics':metrics}))


def paired_bootstrap(base, candidate, seed, draws=2000):
    pair = pd.concat([base.rename('base'), candidate.rename('candidate')], axis=1).dropna()
    rng = np.random.default_rng(seed)
    differences = []; relatives = []
    b = pair.base.to_numpy(); c = pair.candidate.to_numpy()
    for start in range(0, draws, 100):
        ix = rng.integers(0, len(pair), (min(100, draws-start), len(pair)))
        bm = np.median(b[ix], axis=1); cm = np.median(c[ix], axis=1)
        differences.extend(bm-cm); relatives.extend(100*(bm-cm)/bm)
    gain = float(np.median(b)-np.median(c))
    return {'stars':len(pair), 'gain_pp':gain,
            'gain_pp_ci95':np.quantile(differences,[.025,.975]).tolist(),
            'relative_gain_pct':100*gain/float(np.median(b)),
            'relative_gain_pct_ci95':np.quantile(relatives,[.025,.975]).tolist(),
            'stars_improved_pct':float(np.mean(c<b)*100),
            'draws':draws, 'limitation':'Conditional on one fitted model and reference realization.'}


def summarize(args):
    root = Path(args.out); results = {}; tables = {}
    for case in CASES:
        run = root/case/f'seed_{args.seed}'
        if not (run/'results.json').exists():
            continue
        results[case] = json.loads((run/'results.json').read_text())
        tables[case] = pd.read_csv(run/'test_per_star_cv.csv', index_col='tic_id')
    comparisons = {}
    for base, candidate in [('legacy_base','clean_base'), ('legacy_deviation','clean_deviation'),
                            ('clean_base','clean_deviation'), ('legacy_base','legacy_deviation')]:
        if base in tables and candidate in tables:
            assert tables[base].index.equals(tables[candidate].index)
            np.testing.assert_allclose(tables[base].raw, tables[candidate].raw, rtol=1e-12)
            comparisons[candidate+'_vs_'+base] = paired_bootstrap(tables[base].hgb,tables[candidate].hgb,args.seed)
    out = {'seed':args.seed, 'completed_cases':list(results),
           'metrics':{c:r['metrics'] for c,r in results.items()},
           'beta':{c:r['best_beta'] for c,r in results.items()}, 'paired_comparisons':comparisons,
           'limitations':json.loads((root/'prepared.json').read_text())['limitations']}
    write_json(root/'comparison.json', out)
    log(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train', 'summarize'])
    p.add_argument('--out', default='eval/hgb_reference_repair_20261010')
    p.add_argument('--data', default='training_data_v3_tic_pointing.parquet')
    p.add_argument('--split-dir', default='models/hgb_v2')
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--fold-seed', type=int, default=20261010)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--case', choices=CASES)
    args = p.parse_args()
    if args.stage == 'train' and args.case is None:
        p.error('--case is required for train')
    if args.folds < 2:
        p.error('--folds must be at least two')
    {'prepare':prepare, 'train':train, 'summarize':summarize}[args.stage](args)


if __name__ == '__main__':
    main()
