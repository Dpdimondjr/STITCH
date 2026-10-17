"""Preserved spatial and stripe arithmetic; use the public safety wrapper."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

KEYS = ['sector', 'cam', 'ccd']


KS = [5, 10, 20, 50]


SPATIAL = [f'knn{k}_{s}' for k in KS for s in ['mean', 'median', 'mad', 'radius']] + ['ref_count']


SECTOR = ['sector_mean_loo', 'sector_std_loo', 'sector_frac_extreme']


STRIPE = ['col_destripe_loo', 'row_destripe_loo']


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
