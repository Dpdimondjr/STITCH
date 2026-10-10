"""Spatial features from an explicit reference catalog, excluding subject TICs.

Build after splitting stars. Pass training stars as the reference catalog for
training, validation, and test queries. Missing neighborhoods remain missing.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def robust_features(reference, query, k=20, target='flux_offset_loo'):
    if k < 1:
        raise ValueError('k must be positive')
    names = ['neighbor_mean', 'neighbor_median', 'neighbor_mad',
             'neighbor_q25', 'neighbor_q75', 'neighbor_radius',
             'neighbor_count', 'reference_count', 'ccd_mean']
    result = pd.DataFrame(np.nan, index=np.arange(len(query)), columns=names)
    result[['neighbor_count', 'reference_count']] = 0
    # Preserve original query row positions independently of index labels.
    q = query.reset_index(drop=True)
    r = reference.reset_index(drop=True)
    keys = ['sector', 'cam', 'ccd']
    good = r[['col', 'row', target]].notna().all(axis=1)
    good &= np.isfinite(r[['col', 'row', target]]).all(axis=1)
    groups = dict(tuple(r[good].groupby(keys)))
    for key, subjects in q.groupby(keys):
        bank = groups.get(key)
        if bank is None:
            continue
        positions = bank[['col', 'row']].to_numpy()
        values = bank[target].to_numpy()
        ids = bank.tic_id.to_numpy()
        tree = cKDTree(positions)
        # Query enough extra candidates to exclude every record of the subject.
        extra = int(bank.tic_id.value_counts().max())
        for index, subject in subjects.iterrows():
            point = subject[['col', 'row']].to_numpy(dtype=float)
            if not np.isfinite(point).all():
                continue
            distances, nn = tree.query(point, k=min(k + extra, len(bank)))
            distances, nn = np.atleast_1d(distances), np.atleast_1d(nn)
            keep = ids[nn] != subject.tic_id
            nn, distances = nn[keep][:k], distances[keep][:k]
            eligible = ids != subject.tic_id
            result.loc[index, 'reference_count'] = eligible.sum()
            result.loc[index, 'neighbor_count'] = len(nn)
            if not len(nn):
                continue
            neighbors = values[nn]
            median = np.median(neighbors)
            result.loc[index, names[:6]] = [neighbors.mean(), median,
                np.median(np.abs(neighbors - median)),
                np.quantile(neighbors, .25), np.quantile(neighbors, .75),
                distances[-1]]
            result.loc[index, 'ccd_mean'] = values[eligible].mean()
    result.index = query.index
    return result
