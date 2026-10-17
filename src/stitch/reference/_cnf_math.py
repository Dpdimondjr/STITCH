"""Preserved historical CNF spatial statistics."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

KEYS = ['sector','cam','ccd']


KS = [5, 10, 20, 50]


def spatial_features(reference, query):
    """KNN summaries from explicit reference rows, excluding matching TIC."""
    query = query.reset_index(drop=True)
    columns=[f'knn{k}_{stat}' for k in KS for stat in ['mean','median','mad','radius']]
    columns += ['reference_count']
    output=np.full((len(query),len(columns)),np.nan,dtype=np.float32)
    output[:,-1]=0
    groups = dict(tuple(reference.groupby(KEYS, sort=False)))
    for key, qg in query.groupby(KEYS, sort=False):
        ref = groups.get(key)
        if ref is None or len(ref) < 2:
            continue
        coords = ref[['col','row']].to_numpy(np.float32)
        ids = ref.tic_id.to_numpy()
        vals = ref.flux_offset_loo.to_numpy()
        qcoords = qg[['col','row']].to_numpy(np.float32)
        qids = qg.tic_id.to_numpy()
        # There is normally one record per TIC in a sector. Request enough
        # extras to guarantee removal if duplicated records exist.
        extra = int(ref.tic_id.value_counts().max())
        take = min(max(KS) + extra, len(ref))
        distances, indices = cKDTree(coords).query(qcoords, k=take)
        if take == 1:
            distances, indices = distances[:,None], indices[:,None]
        outpos=qg.index.to_numpy()
        selected=np.full((len(qg),max(KS)),-1,dtype=np.int64)
        selected_dist=np.full((len(qg),max(KS)),np.nan,dtype=np.float32)
        for row, (q_id, nn, dd) in enumerate(zip(qids, indices, distances)):
            keep = ids[nn] != q_id
            nn,dd=nn[keep][:max(KS)],dd[keep][:max(KS)]
            selected[row,:len(nn)]=nn; selected_dist[row,:len(dd)]=dd
        count=(selected>=0).sum(axis=1)
        output[outpos,-1]=count
        for ki,k in enumerate(KS):
            good=count>=k
            if not good.any(): continue
            x=vals[selected[good,:k]]
            med=np.median(x,axis=1)
            base=ki*4
            output[outpos[good],base]=x.mean(axis=1)
            output[outpos[good],base+1]=med
            output[outpos[good],base+2]=np.median(np.abs(x-med[:,None]),axis=1)
            output[outpos[good],base+3]=selected_dist[good,k-1]
    return pd.DataFrame(output,index=query.index,columns=columns)
