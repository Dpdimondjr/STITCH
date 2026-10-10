"""Test residual learning over median-KNN, then multi-scale spatial features.

Reference labels are restricted to training stars. Every training query excludes
all rows belonging to its own TIC. Hyperparameters/shrinkage are selected using
validation stars; test metrics are reported afterward.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

SEED = 20260915
KEYS = ['sector','cam','ccd']
KS = [5, 10, 20, 50]


def split_stars(df):
    stars = (df.groupby('tic_id').cam.agg(lambda x: x.mode().iloc[0])
             .rename('cam').reset_index())
    train, temp = train_test_split(stars, test_size=.2, stratify=stars.cam,
                                   random_state=42)
    val, test = train_test_split(temp, test_size=.5, stratify=temp.cam,
                                 random_state=42)
    return tuple(set(x.tic_id) for x in (train,val,test))


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


def scatter_table(frame, predictions):
    tables = []
    for name, offset in predictions.items():
        x = frame[['tic_id','sector_median']].copy()
        x['corrected'] = x.sector_median / np.clip(offset, .85, 1.15)
        stats = x.groupby('tic_id').corrected.agg(['mean','std','count'])
        tables.append(stats['std'].div(stats['mean']).rename(name))
    per_star = pd.concat(tables,axis=1).dropna()
    per_star = per_star[per_star.index.to_series().map(frame.groupby('tic_id').size()).ge(5)]
    return per_star


def tune_alpha(frame, raw_prediction):
    best = None
    for alpha in np.linspace(0,1.25,26):
        pred = 1 + alpha*(raw_prediction-1)
        score = scatter_table(frame, {'x':pred}).x.median()
        if best is None or score < best[1]: best=(float(alpha),float(score))
    return best


def main():
    t0=time.time(); out=Path('eval/residual_multiscale_results'); out.mkdir(exist_ok=True)
    df=pd.read_parquet('training_data_v3.parquet')
    df=df.dropna(subset=['col','row','flux_offset_loo'])
    df=df[df.flux_offset_loo.between(.85,1.15,inclusive='neither') &
          df.n_sectors_total.ge(4) & df.tmag.le(13)].copy().reset_index(drop=True)
    train_ids,val_ids,test_ids=split_stars(df)
    split={'train':df.tic_id.isin(train_ids),'val':df.tic_id.isin(val_ids),
           'test':df.tic_id.isin(test_ids)}
    reference=df[split['train']].copy()
    features={}
    for name,mask in split.items():
        print('spatial features',name,mask.sum(),flush=True)
        features[name]=spatial_features(reference,df[mask])

    base_cols=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
        'tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2',
        'gaiarp','focal_plane_temp']
    single=base_cols+['knn20_median','knn20_mad','knn20_radius','reference_count']
    multi=base_cols+[f'knn{k}_{s}' for k in KS for s in ['mean','median','mad','radius']]+['reference_count']
    models={}; predictions={}; metrics={}
    train_frame=df[split['train']].reset_index(drop=True)
    val_frame=df[split['val']].reset_index(drop=True)
    test_frame=df[split['test']].reset_index(drop=True)
    for label,cols in [('residual_k20',single),('residual_multiscale',multi)]:
        Xtr=pd.concat([train_frame[base_cols],features['train'].drop(columns=[],errors='ignore')],axis=1)[cols]
        Xv=pd.concat([val_frame[base_cols],features['val']],axis=1)[cols]
        Xt=pd.concat([test_frame[base_cols],features['test']],axis=1)[cols]
        base_tr=features['train'].knn20_median.to_numpy()
        y=train_frame.flux_offset_loo.to_numpy()-base_tr
        valid=np.isfinite(base_tr)
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.06,
            max_iter=250,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=.1,
            random_state=SEED,early_stopping=False)
        print('fit',label,valid.sum(),len(cols),flush=True)
        model.fit(Xtr.loc[valid],y[valid])
        pv=features['val'].knn20_median.to_numpy()+model.predict(Xv)
        pt=features['test'].knn20_median.to_numpy()+model.predict(Xt)
        alpha,val_score=tune_alpha(val_frame,pv)
        predictions[label]=1+alpha*(pt-1)
        metrics[label]={'alpha':alpha,'validation_scatter':val_score}
        models[label]=model

    median=features['test'].knn20_median.to_numpy()
    raw=np.ones(len(test_frame))
    per=scatter_table(test_frame,{'raw':raw,'median_knn':median,**predictions})
    summary={name:{'median_scatter_pct':float(per[name].median()*100),
        'reduction_pct':float((1-per[name].median()/per.raw.median())*100),
        'stars_worse_pct':float((per[name]>per.raw).mean()*100)} for name in per}
    payload={'rows':{k:int(v.sum()) for k,v in split.items()},'stars':{
        k:int(df[v].tic_id.nunique()) for k,v in split.items()},'validation':metrics,
        'test':summary,'elapsed_seconds':time.time()-t0,
        'reference_policy':'training stars only; own TIC excluded'}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    per.to_csv(out/'per_star_scatter.csv')
    print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
