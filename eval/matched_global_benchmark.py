"""Matched reference-disjoint KNN/residual/global benchmark on a new random split.

Uses value_benchmark's eligibility and population CV convention, but omits the
subject-derived Gaia anchor and flux-noise predictors. This is a new diagnostic
partition of existing data, not a previously untouched scientific holdout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

COLS = ['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
        'tmag','crowdsap','gaiarp','knn_mean','knn_median','knn_mad','ccd_mean']


def references(ref, query):
    if set(ref.tic_id) & set(query.tic_id):
        raise ValueError('Reference and query TIC sets must be disjoint')
    q=query.copy().reset_index(drop=True)
    for c in ['knn_mean','knn_median','knn_mad','ccd_mean']: q[c]=np.nan
    groups=dict(tuple(ref.groupby(['sector','cam','ccd'])))
    for key,g in q.groupby(['sector','cam','ccd']):
        r=groups.get(key)
        if r is None or len(r)<5: continue
        _,ix=cKDTree(r[['col','row']]).query(g[['col','row']],k=min(20,len(r)))
        v=r.loo.to_numpy()[ix]; med=np.median(v,axis=1)
        q.loc[g.index,'knn_mean']=v.mean(axis=1)
        q.loc[g.index,'knn_median']=med
        q.loc[g.index,'knn_mad']=np.median(np.abs(v-med[:,None]),axis=1)
        q.loc[g.index,'ccd_mean']=r.loo.mean()
    q=q.dropna(subset=['knn_median'])
    return q[q.groupby('tic_id').sector.transform('nunique')>=5].reset_index(drop=True)


def per_star(frame, correction):
    z=frame[['tic_id','sector_median']].copy()
    z['flux']=z.sector_median.to_numpy()/correction
    g=z.groupby('tic_id').flux
    return g.std(ddof=0)/g.mean()


def model(seed):
    return HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.06,
        max_iter=250,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=.1,
        early_stopping=False,random_state=seed)


def fit_global(tr,seed):
    y=np.log(tr.sector_median.to_numpy())-np.log(tr.knn_median.to_numpy())
    def intercept(v):
        return pd.Series(v).groupby(tr.tic_id).transform('median').to_numpy()
    a=intercept(y); history=[]
    for i in range(5):
        m=model(seed+i);m.fit(tr[COLS],y-a)
        residual=y-m.predict(tr[COLS]);a=intercept(residual)
        history.append(float(np.median(np.abs(residual-a))))
        print('global round',i+1,'training centered MAE',history[-1],flush=True)
    return m,history


def bootstrap(per,seed,draws=2000):
    rng=np.random.default_rng(seed); out={}
    ix=rng.integers(0,len(per),size=(draws,len(per)))
    for name in per.columns:
        if name=='raw':continue
        gain=100*(1-np.median(per[name].to_numpy()[ix],axis=1)
                  /np.median(per.knn_tuned.to_numpy()[ix],axis=1))
        out[name]={'gain_over_tuned_knn_pct':float(100*(1-per[name].median()/per.knn_tuned.median())),
                   'ci95':np.quantile(gain,[.025,.975]).tolist()}
    return out


def global_vs_residual(per, seed, draws=2000):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(per), size=(draws, len(per)))
    gains = 100*(1-np.median(per.global_hgb.to_numpy()[ix],axis=1)
                 /np.median(per.residual_hgb.to_numpy()[ix],axis=1))
    return {'gain_pct':float(100*(1-per.global_hgb.median()/per.residual_hgb.median())),
            'ci95':np.quantile(gains,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='training_data_v3.parquet')
    p.add_argument('--out',default='eval/matched_global_results')
    p.add_argument('--seed',type=int,default=20260928)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(args.data).replace([np.inf,-np.inf],np.nan)
    good=(d.sector.between(1,100)&d.cam.isin([1,2,3,4])&d.ccd.isin([1,2,3,4])
        &d.tmag.le(13)&d.sector_median.gt(0)&d.col.between(0,2200)
        &d.row.between(0,2200)&d.gaiarp.notna())
    d=d[good].copy();d=d[d.groupby('tic_id').sector.transform('nunique')>=5].copy()
    if d.duplicated(['tic_id','sector']).any():raise ValueError('Duplicate TIC-sector rows')
    stars=d.groupby('tic_id').cam.agg(lambda s:s.mode().iloc[0])
    ref,rest=train_test_split(stars.index.to_numpy(),test_size=.6,stratify=stars,random_state=args.seed)
    tr,tmp=train_test_split(rest,test_size=1/3,stratify=stars.loc[rest],random_state=args.seed)
    va,te=train_test_split(tmp,test_size=.5,stratify=stars.loc[tmp],random_state=args.seed)
    rng=np.random.default_rng(args.seed)
    parts={'reference':ref,'train':rng.choice(tr,min(6000,len(tr)),replace=False),
           'validation':rng.choice(va,min(1500,len(va)),replace=False),
           'test':rng.choice(te,min(3000,len(te)),replace=False)}
    split=pd.concat([pd.DataFrame({'tic_id':ids,'partition':key}) for key,ids in parts.items()])
    assert not split.tic_id.duplicated().any()
    split.to_csv(out/'split.csv',index=False)
    g=d.groupby('tic_id').sector_median
    d['loo']=d.sector_median/((g.transform('sum')-d.sector_median)/(g.transform('count')-1))
    bank=d[d.tic_id.isin(ref)]
    frames={}
    for name in ['train','validation','test']:
        frames[name]=references(bank,d[d.tic_id.isin(parts[name])])
        print(name,frames[name].tic_id.nunique(),len(frames[name]),flush=True)
    tr,va,te=(frames[k] for k in ['train','validation','test'])
    def bounded(v):return np.clip(v,.5,2)
    choices=[]
    for col in ['knn_mean','knn_median']:
        for a in np.linspace(0,1,21):
            choices.append((per_star(va,bounded(1+a*(va[col].to_numpy()-1))).median(),col,float(a)))
    _,col,a=min(choices)
    pred={'raw':np.ones(len(te)),'knn_median':bounded(te.knn_median.to_numpy()),
          'knn_tuned':bounded(1+a*(te[col].to_numpy()-1))}
    selections={'knn':{'column':col,'strength':a}}
    residual=model(args.seed)
    residual.fit(tr[COLS],tr.loo.to_numpy()-tr.knn_median.to_numpy())
    rv=residual.predict(va[COLS]);rt=residual.predict(te[COLS])
    candidates=[]
    for b in np.linspace(-.25,1.5,36):
        candidates.append((per_star(va,bounded(va.knn_median.to_numpy()+b*rv)).median(),float(b)))
    _,b=min(candidates);pred['residual_hgb']=bounded(te.knn_median.to_numpy()+b*rt)
    selections['residual_strength']=b
    m,history=fit_global(tr,args.seed)
    rv=m.predict(va[COLS]);rt=m.predict(te[COLS])
    candidates=[]
    for b in np.linspace(-.25,1.5,36):
        candidates.append((per_star(va,bounded(np.exp(np.log(va.knn_median)+b*rv))).median(),float(b)))
    _,b=min(candidates);pred['global_hgb']=bounded(np.exp(np.log(te.knn_median)+b*rt))
    selections['global_strength']=b
    per=pd.DataFrame({name:per_star(te,v) for name,v in pred.items()});per.to_csv(out/'per_star_cv.csv')
    result={'seed':args.seed,'dataset_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'columns':COLS,'reference_stars':len(ref),'coverage':{k:{'stars':int(v.tic_id.nunique()),'rows':len(v)} for k,v in frames.items()},
        'selections':selections,'training_history':history,
        'metrics':{k:{'median_cv_pct':float(100*per[k].median()),
            'reduction_pct':float(100*(1-per[k].median()/per.raw.median())),
            'stars_worse_pct':float(100*(per[k]>per.raw).mean())} for k in per},
        'paired_comparisons':bootstrap(per,args.seed),
        'global_vs_residual_hgb':global_vs_residual(per,args.seed),
        'limitations':['New random partition of previously examined data, not a pristine final holdout.',
         'Matched feature set differs from the historical compact NSF; no new NSF was trained.',
         'One training seed; star-bootstrap intervals condition on reference catalog and models.',
         'No independent noise-floor or astrophysical-preservation claim.']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
