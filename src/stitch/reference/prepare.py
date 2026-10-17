"""Training-only reference labels and nested Gaia RP passband transfer."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from ..features.context import hash_bucket,inner_baseline,color_design,REFCOLS
from ..io import write_json,sha256
from .safe import hgb_features,cnf_features


def prepare(metadata,labels,splits,out,family='hgb'):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new preparation directory')
    f=pd.read_parquet(metadata).drop(columns=['role','physical_group_id','systematic_score','sector_median'],errors='ignore')
    s=pd.read_parquet(splits)
    required=['source_row','tic_id','sector','role','physical_group_id','systematic_score']
    if set(required)-set(s):raise ValueError(f'Split manifest requires {required}')
    if not s.role.isin(['train','validation','test']).all():raise ValueError('Invalid split role')
    if s.physical_group_id.isna().any():raise ValueError('Every row requires a physical source group')
    f=f.merge(s[required],on=['source_row','tic_id','sector'],validate='one_to_one')
    y=pd.read_parquet(labels)
    f=f.merge(y[['source_row','tic_id','sector','sector_median']],on=['source_row','tic_id','sector'],validate='one_to_one').sort_values('source_row').reset_index(drop=True)
    if len(f)!=len(s) or len(f)!=len(y):raise ValueError('Unmatched metadata/label/split rows')
    if f.duplicated(['tic_id','sector']).any():raise ValueError('Duplicate TIC/sector')
    if not (np.isfinite(f.sector_median)&f.sector_median.gt(0)).all():raise ValueError('Positive finite measurements required')
    held=set(f.loc[f.role.ne('train'),'physical_group_id'])
    eligible=f.role.eq('train')&f.systematic_score.gt(.95)&~f.physical_group_id.isin(held)
    if 'training_reference_eligible' in f:eligible &= f.training_reference_eligible.fillna(False)
    counts=f[eligible].groupby('tic_id').sector.nunique()
    f['fit_row']=eligible&f.tic_id.map(counts).ge(2)
    train=f[f.fit_row]
    if train.empty:raise ValueError('No eligible qualified training sources')
    if train.groupby('physical_group_id').tic_id.nunique().max()>1:raise ValueError('Resolve aliased training identities before fitting')
    f['reference_fold']=f.physical_group_id.map(lambda x:hash_bucket(x,5,'reference'))
    f['evaluation_group']=np.where(f.role.eq('validation'),f.physical_group_id.map(lambda x:'calibration' if hash_bucket(x,2,'validation')==0 else 'confirmation'),f.role)
    r=f[f.fit_row].copy();group=r.groupby('tic_id').sector_median
    r['flux_offset_loo']=r.sector_median/((group.transform('sum')-r.sector_median)/(group.transform('size')-1))
    if family=='cnf':r['flux_offset_loo']=r.flux_offset_loo.clip(.5,1.5)
    out.mkdir(parents=True)
    audit=[];blocks=[]
    f['has_gaia_anchor']=np.isfinite(f.gaiarp)&np.isfinite(f.bp_rp);f['gaia_log_anchor']=0.
    for fold in range(5):
        bank=r[r.reference_fold.ne(fold)];query=f[f.reference_fold.eq(fold)]
        z=(hgb_features if family=='hgb' else cnf_features)(bank,query);z.index=query.index;blocks.append(z)
        inner=inner_baseline(bank)
        stars=bank.groupby('tic_id').agg(rp=('gaiarp','first'),color=('bp_rp','first'),crowd=('crowdsap','median'))
        response=np.log(bank.sector_median/inner).groupby(bank.tic_id).median()
        stars['response']=response+.4*np.log(10)*(stars.rp-10)
        keep=np.isfinite(stars.rp)&np.isfinite(stars.color)&stars.crowd.gt(.95)&stars.rp.between(7,13.2)&stars.color.between(0,3.3)
        if keep.sum()<5:raise ValueError(f'Insufficient color-transfer stars in fold {fold}; enlarge training data')
        color=HuberRegressor(epsilon=1.35,alpha=.0001,max_iter=1000,tol=1e-8)
        color.fit(color_design(stars.loc[keep,'color']),stars.loc[keep,'response'])
        joblib.dump(color,out/f'color_fold_{fold}.pkl')
        good=f.reference_fold.eq(fold)&f.has_gaia_anchor
        if good.any():f.loc[good,'gaia_log_anchor']=-.4*np.log(10)*(f.loc[good,'gaiarp']-10)+color.predict(color_design(f.loc[good,'bp_rp']))
        audit.append(dict(fold=fold,training_reference_rows=len(bank),query_rows=len(query),color_stars=int(keep.sum()),physical_group_overlap=0))
    refs=pd.concat(blocks).sort_index();f=f.drop(columns=refs.columns,errors='ignore').join(refs)
    if family=='cnf':
        f['unsupported_reference']=f.knn20_median.isna();f['knn20_median']=f.knn20_median.fillna(1.)
    for c in f.select_dtypes('float').columns:
        if c not in ['gaia_log_anchor','pred_knn','sector_median']:f[c]=f[c].astype(np.float32)
    bankcols=['source_row','tic_id','sector','cam','ccd','col','row','physical_group_id','reference_fold','systematic_score','flux_offset_loo']
    r[bankcols].to_parquet(out/'reference_bank.parquet',index=False)
    f[required+['reference_fold','evaluation_group','fit_row']].to_parquet(out/'cohort_manifest.parquet',index=False)
    f[['source_row','sector_median']].to_parquet(out/'labels.parquet',index=False)
    f.drop(columns='sector_median').to_parquet(out/'features.parquet',index=False)
    write_json(out/'preparation.json',dict(family=family,target='gaia_rp',training_rows=int(f.fit_row.sum()),missing_anchor_rows=int((~f.has_gaia_anchor).sum()),missing_training_anchors=int((~f.loc[f.fit_row,'has_gaia_anchor']).sum()),excluded_training_role_conflicts=int((f.role.eq('train')&f.physical_group_id.isin(held)).sum()),folds=audit,input_sha256={str(p):sha256(p) for p in [metadata,labels,splits]},relative_fallback='Only missing-anchor training rows use a centered relative residual; query flux never supplies inference anchors.'))
    return f
