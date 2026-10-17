"""Five-round training with a Gaia RP anchor retained in every residual."""
import json,shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from ..features.context import FEATURES,PARAMS,SEED,group_median,per_star
from ..io import write_json,sha256
from ..bundles import seal_bundle


def residual(frame,total,target='gaia_rp',prior='pred_knn'):
    value=np.log(frame.sector_median.to_numpy()/frame[prior].to_numpy())-total
    if target=='relative':return value-group_median(value,frame.tic_id.to_numpy())
    if target!='gaia_rp':raise ValueError('Unknown target')
    value-=frame.gaia_log_anchor.to_numpy()
    missing=~frame.has_gaia_anchor.to_numpy()
    if missing.any():value[missing]-=group_median(value[missing],frame.tic_id.to_numpy()[missing])
    return value


def train(prepared,out,identity_catalog,config=None):
    prepared=Path(prepared);out=Path(out)
    if out.exists():raise FileExistsError('Use a new model directory')
    f=pd.read_parquet(prepared/'features.parquet').merge(pd.read_parquet(prepared/'labels.parquet'),on='source_row',validate='one_to_one')
    if json.loads((prepared/'preparation.json').read_text())['family']!='hgb':raise ValueError('Expected HGB preparation')
    tr=f[f.fit_row].copy();q=f[f.evaluation_group.eq('calibration')].copy()
    if q.empty:raise ValueError('A designated calibration split is required')
    if not tr.role.eq('train').all():raise ValueError('Training rows include held-out labels')
    params=dict(PARAMS)
    if config:
        params.update(json.loads(Path(config).read_text()))
    impute=tr[FEATURES].replace([np.inf,-np.inf],np.nan).median().fillna(0.)
    x=tr[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(impute).to_numpy(np.float32)
    xq=q[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(impute).to_numpy(np.float32)
    out.mkdir(parents=True);core=out/'physical';core.mkdir()
    joblib.dump(impute,core/'imputation.pkl');total=np.zeros(len(tr));prediction=np.zeros(len(q))
    for round_index in range(5):
        model=HistGradientBoostingRegressor(random_state=SEED+round_index,**params)
        model.fit(x,residual(tr,total));total+=model.predict(x);prediction+=model.predict(xq)
        joblib.dump(model,core/f'round_{round_index}.pkl')
        print(f'HGB round {round_index+1}/5 saved',flush=True)
    score=[]
    factors=q.pred_knn.to_numpy()*np.exp(prediction)
    for view,mask in [('all',np.ones(len(q),bool)),('qualified',q.systematic_score.gt(.95).to_numpy())]:
        cv=per_star(q[mask],factors[mask]);score.append(dict(view=view,rows=int(mask.sum()),metric_stars=len(cv),median_cv_pct=float(cv.median()),missing_anchors=int((~q.loc[mask,'has_gaia_anchor']).sum())))
    for name in ['reference_bank.parquet','cohort_manifest.parquet','preparation.json',*[f'color_fold_{i}.pkl' for i in range(5)]]:shutil.copy2(prepared/name,out/name)
    shutil.copy2(identity_catalog,out/'identity_catalog.parquet')
    thermal=['sector','cam','ccd','focal_plane_temp','focal_plane_temp_std'];f[thermal].drop_duplicates().to_parquet(out/'thermal.parquet',index=False)
    write_json(out/'support.json',dict(ranges={name:[float(tr[name].min()),float(tr[name].max())] for name in ['tmag','bp_rp','sector']}))
    write_json(out/'selected_model.json',dict(selected='physical',target='gaia_rp',rounds=5,beta=1,core_directory='physical',features=FEATURES,model_sha256={p.name:sha256(p) for p in core.glob('*.pkl')},external_anchor_retained=True,default_model_promoted=False,training_rows=len(tr),params=params))
    write_json(out/'training.json',dict(calibration_scores=score,missing_training_anchors=int((~tr.has_gaia_anchor).sum()),target='gaia_rp',retain_anchor_every_round=True,per_star_centering=False,prepared_sha256=sha256(prepared/'features.parquet')))
    seal_bundle(out,dict(kind='hgb',target='gaia_rp',status='new research fit; not automatically promoted'))
