"""Select physical response on training calibration TICs, then freeze/evaluate."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from analyze_calibration_banks import paired_interval
from hgb_reference_repair import per_star,write_json,sha256,log

ROOT=Path('eval/geometry_training_transfer_20261012')


def load_frame(role):
    sample=pd.read_parquet(ROOT/'sample.parquet')
    ids=sample.loc[sample.study_role.eq(role),'source_row']
    geometry=pd.read_parquet(ROOT/'geometry_features.parquet');geometry=geometry[geometry.source_row.isin(ids)]
    meta=pd.read_parquet(ROOT/'baseline/query_features.parquet',columns=['source_row','bp_rp','gaiarp','tmag','cam','ccd','col','row','crowdsap'])
    pred=pd.read_parquet(ROOT/'baseline/predictions.parquet');pred=pred[pred.source_row.isin(ids)]
    f=pred.merge(geometry,on=['source_row','tic_id','sector'],how='left',validate='one_to_one').merge(meta,on='source_row',validate='one_to_one')
    f['geometry_available']=f.geometry_available.fillna(False)
    f['valid_head']=f.geometry_available&f.has_gaia_anchor
    f['log_correction']=f.log_correction_r5
    return f


def center(f,value):
    z=pd.Series(np.asarray(value),index=f.index)
    return (z-z.groupby(f.tic_id).transform('mean')).fillna(0).to_numpy()


def design(f,family):
    values=[f.capture_term,f.crowding_term,f.log_correction]
    names=['capture','crowding','baseline_strength']
    if family in ['scene_strap','geometry_context']:
        values.append(.05*f.overlap_0);names.append('strap_prf_overlap')
    if family=='geometry_context':
        color=f.bp_rp.fillna(1)-1
        values.extend([f.capture_term*color,f.crowding_term*color,.05*f.overlap_0*color,.05*f.area_fraction,.02*np.log(f.aperture_pixels.clip(lower=1))])
        names.extend(['capture_color','crowding_color','strap_color','strap_aperture_area','log_aperture_area'])
        values.extend([.05*f.neighbor_strap_ratio,.05*f.neighbor_aperture_ratio*f.overlap_0])
        names.extend(['neighbor_light_on_straps','neighbor_light_times_target_overlap'])
        for cam in [1,2,3]:
            values.append(.05*f.overlap_0*f.actual_cam.eq(cam));names.append(f'strap_camera_{cam}')
    return np.column_stack([np.clip(center(f,v),-.04,.04) for v in values]),names


def fit_model(f,x,y,mask,alpha):
    ids=f.loc[mask,'tic_id'];weight=1/ids.groupby(ids).transform('size').to_numpy();weight/=weight.mean()
    model=HuberRegressor(epsilon=1.35,alpha=alpha,max_iter=3000,tol=1e-9)
    return model.fit(x[mask],y[mask],sample_weight=weight)


def correction(f,x,model):
    delta=np.clip(model.predict(x)-model.intercept_,-.03,.03)
    delta[~f.valid_head.to_numpy()]=0
    return delta


def calibrate():
    if (ROOT/'head_selection.json').exists():return
    freeze=json.loads((ROOT/'freeze.json').read_text())
    f=load_frame('calibration');assert f.study_role.eq('calibration').all()
    eligible=(f.systematic_score.gt(.95)&f.valid_head).to_numpy()
    y=(np.log(f.sector_median)-f.log_correction-f.gaia_log_anchor).to_numpy()
    all_records=[];output=f.copy();models={}
    for family in freeze['candidates']:
        x,names=design(f,family);family_records=[]
        for alpha in freeze['head_alphas']:
            delta=np.zeros(len(f))
            for fold in range(3):
                fit=eligible&f.head_fold.ne(fold).to_numpy();query=f.head_fold.eq(fold).to_numpy()
                assert not set(f.loc[fit,'tic_id'])&set(f.loc[query,'tic_id'])
                model=fit_model(f,x,y,fit,alpha)
                delta[query]=correction(f,x,model)[query]
            output[f'oof_{family}_a{alpha}']=delta
            for view in ['all','qualified']:
                mask=np.ones(len(f),bool) if view=='all' else f.systematic_score.gt(.95).to_numpy()
                baseline=per_star(f[mask],np.exp(f.loc[mask,'log_correction']))
                candidate=per_star(f[mask],np.exp(f.loc[mask,'log_correction'].to_numpy()+delta[mask]))
                row=dict(family=family,alpha=alpha,view=view,observations=int(mask.sum()),cv_pct=float(candidate.median()),
                    baseline_cv_pct=float(baseline.median()),**paired_interval(baseline,candidate))
                all_records.append(row);family_records.append(row)
        chosen=min([r for r in family_records if r['view']=='qualified'],key=lambda r:r['cv_pct'])
        model=fit_model(f,x,y,eligible,chosen['alpha']);joblib.dump(model,ROOT/f'head_{family}.pkl')
        models[family]=dict(alpha=chosen['alpha'],features=names,coefficients=model.coef_.tolist(),intercept=float(model.intercept_),
            oof_qualified_cv_pct=chosen['cv_pct'],fit_stars=int(f.loc[eligible,'tic_id'].nunique()),fit_rows=int(eligible.sum()),
            model_sha256=sha256(ROOT/f'head_{family}.pkl'))
        log(f'Geometry head {family}: calibration OOF CV {chosen["cv_pct"]:.5f}%')
    selected=min([r for r in all_records if r['view']=='qualified'],key=lambda r:r['cv_pct'])
    output.to_parquet(ROOT/'calibration_oof.parquet',index=False)
    write_json(ROOT/'head_selection.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),script_sha256=sha256(__file__),selected=selected,models=models,
        candidates=all_records,evaluation_outcomes_opened=False,baseline_sha256=sha256(ROOT/'baseline/predictions.parquet'),
        geometry_sha256=sha256(ROOT/'geometry_features.parquet'),sample_sha256=sha256(ROOT/'sample.parquet')))


def evaluate():
    selection=json.loads((ROOT/'head_selection.json').read_text())
    assert sha256(ROOT/'baseline/predictions.parquet')==selection['baseline_sha256']
    assert sha256(ROOT/'geometry_features.parquet')==selection['geometry_sha256']
    f=load_frame('evaluation');records=[];tables=[];coverage=[]
    for view in ['all','qualified']:
        g=f if view=='all' else f[f.systematic_score.gt(.95)]
        coverage.append(dict(view=view,observations=len(g),stars_any=int(g.tic_id.nunique()),metric_stars=int(g.groupby('tic_id').sector.nunique().ge(2).sum()),
            geometry_missing=int((~g.geometry_available).sum()),anchor_missing=int((~g.has_gaia_anchor).sum()),
            detector_mismatches=int((g.geometry_available&((g.cam!=g.actual_cam)|(g.ccd!=g.actual_ccd))).sum())))
    for family,info in selection['models'].items():
        assert sha256(ROOT/f'head_{family}.pkl')==info['model_sha256']
        x,_=design(f,family);model=joblib.load(ROOT/f'head_{family}.pkl');delta=correction(f,x,model)
        f['delta_'+family]=delta
        for view in ['all','qualified']:
            mask=np.ones(len(f),bool) if view=='all' else f.systematic_score.gt(.95).to_numpy()
            baseline=per_star(f[mask],np.exp(f.loc[mask,'log_correction']))
            candidate=per_star(f[mask],np.exp(f.loc[mask,'log_correction'].to_numpy()+delta[mask]))
            records.append(dict(family=family,view=view,observations=int(mask.sum()),baseline_cv_pct=float(baseline.median()),candidate_cv_pct=float(candidate.median()),
                selected=family==selection['selected']['family'],**paired_interval(baseline,candidate)))
            t=pd.DataFrame(dict(baseline=baseline,candidate=candidate)).reset_index();t['family']=family;t['view']=view;tables.append(t)
    f.to_parquet(ROOT/'evaluation_predictions.parquet',index=False)
    pd.concat(tables).to_parquet(ROOT/'evaluation_per_star.parquet',index=False)
    write_json(ROOT/'results.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected=selection['selected'],metrics=records,coverage=coverage,
        scope='First geometry-head evaluation on these 400 TICs; historically examined validation stars selected uniformly from those without cached LC geometry. No calibration labels from validation.'))
    log('Frozen training-star geometry transfer evaluated')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','evaluate']);a=p.parse_args()
    {'calibrate':calibrate,'evaluate':evaluate}[a.stage]()
