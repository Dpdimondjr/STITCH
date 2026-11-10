"""Full-population, physical-source-safe STITCH rebuild; Gaia RP is the target.

Reference features and nested passband fits are rebuilt from individually quiet
training sectors. Historical splits are preserved. Selection uses only the
designated validation-calibration half, before benchmark outcomes are opened.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from hgb_reference_repair import (SECTOR,SPATIAL,STRIPE,PARAMS,DEV,DEV_SOURCE,
    reference_features,per_star,group_median,sha256,write_json,log)
from calibration_bank_study import BASE,training_residual
from gaia_anchor_experiment import color_design
from fit_geometry_training_transfer import design,correction
from analyze_calibration_banks import paired_interval

ROOT=Path('eval/full_rebuild_20261012')
COLLECTION=Path('eval/full_geometry_collection_20261012')
SOURCE=Path('eval/hgb_reference_repair_20261010/clean_features.parquet')
PHYSICAL=COLLECTION/'full_physical_features.parquet'
LINEAGE=COLLECTION/'full_lineage.parquet'
IDENTITY=COLLECTION/'target_identity.parquet'
HEAD=Path('eval/geometry_training_transfer_20261012/identity_control/geometry_head.pkl')
SEED=2026101201
REFCOLS=SECTOR+SPATIAL+STRIPE+['pred_knn']
GEOMETRY=['capture_term','crowding_term','prf_capture','catalog_scene_crowding',
    'nominal_capture','nominal_crowding','overlap_0','neighbor_aperture_ratio',
    'neighbor_strap_ratio','area_fraction','aperture_pixels','sources',
    'omitted_expected_aperture_flux_relative_target','target_header_position_difference',
    'prf_grid_clamp_pixels']
CADENCE=['quality_zero_fraction','default_quality_fraction','n_cadences',
    'pos_corr1_median_default','pos_corr2_median_default','pos_corr1_rms_default',
    'pos_corr2_rms_default','mom_centr1_rms_default','mom_centr2_rms_default',
    'background_half_shift']
HEADER_MAP={'header_crowdsap':'crowdsap','header_flfrcsap':'flfrcsap','header_pdcvar':'pdcvar',
    'header_pr_wght2':'pr_wght2','header_pdc_noi':'pdc_noi','header_pdc_corp':'pdc_corp',
    'header_pdc_totp':'pdc_totp','header_cdpp1_0':'cdpp1_0','header_teff':'teff'}
CADENCE_MAP={'median_sap_bkg_default':'median_sap_bkg','p90_sap_bkg_default':'p90_sap_bkg',
    'bkg_rms_default':'bkg_rms','jitter_rms_default':'jitter_rms','scatter_flag_frac':'scatter_flag_frac'}
FLAGS=['physical_group_id','training_reference_role_conflict','systematic_score',
    'exact_product_identity','geometry_available','cadence_available','headers_available',
    'suspect_catalog_alias','training_reference_eligible','gaia_catalog_anchor_available']
GEOM_CENTERED=[f'geometry_centered_{i}' for i in range(14)]
PHYSICAL_DERIVED=['capture_log','crowding_log','log_background_rp','log_background_rms_rp',
    'log_background_p90_rp','background_half_shift_rp','log_aperture_pixels',
    'log_neighbor_aperture_ratio','log_neighbor_strap_ratio','log_omitted_catalog_light',
    'dev_capture_term','dev_crowding_term','dev_overlap_0','dev_area_fraction',
    'dev_log_aperture_pixels','dev_log_background_rp','dev_log_background_rms_rp',
    'dev_background_half_shift_rp','dev_pos_corr1_rms_default','dev_pos_corr2_rms_default',
    'dev_mom_centr1_rms_default','dev_mom_centr2_rms_default']
EXTRA=GEOMETRY+CADENCE+['flfrcsap']+PHYSICAL_DERIVED+GEOM_CENTERED
CASES=['baseline','physical','relative_control']


def read(path):return json.loads(Path(path).read_text())


def atomic_parquet(f,path):
    assert shutil.disk_usage(ROOT).free>2**30,'One GiB disk reserve reached'
    path=Path(path);tmp=path.with_suffix('.part.parquet');f.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)


def freeze():
    ROOT.mkdir(exist_ok=True)
    inputs=[SOURCE,PHYSICAL,LINEAGE,IDENTITY,HEAD,Path('research_defaults.json'),
        COLLECTION/'manifest.parquet',COLLECTION/'verification.json',COLLECTION/'completion_audit.json']
    expected={str(p):sha256(p) for p in inputs}
    if (ROOT/'freeze.json').exists():
        old=read(ROOT/'freeze.json');assert old['input_sha256']==expected,'Frozen inputs changed'
        return
    assert read(COLLECTION/'completion_audit.json')['status']=='passed'
    snap=ROOT/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path('eval/hgb_reference_repair.py'),Path('eval/calibration_bank_study.py'),
              Path('eval/fit_geometry_training_transfer.py'),Path('eval/gaia_anchor_experiment.py'),
              Path('eval/analyze_calibration_banks.py'),Path('research_defaults.json'),Path('docs/RESEARCH_PLAN.md'),Path('README.md')]:
        shutil.copy2(p,snap/p.name)
    write_json(ROOT/'freeze.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        input_sha256=expected,script_sha256=sha256(__file__),seed=SEED,folds=5,rounds=5,
        params=PARAMS,target='gaia_rp',retain_anchor_every_round=True,center_gaia_target_within_star=False,
        training='All eligible individual TARS >0.95 training rows in TICs with at least two such sectors; exclude aliases and source groups represented in validation/test.',
        references='All reference labels recomputed on this selection; exclude the entire outer query physical group, all validation/test labels. Nested color preprocessing also excludes outer query folds.',
        coordinates='Use exact FITS camera/CCD and native-Gaia projected detector coordinates where available. Use historical raw metadata only for explicitly missing products. Thermal inputs on changed camera/CCD are missing.',
        features={'baseline':BASE,'physical':BASE+EXTRA,'relative_control':BASE},
        selection='Three fixed five-round Gaia candidates, beta=1: baseline, baseline + unchanged frozen geometry_context head, expanded physical HGB. Select minimum qualified CV in a hash-designated half of validation groups. Remaining validation and test outcomes opened only after selection freeze.',
        relative_control='Explicit matched centered-target diagnostic, never eligible for default Gaia selection.',
        missing_anchors='Zero placeholder anchor plus explicitly centered relative training target only for missing RP/color. Query predictions never read query flux for a fallback.',
        scope='Historically examined development populations, not pristine holdouts. One deterministic seed; no full-population noise-floor claim.'))
    log('Full rebuild protocol frozen')


def hash_bucket(value,n,salt):
    import hashlib
    return int.from_bytes(hashlib.blake2b(f'{SEED}:{salt}:{value}'.encode(),digest_size=8).digest(),'big')%n


def deviations(f):
    """Recompute metadata-only LOO deviations using all retained observations."""
    groups=f.groupby('tic_id',sort=False).indices
    for source,dest in zip(DEV_SOURCE,DEV):
        result=np.zeros(len(f),np.float32)
        source_values=f[source].to_numpy(float)
        for indices in groups.values():
            n=len(indices)
            if n<3:continue
            v=source_values[indices]
            if np.isfinite(v).sum()<3:continue
            other=np.broadcast_to(v,(n,n)).copy();np.fill_diagonal(other,np.nan)
            med=np.nanmedian(other,axis=1);width=1.4826*np.nanmedian(np.abs(other-med[:,None]),axis=1)
            sd=np.nanstd(other,axis=1);width=np.where(width<1e-9,np.maximum(sd,1e-9),width)
            z=(v-med)/width;z[np.isfinite(other).sum(axis=1)<2]=0
            result[indices]=np.nan_to_num(z,nan=0,posinf=0,neginf=0)
        f[dest]=result
        log(f'Rebuilt within-star predictor {dest}')


def base_frame():
    raw=list(dict.fromkeys([c for c in BASE if c not in REFCOLS+DEV and not c.startswith('temp_x_') and c not in ['pdctotp_x_crowdsap','n_sectors_obs']]+['source_row','tic_id','role','sector_median','median_sap_bkg','p90_sap_bkg','flfrcsap']))
    f=pd.read_parquet(SOURCE,columns=raw).sort_values('source_row').reset_index(drop=True)
    cols=list(dict.fromkeys(['source_row','tic_id','sector','role','actual_cam','actual_ccd','actual_col','actual_row']+FLAGS+GEOMETRY+CADENCE+list(HEADER_MAP)+list(CADENCE_MAP)))
    p=pd.read_parquet(PHYSICAL,columns=cols).sort_values('source_row').reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['source_row','tic_id','sector','role']],p[['source_row','tic_id','sector','role']],check_dtype=False)
    for c in cols:
        if c not in ['source_row','tic_id','sector','role'] and c not in CADENCE_MAP:f[c]=p[c]
    changes={}
    moved=p.geometry_available&((f.cam!=p.actual_cam)|(f.ccd!=p.actual_ccd))
    for old,new in [('cam','actual_cam'),('ccd','actual_ccd'),('col','actual_col'),('row','actual_row')]:
        f.loc[p.geometry_available,old]=p.loc[p.geometry_available,new]
    for src,dest in HEADER_MAP.items():
        use=p.headers_available;changes[dest]=int(use.sum());f.loc[use,dest]=p.loc[use,src]
    for src,dest in CADENCE_MAP.items():f.loc[p.cadence_available,dest]=p.loc[p.cadence_available,src]
    f.loc[moved,['focal_plane_temp','focal_plane_temp_std']]=np.nan
    # Preserve the historical definition: projected coordinate + median POS_CORR.
    # Archive POS_CORR is an ensemble pointing estimate, not a flux centroid.
    for axis,i in [('col',1),('row',2)]:
        use=p.geometry_available&p.cadence_available
        f.loc[use,'delta_sub_'+axis]=(f.loc[use,axis]+p.loc[use,f'pos_corr{i}_median_default'])%1.
    f['temp_x_col']=f.focal_plane_temp*f.col/2200.;f['temp_x_row']=f.focal_plane_temp*f.row/2200.
    f['temp_x_crowdsap']=f.focal_plane_temp*f.crowdsap;f['pdctotp_x_crowdsap']=f.pdc_totp*f.crowdsap
    f['n_sectors_obs']=f.groupby('tic_id').sector.transform('nunique').astype(float)
    f['reference_fold']=f.physical_group_id.map(lambda x:hash_bucket(x,5,'reference')).astype(np.int8)
    f['evaluation_group']=np.where(f.role.eq('validation'),f.physical_group_id.map(lambda x:'calibration' if hash_bucket(x,2,'validation')==0 else 'confirmation'),f.role)
    count=f[f.training_reference_eligible].groupby('tic_id').sector.nunique()
    f['fit_row']=f.training_reference_eligible&f.tic_id.map(count).ge(2)
    assert not set(f.loc[f.fit_row,'physical_group_id'])&set(f.loc[f.role.ne('train'),'physical_group_id'])
    assert f.loc[f.fit_row].groupby('physical_group_id').tic_id.nunique().max()==1
    deviations(f)
    rpflux=np.power(10.,-.4*(f.gaiarp.fillna(f.tmag)-10))
    for src,dest in [('median_sap_bkg','log_background_rp'),('bkg_rms','log_background_rms_rp'),('p90_sap_bkg','log_background_p90_rp')]:f[dest]=np.log1p(f[src].clip(lower=0)/rpflux)
    f['background_half_shift_rp']=f.background_half_shift/rpflux
    f['capture_log']=np.log(f.flfrcsap.clip(.05,1.5));f['crowding_log']=np.log(f.crowdsap.clip(.05,1.5))
    f['log_aperture_pixels']=np.log1p(f.aperture_pixels)
    f['log_neighbor_aperture_ratio']=np.log1p(f.neighbor_aperture_ratio.clip(lower=0))
    f['log_neighbor_strap_ratio']=np.log1p(f.neighbor_strap_ratio.clip(lower=0))
    f['log_omitted_catalog_light']=np.log1p(f.omitted_expected_aperture_flux_relative_target.clip(lower=0))
    for name in [c[4:] for c in PHYSICAL_DERIVED if c.startswith('dev_')]:f['dev_'+name]=f[name]-f.groupby('tic_id')[name].transform('median')
    # These are centered predictors, never centered Gaia learning targets.
    f['log_correction']=0.;xx,names=design(f,'geometry_context');assert xx.shape[1]==14
    for i,c in enumerate(GEOM_CENTERED):f[c]=xx[:,i]
    f=f.drop(columns='log_correction')
    return f,dict(header_replaced_rows=changes,detector_changed_rows=int(moved.sum()),geometry_design_names=names)


def inner_baseline(ref):
    prediction=pd.Series(1.,index=ref.index)
    for _,g in ref.groupby(['sector','cam','ccd']):
        if len(g)<2:continue
        ids=g.physical_group_id.to_numpy();labels=g.flux_offset_loo.to_numpy()
        k=min(100+int(g.physical_group_id.value_counts().max()),len(g))
        _,ix=cKDTree(g[['col','row']].to_numpy(float)).query(g[['col','row']].to_numpy(float),k=k)
        ix=np.asarray(ix).reshape(len(g),-1)
        ok=ids[ix]!=ids[:,None];ok &= np.cumsum(ok,axis=1)<=100
        pred=np.nanmedian(np.where(ok,labels[ix],np.nan),axis=1)
        prediction.loc[g.index]=np.where(np.isfinite(pred),pred,1.)
    return prediction


def prepare():
    if (ROOT/'prepared.json').exists():return
    f,metadata=base_frame()
    atomic_parquet(f[['source_row','tic_id','sector','role','physical_group_id','reference_fold','evaluation_group','systematic_score','fit_row']],ROOT/'cohort_manifest.parquet')
    tr=f[f.fit_row].copy();g=tr.groupby('tic_id').sector_median
    tr['flux_offset_loo']=tr.sector_median/((g.transform('sum')-tr.sector_median)/(g.transform('size')-1))
    assert np.isfinite(tr.flux_offset_loo).all()
    atomic_parquet(tr[['source_row','tic_id','sector','cam','ccd','col','row','physical_group_id','reference_fold','systematic_score','flux_offset_loo']],ROOT/'reference_bank.parquet')
    blocks=pd.DataFrame(np.nan,index=f.index,columns=REFCOLS);audit=[]
    f['gaia_log_anchor']=0.;f['has_gaia_anchor']=np.isfinite(f.gaiarp)&np.isfinite(f.bp_rp)
    for fold in range(5):
        start=time.monotonic();r=tr[tr.reference_fold.ne(fold)];q=f[f.reference_fold.eq(fold)]
        overlap=set(r.physical_group_id)&set(q.physical_group_id);assert not overlap
        z=reference_features(r,q);blocks.loc[q.index,z.columns]=z.to_numpy()
        inner=inner_baseline(r)
        stars=r.groupby('tic_id').agg(rp=('gaiarp','first'),color=('bp_rp','first'),crowd=('crowdsap','median'))
        response=np.log(r.sector_median/inner).groupby(r.tic_id).median()
        stars['response']=response+.4*np.log(10)*(stars.rp-10)
        keep=np.isfinite(stars.rp)&np.isfinite(stars.color)&stars.crowd.gt(.95)&stars.rp.between(7,13.2)&stars.color.between(0,3.3)
        model=HuberRegressor(epsilon=1.35,alpha=.0001,max_iter=1000,tol=1e-8)
        model.fit(color_design(stars.loc[keep,'color']),stars.loc[keep,'response'])
        joblib.dump(model,ROOT/f'color_fold_{fold}.pkl')
        good=f.reference_fold.eq(fold)&f.has_gaia_anchor
        f.loc[good,'gaia_log_anchor']=-.4*np.log(10)*(f.loc[good,'gaiarp']-10)+model.predict(color_design(f.loc[good,'bp_rp']))
        audit.append(dict(fold=fold,reference_rows=len(r),reference_tics=int(r.tic_id.nunique()),query_rows=len(q),
            query_physical_overlap=len(overlap),nontraining_reference_rows=int(r.role.ne('train').sum()),
            nonqualified_reference_rows=int((~r.systematic_score.gt(.95)).sum()),color_stars=int(keep.sum()),
            color_query_overlap=0,inner_references_exclude_subject=True,seconds=time.monotonic()-start))
        write_json(ROOT/'preparation_progress.json',dict(folds=audit));log(f'Full reference + nested color fold {fold+1}/5 completed')
    f=f.drop(columns=REFCOLS,errors='ignore').join(blocks)
    assert np.isfinite(f.pred_knn).all() and f.pred_knn.gt(0).all()
    # Measured flux is deliberately not stored in the feature table.
    f=f.drop(columns='sector_median')
    for c in f.select_dtypes('float').columns:
        if c not in ['gaia_log_anchor','pred_knn']:f[c]=f[c].astype(np.float32)
    atomic_parquet(f,ROOT/'features.parquet')
    missing={r:dict(rows=int((~g.has_gaia_anchor).sum()),tics=int(g.loc[~g.has_gaia_anchor,'tic_id'].nunique())) for r,g in f.groupby('role')}
    write_json(ROOT/'prepared.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        features_sha256=sha256(ROOT/'features.parquet'),manifest_sha256=sha256(ROOT/'cohort_manifest.parquet'),
        references_sha256=sha256(ROOT/'reference_bank.parquet'),input_sha256=read(ROOT/'freeze.json')['input_sha256'],
        rows=len(f),tics=int(f.tic_id.nunique()),training_rows=int(f.fit_row.sum()),training_tics=int(f.loc[f.fit_row,'tic_id'].nunique()),
        missing_anchors=missing,audit=audit,metadata=metadata,features=read(ROOT/'freeze.json')['features'],script_sha256=sha256(__file__)))
    log(f'Full features ready: {len(f):,} rows; {f.fit_row.sum():,} qualified training rows')


def load_frame():
    f=pd.read_parquet(ROOT/'features.parquet');y=pd.read_parquet(SOURCE,columns=['source_row','sector_median'])
    return f.merge(y,on='source_row',validate='one_to_one').sort_values('source_row').reset_index(drop=True)


def fit(case):
    root=ROOT/case;root.mkdir(exist_ok=True)
    if (root/'complete.json').exists():return
    protocol=read(ROOT/'freeze.json');features=protocol['features'][case]
    assert len(features)==len(set(features)) and not set(features)&{'sector_median','flux_offset','flux_offset_loo','systematic_score','gaia_log_anchor','source_row','tic_id','physical_group_id'}
    f=load_frame();tr=f[f.fit_row].reset_index(drop=True);q=f[f.role.ne('train')].reset_index(drop=True);del f
    median=tr[features].replace([np.inf,-np.inf],np.nan).median().fillna(0);joblib.dump(median,root/'imputation.pkl')
    x=tr[features].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)
    xq=q[features].replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)
    total=np.zeros(len(tr));pred=np.zeros(len(q));history=[]
    output=q[['source_row','tic_id','sector','role','evaluation_group','reference_fold','systematic_score','gaia_log_anchor','has_gaia_anchor','pred_knn']].copy()
    # Signal check: the Gaia anchor must change every nonfallback round target.
    probe=tr.head(100).copy();probe=probe[probe.has_gaia_anchor]
    r0=training_residual(probe,np.zeros(len(probe)));probe.gaia_log_anchor+=.01
    np.testing.assert_allclose(training_residual(probe,np.zeros(len(probe)))-r0,-.01,atol=1e-12)
    for rnd in range(5):
        start=time.monotonic();path=root/f'round_{rnd}.pkl'
        target=training_residual(tr,total)
        if case=='relative_control':target=target-group_median(target,tr.tic_id.to_numpy())
        if path.exists():model=joblib.load(path)
        else:
            log(f'{case}: training round {rnd+1}/5 ({len(features)} predictors)')
            model=HistGradientBoostingRegressor(random_state=SEED+rnd,**PARAMS);model.fit(x,target)
            tmp=path.with_suffix('.tmp');joblib.dump(model,tmp);tmp.replace(path)
        total+=model.predict(x);pred+=model.predict(xq)
        output[f'log_correction_r{rnd+1}']=np.log(q.pred_knn.to_numpy())+pred
        history.append(dict(round=rnd+1,seconds=time.monotonic()-start,training_target_median=float(np.median(target)),training_target_mad=float(np.median(np.abs(target-np.median(target))))))
        atomic_parquet(output,root/'predictions.parquet');write_json(root/'progress.json',dict(history=history))
        log(f'{case}: round {rnd+1}/5 saved ({history[-1]["seconds"]:.1f}s)')
    write_json(root/'complete.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),features=features,params=PARAMS,
        target='relative_control' if case=='relative_control' else 'gaia_rp',seed=SEED,rounds=5,history=history,
        prediction_sha256=sha256(root/'predictions.parquet'),features_sha256=sha256(ROOT/'features.parquet'),
        model_sha256={p.name:sha256(p) for p in root.glob('*.pkl')},script_sha256=sha256(__file__),no_evaluation_metrics_computed=True))


def candidates(f):
    values={}
    for case in CASES:
        p=pd.read_parquet(ROOT/case/'predictions.parquet').set_index('source_row')
        assert set(f.source_row)<=set(p.index)
        values[case]=p.log_correction_r5.reindex(f.source_row).to_numpy()
    g=f.copy();g['log_correction']=values['baseline'];g['valid_head']=g.geometry_available&g.has_gaia_anchor
    x,_=design(g,'geometry_context');head=joblib.load(HEAD)
    values['frozen_geometry']=values['baseline']+correction(g,x,head)
    return values


def select():
    if (ROOT/'selection.json').exists():return
    f=load_frame();f=f[f.evaluation_group.eq('calibration')].copy().reset_index(drop=True)
    assert f.role.eq('validation').all()
    values=candidates(f);records=[]
    for case in ['baseline','frozen_geometry','physical']:
        g=f[f.systematic_score.gt(.95)];cv=per_star(g,np.exp(values[case][f.systematic_score.gt(.95)]))
        records.append(dict(case=case,cv_pct=float(cv.median()),observations=len(g),metric_tics=len(cv)))
    selected=min(records,key=lambda r:r['cv_pct'])['case']
    write_json(ROOT/'selection.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected=selected,
        candidates=records,beta=1,rounds=5,selection_only='Validation calibration groups, qualified sectors',
        confirmation_or_test_outcomes_used=False,relative_control_not_eligible=True,
        cohort_sha256=sha256(ROOT/'cohort_manifest.parquet'),head_sha256=sha256(HEAD),
        model_sha256={case:read(ROOT/case/'complete.json')['model_sha256'] for case in CASES}))
    log(f'Full rebuild selection frozen: {selected}')


def evaluate():
    if (ROOT/'results.json').exists():return
    chosen=read(ROOT/'selection.json')['selected'];f=load_frame();f=f[f.role.ne('train')].copy().reset_index(drop=True)
    values=candidates(f);output=f[['source_row','tic_id','sector','role','evaluation_group','systematic_score','has_gaia_anchor','geometry_available','cadence_available','suspect_catalog_alias','physical_group_id','gaia_log_anchor','sector_median','pred_knn']].copy()
    for case,value in values.items():output['log_correction_'+case]=value
    atomic_parquet(output,ROOT/'benchmark_predictions.parquet');records=[];tables=[]
    groups={'validation_calibration':f.evaluation_group.eq('calibration'),
        'validation_confirmation':f.evaluation_group.eq('confirmation'),'validation_full':f.role.eq('validation'),'test_development':f.role.eq('test')}
    for group,groupmask in groups.items():
        for view in ['all','qualified']:
            mask=groupmask.to_numpy().copy()
            if view=='qualified':mask &= f.systematic_score.gt(.95).to_numpy()
            g=f[mask];base=per_star(g,np.exp(values['baseline'][mask]));raw=per_star(g,np.ones(len(g)))
            for case,value in values.items():
                cv=per_star(g,np.exp(value[mask]));gain=paired_interval(base,cv)
                anchored=g.has_gaia_anchor.to_numpy();e=np.log(g.sector_median.to_numpy())-value[mask]-g.gaia_log_anchor.to_numpy()
                agreement=pd.Series(e[anchored],index=g.loc[anchored,'tic_id']).groupby(level=0).median()
                records.append(dict(group=group,view=view,case=case,selected=case==chosen,rows=len(g),tics_any=int(g.tic_id.nunique()),metric_tics=len(cv),
                    missing_tars=int(g.systematic_score.isna().sum()),missing_anchor_rows=int((~g.has_gaia_anchor).sum()),missing_geometry_rows=int((~g.geometry_available).sum()),
                    cv_pct=float(cv.median()),raw_cv_pct=float(raw.median()),p90_cv_pct=float(cv.quantile(.9)),paired_gain=gain,
                    gaia_agreement_median_abs_log_pct=float(100*agreement.abs().median())))
                z=pd.DataFrame(dict(baseline=base,candidate=cv)).reset_index();z['group']=group;z['view']=view;z['case']=case;tables.append(z)
    atomic_parquet(pd.concat(tables,ignore_index=True),ROOT/'per_star.parquet')
    write_json(ROOT/'results.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected=chosen,metrics=records,
        predictions_sha256=sha256(ROOT/'benchmark_predictions.parquet'),selection_sha256=sha256(ROOT/'selection.json'),
        scope='Full historical development benchmarks. Validation calibration/full metrics include selection; confirmation/test are subsequent checks, not pristine holdouts. Single seed; no total-floor estimate.'))
    for r in records:
        if r['group']=='test_development' and r['case']==chosen:log(f'Selected full test development {r["view"]}: {r["cv_pct"]:.6f}% ({r["metric_tics"]} TICs)')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','prepare','fit','select','evaluate','all']);p.add_argument('--case',choices=CASES,default='baseline');a=p.parse_args()
    freeze()
    if a.stage in ['prepare','all']:prepare()
    if a.stage=='fit':fit(a.case)
    if a.stage=='all':
        for case in CASES:fit(case)
    if a.stage in ['select','all']:select()
    if a.stage in ['evaluate','all']:evaluate()


if __name__=='__main__':main()
