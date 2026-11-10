"""Matched SAP / header-adjusted SAP / PDCSAP HGB development pilot."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,hashlib,gzip,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
import requests,joblib
from astropy.io import fits
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from hgb_reference_repair import reference_features,SECTOR,SPATIAL,STRIPE,PARAMS
from full_physical_rebuild import inner_baseline
from gaia_anchor_experiment import color_design

ROOT=Path('eval/matched_sap_hgb_20261014')
BASE=Path('eval/full_rebuild_20261012')
VARIABLE=Path('eval/long_period_variable_expansion_20261014')
BRANCHES=['sap','adjusted_sap','pdcsap']
REF=SECTOR+SPATIAL+STRIPE+['pred_knn']
SEED=2026101407

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def hashkey(x):return hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest()

def verify_implementation():
    cfg=json.loads((ROOT/'freeze.json').read_text())
    amendment=json.loads((ROOT/'implementation_amendment.json').read_text())
    assert amendment['original_sha256']==cfg['script_sha256']==sha(ROOT/'source_snapshot.py')
    assert amendment['amended_sha256']==sha(__file__)
    return cfg

def stratified(stars,n):
    s=stars.copy();s['color_bin']=pd.cut(s.bp_rp,[-np.inf,.5,1,1.5,2,2.5,3,np.inf],labels=False)
    s['mag_bin']=pd.cut(s.tmag,[-np.inf,8,10,12,np.inf],labels=False)
    s['hash']=s.physical_group_id.map(hashkey);s=s.sort_values('hash')
    s['within_bin_rank']=s.groupby(['color_bin','mag_bin']).cumcount()
    return s.sort_values(['within_bin_rank','hash']).head(n)

def freeze():
    assert not ROOT.exists();ROOT.mkdir();(ROOT/'fits').mkdir()
    features=json.loads(Path('eval/restricted_input_hgb_20261013/restricted/complete.json').read_text())['features']
    f=pd.read_parquet(BASE/'features.parquet')
    columns=list(dict.fromkeys(['source_row','tic_id','sector','role','evaluation_group','physical_group_id','reference_fold','fit_row','has_gaia_anchor','systematic_score','gaiarp','bp_rp','tmag','crowdsap','gaia_log_anchor']+features))
    f=f[columns];q=f[f.has_gaia_anchor&f.systematic_score.gt(.95)&(f.role.ne('train')|f.fit_row)]
    selected=[];source=[]
    for group,n in [('train',512),('calibration',64),('confirmation',64)]:
        g=q[q.evaluation_group.eq(group)];count=g.groupby('tic_id').size();ids=count[count>=8].index
        stars=g[g.tic_id.isin(ids)].groupby('tic_id',as_index=False).first()
        pick=stratified(stars,n);assert len(pick)==n
        source.append(pick[['tic_id','physical_group_id','evaluation_group','tmag','bp_rp','color_bin','mag_bin','hash']])
        for tic in pick.tic_id:
            eligible=g[g.tic_id.eq(tic)].sort_values('sector')
            eligible=eligible.iloc[np.rint(np.linspace(0,len(eligible)-1,8)).astype(int)].copy();eligible['selection']='qualified'
            extra=f[f.tic_id.eq(tic)&~f.systematic_score.gt(.95)].sort_values('sector')
            if len(extra):
                extra=extra.iloc[np.unique(np.rint(np.linspace(0,len(extra)-1,min(2,len(extra)))).astype(int))].copy();extra['selection']='unqualified_context'
                eligible=pd.concat([eligible,extra])
            selected.append(eligible)
    z=pd.concat(selected,ignore_index=True);assert not z.groupby('physical_group_id').evaluation_group.nunique().gt(1).any()
    meta=pd.read_parquet('eval/full_geometry_collection_20261012/full_physical_features.parquet',columns=['source_row','filename','data_uri'])
    z=z.merge(meta,on='source_row',validate='one_to_one')
    z.to_parquet(ROOT/'selected_features.parquet',index=False);pd.concat(source).to_parquet(ROOT/'source_selection.parquet',index=False)
    cached={}
    # Include the immediately preceding SAP pilot's verified archive products.
    for base in [Path('tess_cache'),Path('eval/upstream_pixel_diagnosis_20261014/sap_pilot/fits')]:
        for p in base.rglob('*_lc.fits*'):
            if p.is_file():cached.setdefault(p.name.removesuffix('.gz'),str(p))
    z['cached_path']=z.filename.map(cached)
    z[['source_row','tic_id','sector','filename','data_uri','cached_path']].to_parquet(ROOT/'requests.parquet',index=False)
    inputs=[BASE/'features.parquet',BASE/'cohort_manifest.parquet',Path('research_defaults.json'),Path('eval/hgb_reference_repair.py'),Path('eval/full_physical_rebuild.py'),Path('eval/gaia_anchor_experiment.py')]
    inputs+=list((BASE/'physical').glob('*.pkl'))
    save(ROOT/'freeze.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),script_sha256=sha(__file__),selected_sha256=sha(ROOT/'selected_features.parquet'),input_hashes={str(p):sha(p) for p in inputs},features=features,seed=SEED,params=PARAMS,rounds=5,
      sample='512 training,64 validation-calibration,64 validation-confirmation stars, hash/round-robin stratified in fixed magnitude/color bins. Eight evenly spaced individually qualified sectors each; up to two unqualified sectors retained for all-sector evaluation. Physical source roles unchanged. Examined development populations, not pristine holdouts. No test labels opened.',
      branches=BRANCHES,measurement='Same Q0 finite positive SAP and PDCSAP cadence mask and positive header-adjusted SAP. Header adjustment [SAP-(1-CROWDSAP)*median(Q0 positive SAP)]/FLFRCSAP. Report total, SAP-valid and common cadence counts. Keep all acquisition failures without replacement.',
      fairness='All three branches have identical rows,102 restricted predictors, model settings, five rounds and seed. Exclude target variability/PDC diagnostics and flux centroids already removed by the restricted allowlist. Rebuild every reference label/feature and nested Gaia color transfer independently for each flux product. Original full-population HGB remains contextual comparator, not matched capacity/data.',
      reference='Five existing physical-group folds. Bank contains only selected qualified training rows; no validation/test labels. Entire outer query fold excluded; inner reference predictions exclude query physical group. Fallback unity if no same sector/camera/CCD reference, report it.',
      anchor='Training-only outer-fold-excluded Huber cubic color/passband transfer as in full rebuild; retain Gaia RP anchor at every round. No per-star target centering. All selected stars have anchors; report any application fallback.',
      context='Inherited non-reference metadata/deviation/geometry predictors retain original all-collected-sector context. Same context in every branch; no new measured variable flux enters features. These are not certified causal instrumental-only inputs.',
      selection='Select lower qualified median per-star CV between raw-SAP and header-adjusted-SAP HGB on validation-calibration only, fixed round5. Always retain matched PDCSAP HGB and uncorrected controls. Report Gaia-anchor residuals, paired star intervals and all-sector/qualified attrition before making any promotion claim.',
      science='After selection, apply supported models to all eight previously examined long-period variables. Metadata support gate: >=3 selected training stars and >=1 selected calibration star within0.75Tmag and0.35BP-RP; Gaia/color support and reference fallbacks reported. Unsupported stars, including bright-red case if still unsupported, are not inferred. Retain original3/5/7-day boundary tests, all/individual-TARS views, full trajectories and sector-factor changes. No tuning on variable outcomes, no model promotion, no independent stellar amplitude truth.',
      scope='Larger HGB pilot, not full-population retraining or proof of a noise floor. Conditional invariance with fixed metadata is structural; paired injection cannot certify baseline offsets.'))
    shutil.copy2(__file__,ROOT/'source_snapshot.py')
    print(z.groupby(['evaluation_group','selection']).size(),flush=True)

def acquire():
    def one(r):
        for attempt in range(3):
            try:
                if isinstance(r.cached_path,str) and Path(r.cached_path).exists():p=Path(r.cached_path)
                else:
                    p=ROOT/'fits'/(r.filename+'.gz')
                    if not p.exists():
                        assert shutil.disk_usage(ROOT).free>2*2**30,'2GiB disk reserve'
                        q=requests.get('https://mast.stsci.edu/api/v0.1/Download/file',params={'uri':r.data_uri},timeout=(15,75));q.raise_for_status();assert q.content.startswith(b'SIMPLE')
                        p.write_bytes(gzip.compress(q.content,compresslevel=3,mtime=0))
                with fits.open(p,memmap=False) as h:
                    assert int(h[0].header['TICID'])==r.tic_id and int(h[0].header['SECTOR'])==r.sector
                    d=h[1].data;sap=np.asarray(d['SAP_FLUX'],float);pdc=np.asarray(d['PDCSAP_FLUX'],float)
                    initial=(d['QUALITY']==0)&np.isfinite(d['TIME'])&np.isfinite(sap)&(sap>0)
                    adjusted=(sap-(1-float(h[1].header['CROWDSAP']))*np.median(sap[initial]))/float(h[1].header['FLFRCSAP'])
                    common=initial&np.isfinite(pdc)&(pdc>0)&np.isfinite(adjusted)&(adjusted>0)
                    assert common.sum()>=50
                    return dict(source_row=r.source_row,path=str(p),sha256=sha(p),status='available',total=len(d),sap_valid=int(initial.sum()),common=int(common.sum()),sap=float(np.median(sap[common])),adjusted_sap=float(np.median(adjusted[common])),pdcsap=float(np.median(pdc[common])))
            except Exception as e:err=repr(e)
        return dict(source_row=r.source_row,status='failed',error=err)
    req=pd.read_parquet(ROOT/'requests.parquet');out=[]
    if (ROOT/'acquisition.parquet').exists():
        old=pd.read_parquet(ROOT/'acquisition.parquet');out=old[old.status.eq('available')].to_dict('records');req=req[~req.source_row.isin(old[old.status.eq('available')].source_row)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(one,r) for r in req.itertuples()]):
            out.append(future.result())
            if len(out)%100==0:
                pd.DataFrame(out).to_parquet(ROOT/'acquisition.parquet',index=False);print('Acquired',len(out),'rows',flush=True)
    pd.DataFrame(out).to_parquet(ROOT/'acquisition.parquet',index=False)
    print('Acquisition complete',pd.DataFrame(out).status.value_counts().to_dict(),flush=True)

def prepare(branch):
    dest=ROOT/branch;assert not dest.exists();dest.mkdir()
    f=pd.read_parquet(ROOT/'selected_features.parquet');a=pd.read_parquet(ROOT/'acquisition.parquet')
    assert len(f)==len(a) and a.status.eq('available').all(),'No silent acquisition attrition'
    f=f.merge(a[['source_row',branch]],on='source_row',validate='one_to_one').rename(columns={branch:'sector_median'})
    tr=f[f.role.eq('train')&f.systematic_score.gt(.95)&f.fit_row].copy()
    g=tr.groupby('tic_id').sector_median
    tr['flux_offset_loo']=tr.sector_median/((g.transform('sum')-tr.sector_median)/(g.transform('size')-1))
    tr[['source_row','tic_id','sector','cam','ccd','col','row','physical_group_id','reference_fold','systematic_score','flux_offset_loo']].to_parquet(dest/'reference_bank.parquet',index=False)
    f=f.drop(columns=REF+['gaia_log_anchor'],errors='ignore');blocks=[];audit=[]
    for fold in range(5):
        r=tr[tr.reference_fold.ne(fold)].copy();query=f[f.reference_fold.eq(fold)].copy()
        assert not set(r.physical_group_id)&set(query.physical_group_id)
        refs=reference_features(r,query);refs.index=query.index
        inner=inner_baseline(r)
        stars=r.groupby('tic_id').agg(rp=('gaiarp','first'),color=('bp_rp','first'),crowd=('crowdsap','median'))
        stars['response']=np.log(r.sector_median/inner).groupby(r.tic_id).median()+.4*np.log(10)*(stars.rp-10)
        keep=np.isfinite(stars.rp)&np.isfinite(stars.color)&stars.crowd.gt(.95)&stars.rp.between(7,13.2)&stars.color.between(0,3.3)
        assert keep.sum()>=30
        model=HuberRegressor(epsilon=1.35,alpha=.0001,max_iter=1000,tol=1e-8).fit(color_design(stars.loc[keep,'color']),stars.loc[keep,'response'])
        joblib.dump(model,dest/f'color_fold_{fold}.pkl')
        query['gaia_log_anchor']=-.4*np.log(10)*(query.gaiarp-10)+model.predict(color_design(query.bp_rp))
        blocks.append(query.join(refs))
        audit.append(dict(fold=fold,reference_rows=len(r),reference_stars=r.tic_id.nunique(),color_stars=int(keep.sum()),query_rows=len(query),query_group_overlap=0,nontraining_reference_labels=0,reference_fallbacks=int(refs.ref_count.eq(0).sum())))
    data=pd.concat(blocks).sort_values('source_row').reset_index(drop=True);data.to_parquet(dest/'features_and_targets.parquet',index=False)
    save(dest/'preparation.json',dict(audit=audit,missing_anchors=int((~data.has_gaia_anchor).sum()),rows=len(data),training_rows=len(tr)))
    print(branch,'prepared',flush=True)

def fit(branch):
    cfg=verify_implementation()
    dest=ROOT/branch;f=pd.read_parquet(dest/'features_and_targets.parquet');tr=f.role.eq('train')&f.systematic_score.gt(.95)&f.fit_row
    names=cfg['features'];median=f.loc[tr,names].replace([np.inf,-np.inf],np.nan).median().fillna(0)
    joblib.dump(median,dest/'imputation.pkl');x=f[names].astype(float).replace([np.inf,-np.inf],np.nan).fillna(median).to_numpy(np.float32)
    total=np.zeros(len(f));history=[]
    base=np.log(f.sector_median/f.pred_knn)-f.gaia_log_anchor
    # Perturbing the anchor must change every round target, not cancel by centering.
    for rnd in range(5):
        target=base-total;probe=np.log(f.sector_median/f.pred_knn)-(f.gaia_log_anchor+.01)-total
        np.testing.assert_allclose(probe-target,-.01,atol=1e-12)
        start=time.monotonic();model=HistGradientBoostingRegressor(random_state=SEED+rnd,**PARAMS)
        model.fit(x[tr],target[tr]);total+=model.predict(x);joblib.dump(model,dest/f'round_{rnd}.pkl')
        history.append(dict(round=rnd+1,seconds=time.monotonic()-start));print(branch,'round',rnd+1,flush=True)
    p=f[['source_row','tic_id','sector','physical_group_id','role','evaluation_group','systematic_score','gaia_log_anchor','has_gaia_anchor','sector_median','ref_count']].copy()
    p['correction']=f.pred_knn*np.exp(total);p['corrected']=f.sector_median/p.correction
    p.to_parquet(dest/'predictions.parquet',index=False);save(dest/'complete.json',dict(features=names,history=history,target='Gaia RP',anchor_retained_every_round=True))

def score():
    rows=[];stars=[]
    for branch in BRANCHES:
        p=pd.read_parquet(ROOT/branch/'predictions.parquet')
        for group in ['calibration','confirmation']:
            for view in ['all','qualified']:
                z=p[p.evaluation_group.eq(group)].copy()
                if view=='qualified':z=z[z.systematic_score.gt(.95)]
                for method in ['sector_median','corrected']:
                    cv=z.groupby('tic_id')[method].agg(lambda x:100*np.std(x,ddof=1)/np.mean(x))
                    error=np.log(z[method])-z.gaia_log_anchor
                    rows.append(dict(branch=branch,group=group,view=view,method=method,stars=len(cv),sectors=len(z),median_cv_pct=float(cv.median()),anchor_log_rmse=float(np.sqrt(np.mean(error**2))),reference_fallbacks=int(z.ref_count.eq(0).sum())))
                    stars.extend(dict(branch=branch,group=group,view=view,method=method,tic_id=int(t),cv_pct=float(v)) for t,v in cv.items())
    # Selection reads only calibration metrics even though the fixed reporting
    # table is computed together. No alternative settings exist to tune here.
    eligible=[x for x in rows if x['group']=='calibration' and x['view']=='qualified' and x['method']=='corrected' and x['branch']!='pdcsap']
    save(ROOT/'selection.json',dict(selected_sap=min(eligible,key=lambda x:x['median_cv_pct'])['branch'],calibration_candidates=eligible,basis='Qualified validation-calibration CV, fixed five rounds',science_labels_used=False))
    pd.DataFrame(rows).to_csv(ROOT/'quiet_scores.csv',index=False);pd.DataFrame(stars).to_csv(ROOT/'quiet_per_star.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action in ['prepare','fit']:globals()[action](sys.argv[2])
    else:globals()[action]()
