"""Retained Gaia RP target, with cross-fitted training-only color calibration."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hgb_reference_repair import PARAMS,group_median,per_star,paired_bootstrap,sha256,write_json,log
from investigate_residual_atlas import save_json

PARENT=Path('eval/hgb_reference_repair_20261010')
OUT=Path('eval/gaia_anchor_20261011')


def color_design(color):
    c=np.clip(np.asarray(color,float),-.6,3.5)-.9
    return np.column_stack([c,c*c,c*c*c])


def prepare():
    path=OUT/'features.parquet'
    if path.exists():return
    f=pd.read_parquet(PARENT/'clean_features.parquet')
    stars=f.groupby('tic_id').agg(role=('role','first'),fold=('reference_fold','first'),
        rp=('gaiarp','first'),color=('bp_rp','first'),crowd=('crowdsap','median'),
        flux=('sector_median','median'))
    logratio=np.log(f.sector_median/f.pred_knn)
    stars['log_flux_over_knn']=logratio.groupby(f.tic_id).median()
    stars['log_rp_flux']=-.4*np.log(10)*(stars.rp-10)
    stars['color_response']=stars.log_flux_over_knn-stars.log_rp_flux
    good=stars.rp.notna()&stars.color.notna()
    cal=good&(stars.role=='train')&(stars.crowd>.95)&stars.rp.between(7,13.2)&stars.color.between(0,3.3)
    models={};counts={};stars['color_prediction']=np.nan
    for fold in range(10):
        fit=cal&(stars.fold!=fold);query=good&(stars.fold==fold)
        model=HuberRegressor(epsilon=1.35,alpha=.0001,max_iter=500,tol=1e-8)
        model.fit(color_design(stars.loc[fit,'color']),stars.loc[fit,'color_response'])
        stars.loc[query,'color_prediction']=model.predict(color_design(stars.loc[query,'color']))
        models[fold]=model;counts[fold]=int(fit.sum())
    stars['gaia_log_anchor']=stars.log_rp_flux+stars.color_prediction
    stars['anchor_residual_pct']=100*np.expm1(stars.log_flux_over_knn-stars.gaia_log_anchor)
    f['gaia_log_anchor']=f.tic_id.map(stars.gaia_log_anchor)
    f['has_gaia_anchor']=f.gaia_log_anchor.notna()
    # A missing-anchor training star gets the baseline relative target.
    ids=f.tic_id.to_numpy();fallback=group_median(logratio.to_numpy(),ids)
    f['gaia_log_anchor']=f.gaia_log_anchor.fillna(pd.Series(fallback,index=f.index))
    tr=f.role=='train';a=logratio[tr].to_numpy();b=a-f.loc[tr,'gaia_log_anchor'].to_numpy()
    centered_difference=np.max(abs((a-group_median(a,ids[tr]))-(b-group_median(b,ids[tr]))))
    assert centered_difference<1e-12
    f.to_parquet(path,index=False);stars.to_parquet(OUT/'star_anchors.parquet')
    joblib.dump(models,OUT/'color_calibrations.pkl')
    result={'calibration_stars_per_fold':counts,'centering_invariance_max_difference':centered_difference,
        'coverage':{role:{'rows':len(g),'missing_anchor_rows':int((~g.has_gaia_anchor).sum()),
            'missing_anchor_stars':int(g.loc[~g.has_gaia_anchor,'tic_id'].nunique())} for role,g in f.groupby('role')},
        'anchor_residual_by_role':{role:g.anchor_residual_pct.quantile([.05,.16,.5,.84,.95]).to_dict() for role,g in stars.groupby('role')},
        'parent_prepared_sha256':sha256(PARENT/'prepared.json'),'script_sha256':sha256(__file__)}
    save_json(OUT/'prepared.json',result)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    v=stars[(stars.role=='validation')&good]
    axes[0].scatter(v.rp,v.log_flux_over_knn/np.log(10),s=6,alpha=.3)
    axes[0].set(xlabel='Gaia RP magnitude',ylabel='log10(TESS sector flux / KNN), star median',title='External brightness anchor')
    axes[1].scatter(v.color,v.color_response,s=6,alpha=.3,label='Validation stars')
    color=np.linspace(0,3.3,100)
    for fold,model in models.items():axes[1].plot(color,model.predict(color_design(color)),color='#b55338',alpha=.2)
    axes[1].set(xlabel='Gaia BP−RP (mag)',ylabel='log(TESS / relative RP flux)',title='Training-only color transfer')
    axes[2].scatter(v.color,v.anchor_residual_pct,s=6,alpha=.3)
    axes[2].axhline(0,color='.6');axes[2].set(xlabel='Gaia BP−RP (mag)',ylabel='Star baseline − Gaia prediction (%)',title='External-anchor scatter (not instrumental noise)')
    fig.savefig(OUT/'gaia_anchor_calibration.png',dpi=180);plt.close(fig)
    log('Gaia anchor preparation complete')


def fit(case):
    run=OUT/case;run.mkdir(exist_ok=True)
    config={'case':case,'alpha':0. if case=='absolute' else .5,'seed':42,'rounds':5,
        'params':PARAMS,'prepared_sha256':sha256(OUT/'prepared.json'),'script_sha256':sha256(__file__)}
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config
    else:write_json(run/'config.json',config)
    if (run/'results.json').exists():return
    f=pd.read_parquet(OUT/'features.parquet');frames={r:g.reset_index(drop=True) for r,g in f.groupby('role')};del f
    feats=json.loads((PARENT/'prepared.json').read_text())['features_deviation']
    med=frames['train'][feats].median().fillna(0);joblib.dump(med,run/'imputation.pkl')
    x={r:g[feats].fillna(med).to_numpy(np.float32) for r,g in frames.items()}
    tr=frames['train'];ids=tr.tic_id.to_numpy();good=tr.has_gaia_anchor.to_numpy()
    residual=np.log(tr.sector_median.to_numpy()/tr.pred_knn.to_numpy())-tr.gaia_log_anchor.to_numpy()
    delta={r:np.zeros(len(frames[r])) for r in ['validation','test']}
    history=[]
    for rnd in range(5):
        residual-=np.where(good,config['alpha'],1.)*group_median(residual,ids)
        path=run/f'round_{rnd}.pkl';start=time.monotonic()
        if path.exists():model=joblib.load(path)
        else:
            model=HistGradientBoostingRegressor(random_state=42+rnd,**PARAMS)
            log(f'{case} fitting round {rnd+1}');model.fit(x['train'],residual)
            temp=path.with_suffix('.pkl.tmp');joblib.dump(model,temp);temp.replace(path)
        residual-=model.predict(x['train'])
        for role in delta:delta[role]+=model.predict(x[role])
        history.append({'round':rnd+1,'seconds':time.monotonic()-start,
            'median_abs_training_residual':float(np.median(abs(residual)))})
        write_json(run/'progress.json',{'status':'training','completed_rounds':rnd+1,'history':history})
        log(f'{case} round {rnd+1} complete')
    fallback={}
    for role in delta:
        old=pd.read_parquet(PARENT/f'clean_deviation/seed_42/{role}_predictions.parquet').set_index('source_row')
        fallback[role]=old.loc[frames[role].source_row,'correction'].to_numpy()
    def correction(role,beta):
        g=frames[role];pred=g.pred_knn.to_numpy()*np.exp(beta*delta[role])
        return np.where(g.has_gaia_anchor,pred,fallback[role])
    curve=[{'beta':float(beta),'validation_cv_pct':float(per_star(frames['validation'],correction('validation',beta)).median())} for beta in np.linspace(.5,2,31)]
    best=min(curve,key=lambda r:r['validation_cv_pct']);metrics={}
    for role in delta:
        g=frames[role];pred=correction(role,best['beta'])
        baseline=per_star(g,fallback[role]);candidate=per_star(g,pred)
        raw=per_star(g,np.ones(len(g)))
        table=pd.DataFrame({'raw':raw,'baseline':baseline,'candidate':candidate});table.to_csv(run/f'{role}_per_star_cv.csv')
        saved=g[['source_row','tic_id','sector','cam','ccd','sector_median','reference_fold','pred_knn','has_gaia_anchor','gaia_log_anchor']].copy()
        saved['log_delta']=delta[role];saved['correction']=pred;saved.to_parquet(run/f'{role}_predictions.parquet',index=False)
        metrics[role]={'cv_pct':float(candidate.median()),'baseline_cv_pct':float(baseline.median()),
            'p90_cv_pct':float(candidate.quantile(.9)),'fraction_worse_than_raw':float((candidate>raw).mean()),
            'paired_gain':paired_bootstrap(baseline,candidate,20261014)}
    result={'case':case,'best_beta':best['beta'],'beta_curve':curve,'metrics':metrics,'history':history}
    write_json(run/'results.json',result);write_json(run/'progress.json',{'status':'complete','metrics':metrics})


def summarize():
    results={c:json.loads((OUT/c/'results.json').read_text()) for c in ['absolute','soft']}
    save_json(OUT/'comparison.json',results)
    lines=['# Gaia external-anchor comparison','',
        '| Target | Validation CV (%) | Test CV (%) | Validation-selected strength |','|---|---:|---:|---:|',
        '| Relative control, seed 42 | 0.813614 | 0.815097 | 1.20 |']
    for c,r in results.items():lines.append(f'| {c} | {r["metrics"]["validation"]["cv_pct"]:.6f} | {r["metrics"]["test"]["cv_pct"]:.6f} | {r["best_beta"]:.2f} |')
    lines+=['','Same original roles, features, five rounds and tree capacity. RP/color calibration uses other training folds only. Test outcomes do not select model strength.','',
        'An external anchor can have stellar/passband/catalog scatter much larger than its formal photometric error. This experiment uses the cached TIC Gaia magnitudes, not a fresh quality-screened Gaia DR3 sample.','',
        '![Calibration](gaia_anchor_calibration.png)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit','batch','summarize']);p.add_argument('--case',choices=['absolute','soft']);a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stage=='prepare':prepare()
    elif a.stage=='fit':fit(a.case)
    elif a.stage=='summarize':summarize()
    else:
        status={'status':'preparing','started_utc':pd.Timestamp.now(tz='UTC').isoformat()};write_json(OUT/'batch_status.json',status)
        try:
            prepare();status['status']='training';write_json(OUT/'batch_status.json',status)
            env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
            def lane(case):
                with (OUT/f'{case}.log').open('a') as stream:
                    subprocess.run([sys.executable,__file__,'fit','--case',case],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(lane,c) for c in ['absolute','soft']]
                for future in futures:future.result()
            summarize();status['status']='complete'
        except Exception as exc:status.update(status='failed',error=str(exc));raise
        finally:write_json(OUT/'batch_status.json',status)


if __name__=='__main__':main()
