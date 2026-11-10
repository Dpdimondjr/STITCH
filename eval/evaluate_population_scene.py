"""Flux-independent catalog/aperture feature on a frozen random validation sample."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_scene_investigation as scene
from hgb_reference_repair import paired_bootstrap
from investigate_residual_atlas import save_json,log

OUT=Path('eval/residual_investigation_20261011/population_scene')
BASE=OUT.parent


def metadata_feature(path,tic,sector,catalog_root,midtime_override=None,position_function=None):
    with fits.open(path,memmap=True) as h:
        header=h[2].header;bits=h[2].data.copy();shape=bits.shape
        cam=int(h[0].header['CAMERA']);ccd=int(h[0].header['CCD'])
        assert int(h[0].header['TICID'])==tic and int(h[0].header['SECTOR'])==sector
        nominal_c=float(h[1].header['CROWDSAP']);nominal_q=float(h[1].header['FLFRCSAP'])
        times=h[1].data['TIME'];good=(h[1].data['QUALITY']==0)&np.isfinite(times)
        midtime=float(np.median(times[good])) if midtime_override is None else midtime_override
        x,y=WCS(header).celestial.all_world2pix(header['RA_OBJ'],header['DEC_OBJ'],0)
        col,row=WCS(header,key='P').all_pix2world(x,y,0)
    if not (nominal_c>0 and nominal_q>0):raise ValueError('Nonpositive nominal crowding/capture')
    mask=((bits.astype(int)&2)!=0)&((bits.astype(int)&1)!=0)
    prf=scene.PRF.TESS_PRF(cam,ccd,sector,float(np.clip(col,45.001,2091.999)),float(np.clip(row,1.001,2047.999)),
        localdatadir=str(Path('tess_cache/prf_models')/('start_s0001' if sector<4 else 'start_s0004')))
    cat=(position_function or scene.catalog_positions)(catalog_root/f'tic_{tic}.ecsv',tic,header,midtime)
    target=cat[cat.is_target].iloc[0]
    near=cat.usable&cat.x.between(-6,shape[1]+5)&cat.y.between(-6,shape[0]+5)
    cat=cat[near|cat.is_target].copy()
    templates=scene.templates_at(prf,cat[['x','y']].to_numpy(),shape)
    cat['stamp_capture']=templates.sum(axis=(1,2));cat['aperture_capture']=templates[:,mask].sum(axis=1)
    cat['relative_flux']=10**(-.4*(cat.tmag-target.tmag))
    cat['expected_stamp_contribution']=cat.stamp_capture*cat.relative_flux
    cat['expected_aperture_contribution']=cat.aperture_capture*cat.relative_flux
    cat=cat.sort_values(['is_target','expected_stamp_contribution'],ascending=[False,False])
    keep=[];rejected=[];extended=[]
    for i,s in cat.iterrows():
        if not s.is_target and s.expected_stamp_contribution<.001:rejected.append(i);continue
        if keep and np.min(np.linalg.norm(cat.loc[keep,['x','y']].to_numpy()-s[['x','y']].to_numpy(float),axis=1))<.15:
            rejected.append(i);continue
        if len(keep)>=18:rejected.append(i);continue
        keep.append(i)
    # Prespecified exploratory source-truncation control; primary stays frozen.
    for i,s in cat.iterrows():
        if not s.is_target and s.expected_stamp_contribution<1e-5:continue
        if extended and np.min(np.linalg.norm(cat.loc[extended,['x','y']].to_numpy()-s[['x','y']].to_numpy(float),axis=1))<.15:continue
        extended.append(i)
    selected=cat.loc[keep];assert selected.iloc[0].is_target
    expected=selected.expected_aperture_contribution.to_numpy();capture=selected.iloc[0].aperture_capture
    crowd=expected[0]/expected.sum()
    crowd_extended=expected[0]/cat.loc[extended,'expected_aperture_contribution'].sum()
    t=Table.read(catalog_root/f'tic_{tic}.ecsv')
    rp=pd.Series(np.asarray(np.ma.filled(t['gaiarp'],np.nan),float),index=np.asarray(t['ID']).astype(str))
    rp_delta=selected.tic_id.map(rp)-rp.get(str(tic),np.nan)
    relative_rp=np.where(rp_delta.notna(),10**(-.4*rp_delta),selected.relative_flux)
    expected_rp=selected.aperture_capture.to_numpy()*relative_rp
    crowd_rp=expected_rp[0]/expected_rp.sum()
    capture_term=np.log(capture/nominal_q);crowd_term=np.log(nominal_c/crowd)
    return dict(tic_id=tic,sector=sector,actual_cam=cam,actual_ccd=ccd,midtime=midtime,
        predicted_log_mismatch=capture_term+crowd_term,capture_term=capture_term,crowding_term=crowd_term,
        extended_log_mismatch=capture_term+np.log(nominal_c/crowd_extended),
        rp_log_mismatch=capture_term+np.log(nominal_c/crowd_rp),
        nominal_log_factor=np.log(nominal_c/nominal_q),catalog_scene_crowding=crowd,
        prf_capture=capture,nominal_crowding=nominal_c,nominal_capture=nominal_q,
        sources=len(keep),extended_sources=len(extended),aperture_pixels=int(mask.sum()),
        col=float(col),row=float(row),rp_fallback_sources=int(rp_delta.isna().sum()),
        omitted_expected_aperture_flux_relative_target=float(cat.loc[rejected,'expected_aperture_contribution'].sum()/expected[0]) if rejected else 0.)


def parity():
    checks=[];mask_checks=[];time_checks=[]
    for name in ['scene','scene_confirmation','catalog_prospective']:
        p=BASE/name;sample=pd.read_parquet(p/'sample.parquet')
        prior=pd.read_parquet(p/('measurements.parquet' if name=='catalog_prospective' else 'mechanism_comparison.parquet')).set_index(['tic_id','sector'])
        col='predicted_log_mismatch' if name=='catalog_prospective' else 'catalog_only_mismatch'
        for r in sample.itertuples(index=False):
            with fits.open(r.lc_path,memmap=True) as a,fits.open(r.tpf_path,memmap=True) as b:
                mask_checks.append(bool(np.array_equal(a[2].data,b[2].data)
                    and WCS(a[2].header).to_header()==WCS(b[2].header).to_header()
                    and WCS(a[2].header,key='P').to_header()==WCS(b[2].header,key='P').to_header()))
            midtime=float(np.median(np.load(scene.CACHE/f'tic_{r.tic_id}_s{r.sector:04d}.npz')['time']))
            f=metadata_feature(r.lc_path,r.tic_id,r.sector,p/'catalogs',midtime)
            checks.append(abs(f['predicted_log_mismatch']-prior.loc[(r.tic_id,r.sector),col]))
            operational=metadata_feature(r.lc_path,r.tic_id,r.sector,p/'catalogs')
            time_checks.append(abs(f['predicted_log_mismatch']-operational['predicted_log_mismatch']))
    assert all(mask_checks);np.testing.assert_allclose(checks,0,atol=1e-12)
    assert max(time_checks)<1e-5
    save_json(OUT/'lc_feature_parity.json',{'observations':len(checks),'all_masks_and_wcs_identical':all(mask_checks),
        'max_feature_difference_same_time':max(checks),'max_feature_difference_lc_only_time':max(time_checks),'passed':True})
    log('LC-only feature parity complete on all 640 prior observations')


def one_star(task):
    tic,records=task;path=OUT/'features_by_star'/f'tic_{tic}.parquet'
    if path.exists():return len(pd.read_parquet(path))
    rows=[]
    for row in records:
        record={'tic_id':tic,'sector':row['sector'],'available':False}
        try:
            if not isinstance(row['local_path'],str) or not Path(row['local_path']).exists():raise ValueError('LC unavailable')
            record.update(metadata_feature(row['local_path'],tic,row['sector'],OUT/'catalogs'))
            record['available']=True
            # Flux is read only here for an independent version/lineage check.
            with fits.open(row['local_path'],memmap=True) as h:
                flux=h[1].data['PDCSAP_FLUX'];good=(h[1].data['QUALITY']==0)&np.isfinite(flux)
                record['lc_sector_median']=float(np.median(flux[good]));record['lc_procver']=h[0].header.get('PROCVER')
        except Exception as exc:record['error']=str(exc)
        rows.append(record)
    pd.DataFrame(rows).to_parquet(path,index=False);return len(rows)


def extract():
    assert json.loads((OUT/'lc_feature_parity.json').read_text())['passed']
    sample=pd.read_parquet(OUT/'sample.parquet');products=pd.read_csv(OUT/'products.csv')
    f=sample.merge(products[['tic_id','sector','local_path']],on=['tic_id','sector'],how='left',validate='one_to_one')
    (OUT/'features_by_star').mkdir(exist_ok=True)
    tasks=[(int(tic),g.to_dict('records')) for tic,g in f.groupby('tic_id')]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one_star,task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%20==0:log(f'Population metadata {i}/300 stars')
    measurements=pd.concat([pd.read_parquet(OUT/'features_by_star'/f'tic_{tic}.parquet') for tic,_ in tasks],ignore_index=True)
    measurements.to_parquet(OUT/'measurements.parquet',index=False)
    log('Population metadata extraction complete')


def cv(frame,values):
    g=pd.Series(np.asarray(values),index=frame.index).groupby(frame.tic_id)
    return 100*g.std(ddof=0)/g.mean()


def summarize():
    sample=pd.read_parquet(OUT/'sample.parquet');features=pd.read_parquet(OUT/'measurements.parquet')
    pred=pd.read_parquet('eval/hgb_reference_validation_20261010/ensembles/clean_deviation/validation_predictions.parquet')
    f=sample.merge(features,on=['tic_id','sector'],validate='one_to_one').merge(pred[['source_row','sector_median','correction']],on='source_row',validate='one_to_one')
    flux=f.sector_median/f.correction;baseline=cv(f,flux)
    methods={'primary_half_strength':.5*f.predicted_log_mismatch,
        'capture_only':.5*f.capture_term,'crowding_only':.5*f.crowding_term,
        'extended_sources':.5*f.extended_log_mismatch,'rp_relative_brightness':.5*f.rp_log_mismatch,
        'nominal_pipeline_terms':.5*f.nominal_log_factor}
    stars=sample.groupby('tic_id').agg(population_half=('population_half','first'),tmag=('tmag','median'),crowdsap=('crowdsap','median'),sectors=('sector','size'))
    stars['baseline']=baseline;stars['complete_features']=f.groupby('tic_id').available.all()
    results={}
    for name,delta in methods.items():
        d=delta.where(f.available&np.isfinite(delta),0.)
        values=cv(f,flux/np.exp(d));stars[name]=values
        results[name]={'baseline_cv_pct':float(baseline.median()),'candidate_cv_pct':float(values.median()),
            'p90_before':float(baseline.quantile(.9)),'p90_after':float(values.quantile(.9)),
            'fraction_improved':float((values<baseline).mean()),'paired_gain':paired_bootstrap(baseline,values,20261015),
            'halves':{half:{'stars':len(g),'before':float(g.baseline.median()),'after':float(g[name].median()),
                'paired_gain':paired_bootstrap(g.baseline,g[name],20261015)} for half,g in stars.groupby('population_half')}}
    f['baseline_flux']=flux;f['primary_flux']=flux/np.exp(methods['primary_half_strength'].where(f.available,0.).fillna(0))
    f['lc_median_difference_pct']=100*(f.lc_sector_median/f.sector_median-1)
    f['log_residual']=np.log(flux)-np.log(flux).groupby(f.tic_id).transform('mean')
    f['centered_mismatch']=f.predicted_log_mismatch-f.groupby('tic_id').predicted_log_mismatch.transform('mean')
    complete=stars[stars.complete_features]
    result={'stars':len(stars),'observations':len(f),'available_features':int(f.available.sum()),'complete_stars':len(complete),
        'primary':results['primary_half_strength'],'exploratory_controls':{k:v for k,v in results.items() if k!='primary_half_strength'},
        'complete_stars_gain':paired_bootstrap(complete.baseline,complete.primary_half_strength,20261015),
        'geometry_disagreements':int(((f.cam!=f.actual_cam)|(f.ccd!=f.actual_ccd)).sum()),
        'flux_lineage_abs_difference_pct':f.lc_median_difference_pct.abs().quantile([.5,.95,.99,1]).to_dict(),
        'flux_lineage_over_0_1pct':int((f.lc_median_difference_pct.abs()>.1).sum()),
        'limitations':['Random 300-star validation subset, not the full test benchmark.',
            'Excludes the 160 earlier pilot stars; all retained sectors included.',
            'Primary strength and feature frozen; secondary controls exploratory.',
            'Missing features receive zero adjustment in primary analysis.']}
    save_json(OUT/'results.json',result);stars.to_csv(OUT/'per_star_cv.csv');f.to_parquet(OUT/'observations.parquet',index=False)
    f.loc[~f.available,['tic_id','sector','error']].to_csv(OUT/'feature_failures.csv',index=False) if 'error' in f else None
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    for half,g in stars.groupby('population_half'):
        axes[0,0].scatter(g.baseline,g.primary_half_strength,s=18,alpha=.6,label=half)
    limit=max(stars.baseline.quantile(.99),stars.primary_half_strength.quantile(.99))*1.1
    axes[0,0].plot([0,limit],[0,limit],'--',color='.5');axes[0,0].set(xlim=(0,limit),ylim=(0,limit),xlabel='Frozen ensemble CV (%)',ylabel='After fixed catalog correction (%)',title='300 random stars; all retained sectors');axes[0,0].legend(title='Fixed half')
    axes[0,1].hexbin(100*f.centered_mismatch,100*f.log_residual,gridsize=55,mincnt=1,bins='log',cmap='viridis',extent=(-5,5,-5,5))
    axes[0,1].set(xlabel='Within-star catalog mismatch (%)',ylabel='Within-star STITCH log residual (%)',title='Signed mechanism check')
    for col,label in [('baseline','Frozen ensemble'),('primary_half_strength','Fixed half-strength scene')]:
        v=np.sort(stars[col]);axes[1,0].plot(v,np.arange(1,len(v)+1)/len(v),label=label)
    axes[1,0].set(xlim=(0,stars.baseline.quantile(.95)),xlabel='Per-star CV (%)',ylabel='Cumulative fraction of stars',title='Population distribution');axes[1,0].legend(fontsize=9)
    axes[1,1].scatter(stars.tmag,stars.baseline-stars.primary_half_strength,c=stars.crowdsap,s=22,cmap='viridis',vmin=.5,vmax=1)
    axes[1,1].axhline(0,color='.5');axes[1,1].set(xlabel='Tmag',ylabel='CV improvement (percentage points)',title='Failure check: magnitude and crowding')
    fig.savefig(OUT/'population_scene_result.png',dpi=180);plt.close(fig)
    # Largest improvements and harms, selected symmetrically and labeled.
    ranked=(stars.baseline-stars.primary_half_strength).sort_values();selected=list(ranked.head(4).index)+list(ranked.tail(4).index)
    fig,axes=plt.subplots(4,2,figsize=(14,13),constrained_layout=True)
    for ax,tic in zip(axes.flat,selected):
        g=f[f.tic_id==tic].sort_values('sector');b=100*(g.baseline_flux/g.baseline_flux.median()-1);a=100*(g.primary_flux/g.primary_flux.median()-1)
        ax.plot(g.sector,b,'o-',ms=3,label='Ensemble');ax.plot(g.sector,a,'o-',ms=3,label='+ fixed scene');ax.axhline(0,color='.7',lw=.7)
        ax.set(title=f'TIC {tic}: CV {stars.loc[tic,"baseline"]:.2f}% → {stars.loc[tic,"primary_half_strength"]:.2f}%',xlabel='Sector (lines cross gaps)',ylabel='Relative flux (%)')
    axes.flat[0].legend(fontsize=8);fig.savefig(OUT/'population_harm_and_gain_trajectories.png',dpi=180);plt.close(fig)
    (OUT/'RESULTS.md').write_text('# Random-population catalog scene test\n\n'+(OUT/'results.json').read_text()+'\n\n![Population result](population_scene_result.png)\n\n![Symmetric failure cases](population_harm_and_gain_trajectories.png)\n')
    log('Population scene evaluation complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','extract','summarize','all']);a=p.parse_args()
    warnings.filterwarnings('ignore',message=".*unitfix.*")
    if a.stage in ['parity','all']:parity()
    if a.stage in ['extract','all']:extract()
    if a.stage in ['summarize','all']:summarize()


if __name__=='__main__':main()
