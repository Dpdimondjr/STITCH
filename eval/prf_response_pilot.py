"""Bounded aperture/PRF pilot using official models and product WCS positions.

No science-target brightness is used in response features. MOM centroids are
diagnostics only. POS_CORR is centered about its retained-cadence median; the
motion features describe relative changes, not an absolute pointing solution.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

from background_bias_diagnostic import global_predict

CORE = ['sector','camera','ccd','tmag','crowdsap','flfrcsap','wcs_col','wcs_row',
        'phase_col','phase_row','ap_npix','cadence_days','data_rel',
        'pos_corr1_rms','pos_corr2_rms','pos_corr_major_rms','pos_corr_minor_rms']
PRF_COLS = ['prf_capture','prf_capture_discrepancy','prf_dlogq_dx','prf_dlogq_dy',
            'prf_motion_median','prf_motion_span','prf_motion_std','prf_blur_response']


def positions(header):
    """Use alternate physical WCS; do not assume CRPIX=1 or add 44 columns."""
    sky=WCS(header).celestial
    x,y=sky.all_world2pix(float(header['RA_OBJ']),float(header['DEC_OBJ']),0)
    col,row=WCS(header,key='P').all_pix2world(x,y,0)
    return float(x),float(y),float(col),float(row)


def response(prf, aperture, x, y, dx, dy):
    mask=np.asarray(aperture,bool)
    def capture(a,b):return float(prf.locate(a,b,mask.shape)[mask].sum())
    q=capture(x,y)
    if not np.isfinite(q) or not 0<q<=1.000001:raise ValueError('Invalid PRF capture')
    h=.05
    gx=(np.log(capture(x+h,y))-np.log(capture(x-h,y)))/(2*h)
    gy=(np.log(capture(x,y+h))-np.log(capture(x,y-h)))/(2*h)
    trajectory=np.array([capture(x+a,y+b)/q-1 for a,b in zip(dx,dy)])
    # Pad to avoid artificial loss at postage-stamp edge during convolution.
    pad=8;shape=(mask.shape[0]+2*pad,mask.shape[1]+2*pad)
    full=prf.locate(x+pad,y+pad,shape)
    blurred=gaussian_filter(full,.25,mode='constant')
    blur_q=blurred[pad:-pad,pad:-pad][mask].sum()
    return dict(prf_capture=q,prf_dlogq_dx=gx,prf_dlogq_dy=gy,
        prf_motion_median=float(np.median(trajectory)),
        prf_motion_span=float(np.ptp(trajectory)),prf_motion_std=float(np.std(trajectory)),
        prf_blur_response=float(blur_q/q-1))


def extract(record, root):
    import PRF
    out={'tic_id':int(record.tic_id),'sector':int(record.sector),'error':''}
    try:
        with fits.open(record.path,memmap=False) as h:
            tab,ap=h[1],h[2];mask=(ap.data.astype(int)&2)!=0
            x,y,col,row=positions(ap.header)
            if not(-.5<x<mask.shape[1]-.5 and -.5<y<mask.shape[0]-.5):
                raise ValueError('WCS target outside stamp')
            if int(mask.sum())!=int(ap.header['NPIXSAP']):raise ValueError('Aperture bitmask/header mismatch')
            q=np.asarray(tab.data['QUALITY'])
            valid=(q==0)&np.isfinite(tab.data['PDCSAP_FLUX'])&(tab.data['PDCSAP_FLUX']>0)
            dx=np.asarray(tab.data['POS_CORR1'],float)[valid]
            dy=np.asarray(tab.data['POS_CORR2'],float)[valid]
            ok=np.isfinite(dx)&np.isfinite(dy);dx,dy=dx[ok],dy[ok]
            if len(dx)<50:raise ValueError('Insufficient retained pointing cadences')
            dx-=np.median(dx);dy-=np.median(dy)
            sample=np.unique(np.linspace(0,len(dx)-1,25).astype(int))
            prf=PRF.TESS_PRF(int(record.camera),int(record.ccd),int(record.sector),
                             col,row,localdatadir=str(root/'start_s0004'))
            out.update(response(prf,mask,x,y,dx[sample],dy[sample]))
            out.update(wcs_col=col,wcs_row=row,phase_col=col%1,phase_row=row%1,
                wcs_mom_separation=float(np.hypot(col-record.mom_col,row-record.mom_row)),
                aperture_pixels=int(mask.sum()),motion_samples=len(sample),
                prf_capture_discrepancy=float(np.log(out['prf_capture']/record.flfrcsap)))
    except Exception as exc:out['error']=f'{type(exc).__name__}: {exc}'
    return out


def cv(frame, log_correction):
    z=frame[['tic_id','pdcsap_median']].copy()
    z['flux']=z.pdcsap_median.to_numpy()/np.exp(log_correction)
    g=z.groupby('tic_id').flux
    return g.std(ddof=0)/g.mean()


def benchmark(d,seed):
    ids=np.sort(d.tic_id.unique())
    tr,tmp=train_test_split(ids,test_size=.4,random_state=seed)
    va,te=train_test_split(tmp,test_size=.5,random_state=seed)
    train=d[d.tic_id.isin(tr)];val=d[d.tic_id.isin(va)];test=d[d.tic_id.isin(te)]
    per=pd.DataFrame({'raw':cv(test,np.zeros(len(test)))})
    strengths={}
    for name,cols in [('core',CORE),('core_prf',CORE+PRF_COLS)]:
        # One shared fit per feature block predicts val and test together.
        query=pd.concat([val,test],ignore_index=True)
        pred=global_predict(train,query,columns=cols)
        vp,tp=pred[:len(val)],pred[len(val):]
        candidates=[(float(cv(val,a*vp).median()),float(a)) for a in np.linspace(0,1.5,31)]
        _,alpha=min(candidates);strengths[name]=alpha
        per[name]=cv(test,alpha*tp)
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(per),size=(2000,len(per)))
    gain=100*(1-np.median(per.core_prf.to_numpy()[ix],axis=1)/np.median(per.core.to_numpy()[ix],axis=1))
    result={'seed':seed,'train_stars':len(tr),'validation_stars':len(va),'test_stars':len(te),
        'strengths':strengths,'median_cv_pct':{c:float(100*per[c].median()) for c in per},
        'prf_gain_over_core_pct':float(100*(1-per.core_prf.median()/per.core.median())),
        'paired_star_ci95':np.quantile(gain,[.025,.975]).tolist()}
    split=pd.concat([pd.DataFrame({'tic_id':a,'partition':b}) for a,b in [(tr,'train'),(va,'validation'),(te,'test')]])
    return result,per,split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='eval/prf_response_results')
    p.add_argument('--max-stars',type=int,default=300)
    p.add_argument('--prf-root',default='tess_cache/prf_models')
    p.add_argument('--reuse-features',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    try:import PRF
    except ImportError:
        sys.path.insert(0,str(Path('tess_cache/prf_vendor').resolve()));import PRF
    d=pd.read_parquet('eval/background_bias_results/diagnostic_rows.parquet')
    source=pd.read_parquet('eval/spoc_cohort_results/cohort_features.parquet')
    d=d.merge(source[['tic_id','sector','path']],on=['tic_id','sector'],validate='one_to_one')
    d=d[(d.camera==3)&d.ccd.isin([3,4])&(d.sector>=4)].copy()
    counts=d.groupby('tic_id').sector.nunique();ids=counts[counts>=5].index.to_numpy()
    ids=np.random.default_rng(20260928).choice(ids,min(a.max_stars,len(ids)),replace=False)
    d=d[d.tic_id.isin(ids)].sort_values(['tic_id','sector']).reset_index(drop=True)
    if a.reuse_features:
        features=pd.read_parquet(out/'response_features.parquet')
        assert set(zip(features.tic_id,features.sector))==set(zip(d.tic_id,d.sector))
    else:
        rows=[]
        for i,r in enumerate(d.itertuples(index=False)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');rows.append(extract(r,Path(a.prf_root)))
            if i%100==0:print('extracted',i,'/',len(d),flush=True)
        features=pd.DataFrame(rows);features.to_parquet(out/'response_features.parquet',index=False)
    good=features[features.error.eq('')].drop(columns='error')
    d=d.merge(good,on=['tic_id','sector'],validate='one_to_one')
    d=d[d.groupby('tic_id').sector.transform('nunique')>=5].reset_index(drop=True)
    if d.tic_id.nunique()<30:raise ValueError('Insufficient supported stars for pilot')
    catalog=pd.read_parquet('training_data_v3.parquet',columns=['tic_id','sector','col','row','cam','ccd'])
    catalog=catalog.rename(columns={'ccd':'catalog_ccd'})
    audit=d.merge(catalog,on=['tic_id','sector'],how='left',validate='one_to_one')
    delta=np.hypot(audit.wcs_col-audit.col,audit.wcs_row-audit.row)
    experiments=[]
    for seed in [20260928,20260929,20260930]:
        result,per,split=benchmark(d,seed);experiments.append(result)
        per.to_csv(out/f'per_star_{seed}.csv');split.to_csv(out/f'split_{seed}.csv',index=False)
        print(result,flush=True)
    manifests=sorted(Path(a.prf_root).glob('start_s0004/cam3_ccd[34]/manifest.json'))
    result={'selected_stars':len(ids),'attempted_rows':len(features),'successful_rows':len(good),
        'evaluated_stars':int(d.tic_id.nunique()),'evaluated_rows':len(d),
        'failures':features.loc[features.error.ne(''),'error'].value_counts().to_dict(),
        'prf_package':'TESS-PRF==0.1.3','prf_model_version':'UPDATED_2.0, sectors >=4',
        'model_manifest_sha256':{str(x.relative_to(Path(a.prf_root))):hashlib.sha256(x.read_bytes()).hexdigest() for x in manifests},
        'feature_source_sha256':hashlib.sha256(Path('eval/background_bias_results/diagnostic_rows.parquet').read_bytes()).hexdigest(),
        'coordinate_audit':{'wcs_mom_distance_quantiles_px':d.wcs_mom_separation.quantile([.5,.9,.99]).to_dict(),
            'wcs_v3_distance_quantiles_px':delta.quantile([.5,.9,.99]).to_dict(),
            'v3_matched_rows':int(delta.notna().sum()),
            'camera_ccd_disagreements':int(((audit.camera!=audit.cam)|(audit.ccd!=audit.catalog_ccd))[audit.cam.notna()].sum())},
        'capture_quantiles':d.prf_capture.quantile([.01,.5,.99]).to_dict(),
        'capture_vs_flfrcsap_correlation':float(d.prf_capture.corr(d.flfrcsap)),
        'experiments':experiments,'limitations':[
            'Camera 3 CCDs 3/4 only, bounded previously examined quiet-star cache cohort.',
            'No new KNN/NSF comparison; tests only the incremental PRF block on a matched global baseline.',
            'Product sky coordinates/WCS are used; no fresh Gaia epoch-propagation solution.',
            'PRF package normalizes finite 13x13 support; capture excludes unknown far wings.',
            'Motion sampled at 25 retained cadences, median-centered; no within-exposure jitter convolution.',
            'Additional 0.25-pixel blur is a response probe, not a measured focus correction.',
            'Bootstrap conditions on models; three splits reuse the same cohort.',
            'No independent background pixels or signal-preservation validation in this pilot.']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
