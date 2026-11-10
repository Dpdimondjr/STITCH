"""Sensitivity to the TIC-recommended native Gaia DR2 astrometric epoch."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
import fit_scene_investigation as scene
from evaluate_population_scene import cv
from hgb_reference_repair import paired_bootstrap
from investigate_residual_atlas import save_json,log

ROOT=Path('eval/residual_investigation_20261011/population_scene')
OUT=ROOT/'native_gaia_geometry'


def native_positions(path,tic,header,midtime):
    old=scene.catalog_positions(path,tic,header,midtime);t=Table.read(path)
    def num(c):return np.asarray(np.ma.filled(t[c],np.nan),float)
    flags=np.asarray(np.ma.filled(t['POSflag'],''),str)
    use=np.isin(flags,['gaia2','tmgaia2'])&np.isfinite(num('RA_orig'))&np.isfinite(num('Dec_orig'))
    ra=np.where(use,num('RA_orig'),num('ra'));dec=np.where(use,num('Dec_orig'),num('dec'))
    epoch=np.where(use,2015.5,2000.)
    sky=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,pm_ra_cosdec=np.nan_to_num(num('pmRA'))*u.mas/u.yr,
        pm_dec=np.nan_to_num(num('pmDEC'))*u.mas/u.yr,obstime=Time(epoch,format='jyear'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');new=sky.apply_space_motion(Time(midtime+2457000,format='jd',scale='tdb'))
    x,y=WCS(header).celestial.all_world2pix(new.ra.deg,new.dec.deg,0)
    old['native_position_difference_pixels']=np.hypot(x-old.x,y-old.y)
    old['native_gaia_available']=use;old['x']=x;old['y']=y
    return old


def feature(r):
    with fits.open(r['local_path'],memmap=True) as h:
        header=h[2].header;bits=h[2].data;shape=bits.shape
        mask=((bits.astype(int)&2)!=0)&((bits.astype(int)&1)!=0)
    prf=scene.PRF.TESS_PRF(int(r['actual_cam']),int(r['actual_ccd']),int(r['sector']),
        float(np.clip(r['col'],45.001,2091.999)),float(np.clip(r['row'],1.001,2047.999)),
        localdatadir=str(Path('tess_cache/prf_models')/('start_s0001' if r['sector']<4 else 'start_s0004')))
    cat=native_positions(ROOT/'catalogs'/f'tic_{r["tic_id"]}.ecsv',r['tic_id'],header,r['midtime'])
    max_difference=cat.native_position_difference_pixels.max();target=cat[cat.is_target].iloc[0]
    near=cat.usable&cat.x.between(-6,shape[1]+5)&cat.y.between(-6,shape[0]+5);cat=cat[near|cat.is_target].copy()
    templates=scene.templates_at(prf,cat[['x','y']].to_numpy(),shape)
    cat['stamp_capture']=templates.sum(axis=(1,2));cat['aperture_capture']=templates[:,mask].sum(axis=1)
    cat['relative_flux']=10**(-.4*(cat.tmag-target.tmag))
    cat['expected_stamp_contribution']=cat.stamp_capture*cat.relative_flux
    cat['expected_aperture_contribution']=cat.aperture_capture*cat.relative_flux
    cat=cat.sort_values(['is_target','expected_stamp_contribution'],ascending=[False,False]);keep=[]
    for i,s in cat.iterrows():
        if not s.is_target and s.expected_stamp_contribution<.001:continue
        if keep and np.min(np.linalg.norm(cat.loc[keep,['x','y']].to_numpy()-s[['x','y']].to_numpy(float),axis=1))<.15:continue
        if len(keep)>=18:continue
        keep.append(i)
    selected=cat.loc[keep];expected=selected.expected_aperture_contribution.to_numpy()
    crowd=expected[0]/expected.sum();capture=selected.iloc[0].aperture_capture
    mismatch=np.log(capture/r['nominal_capture'])+np.log(r['nominal_crowding']/crowd)
    return {'tic_id':r['tic_id'],'sector':r['sector'],'native_log_mismatch':mismatch,
        'max_cone_position_difference_pixels':float(max_difference),
        'target_position_difference_pixels':float(target.native_position_difference_pixels),
        'target_native_gaia_available':bool(target.native_gaia_available),'sources':len(keep)}


def star(task):
    tic,rows=task;path=OUT/'by_star'/f'tic_{tic}.parquet'
    if path.exists():return
    output=[]
    for r in rows:
        try:output.append(feature(r))
        except Exception as exc:output.append({'tic_id':tic,'sector':r['sector'],'error':str(exc)})
    pd.DataFrame(output).to_parquet(path,index=False)


def main():
    (OUT/'by_star').mkdir(parents=True,exist_ok=True)
    (OUT/'PROTOCOL.md').write_text('''# Native Gaia position sensitivity

MAST TIC v8.2 notes recommend RA_orig/Dec_orig from the source catalog rather
than forward propagation of the TIC J2000 coordinates. For POSflag gaia2 or
tmgaia2, use original coordinates at Gaia DR2 epoch J2015.5; retain and explicitly
flag the legacy fallback for other position sources. Recompute the same frozen
catalog/Tmag/PRF feature for all 300 random stars and every retained sector.
This sensitivity check does not replace or retune the frozen primary result.

Sources: https://outerspace.stsci.edu/spaces/TESS/pages/111023212/TIC%2Bv8.2%2Band%2BCTL%2Bv8.xx%2BData%2BRelease%2BNotes
and https://www.cosmos.esa.int/web/gaia/dr2 .
''')
    obs=pd.read_parquet(ROOT/'observations.parquet');products=pd.read_csv(ROOT/'products.csv')
    f=obs.merge(products[['tic_id','sector','local_path']],on=['tic_id','sector'],how='left',validate='one_to_one')
    tasks=[(int(tic),g.to_dict('records')) for tic,g in f.groupby('tic_id')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(star,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%50==0:log(f'Native Gaia geometry {i}/300 stars')
    m=pd.concat([pd.read_parquet(OUT/'by_star'/f'tic_{tic}.parquet') for tic,_ in tasks],ignore_index=True)
    m.to_parquet(OUT/'measurements.parquet',index=False);f=f.merge(m,on=['tic_id','sector'],validate='one_to_one',suffixes=('','_native'))
    baseline=cv(f,f.baseline_flux);primary=cv(f,f.primary_flux)
    native=cv(f,f.baseline_flux/np.exp(.5*f.native_log_mismatch.fillna(f.predicted_log_mismatch).fillna(0)))
    diff=f.native_log_mismatch-f.predicted_log_mismatch
    pd.DataFrame({'baseline':baseline,'primary':primary,'native_geometry':native}).to_csv(OUT/'per_star_cv.csv')
    result={'observations':len(f),'available_native_features':int(f.native_log_mismatch.notna().sum()),
        'target_native_gaia_available_rows':int(f.target_native_gaia_available.fillna(False).sum()),
        'max_position_difference_pixels':float(f.max_cone_position_difference_pixels.max()),
        'max_absolute_feature_difference':float(diff.abs().max()),
        'feature_difference_percent_quantiles':(100*diff.abs()).quantile([.5,.95,.99,1]).to_dict(),
        'source_count_changed_rows':int((f.sources!=f.sources_native).sum()),
        'primary_cv_pct':float(primary.median()),'native_cv_pct':float(native.median()),
        'native_vs_baseline':paired_bootstrap(baseline,native,20261016),
        'native_vs_primary':paired_bootstrap(primary,native,20261016),
        'limitation':'Non-Gaia2 position sources retain the explicitly flagged legacy fallback; native Gaia coordinates are recommended for subsequent implementations.'}
    save_json(OUT/'results.json',result);log('Native Gaia geometry sensitivity complete')


if __name__=='__main__':main()
