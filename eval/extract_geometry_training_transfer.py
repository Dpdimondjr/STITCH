"""Source geometry from compact FITS metadata; no measured target flux used."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u
from fit_scene_investigation import PRF,templates_at
from hgb_reference_repair import sha256,write_json,log
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from fetch_geometry_training_transfer import ROOT,compact_geometry

STRAPS=Path('eval/gaia_neighbor_variability_20261011/strap_geometry/handbook_strap_columns.csv')


def positions(table,tic,header,midtime):
    def num(c):return np.asarray(np.ma.filled(table[c],np.nan),float)
    ids=np.asarray(table['ID']).astype(str);flags=np.asarray(np.ma.filled(table['POSflag'],''),str)
    native=np.isin(flags,['gaia2','tmgaia2'])&np.isfinite(num('RA_orig'))&np.isfinite(num('Dec_orig'))
    ra=np.where(native,num('RA_orig'),num('ra'));dec=np.where(native,num('Dec_orig'),num('dec'))
    sky=SkyCoord(ra=ra*u.deg,dec=dec*u.deg,pm_ra_cosdec=np.nan_to_num(num('pmRA'))*u.mas/u.yr,
        pm_dec=np.nan_to_num(num('pmDEC'))*u.mas/u.yr,obstime=Time(np.where(native,2015.5,2000.),format='jyear'))
    moved=sky.apply_space_motion(Time(midtime+2457000,format='jd',scale='tdb'))
    x,y=WCS(header).celestial.all_world2pix(moved.ra.deg,moved.dec.deg,0)
    target=ids==str(tic);assert target.sum()==1
    disposition=np.asarray(np.ma.filled(table['disposition'],''),str);mag=num('Tmag')
    usable=(~np.isin(disposition,['DUPLICATE','ARTIFACT'])&np.isfinite(mag)&np.isfinite(x)&np.isfinite(y))|target
    return pd.DataFrame(dict(tic_id=ids,x=x,y=y,tmag=mag,gaiarp=num('gaiarp'),is_target=target,usable=usable,native_gaia=native))


def feature(primary,lc,header,bits,tic,sector,table,columns,midtime=None):
    assert int(primary['TICID'])==tic and int(primary['SECTOR'])==sector
    cam=int(primary['CAMERA']);ccd=int(primary['CCD'])
    nominal_c=float(lc['CROWDSAP']);nominal_q=float(lc['FLFRCSAP'])
    if nominal_c<=0 or nominal_q<=0:raise ValueError('Nonpositive header crowding/capture')
    midpoint=(float(lc['TSTART'])+float(lc['TSTOP']))/2
    if midtime is None:midtime=midpoint
    mask=((bits.astype(int)&2)!=0)&((bits.astype(int)&1)!=0);shape=bits.shape
    if not mask.any():raise ValueError('No aperture pixels')
    sky=WCS(header).celestial;physical=WCS(header,key='P')
    x,y=sky.all_world2pix(header['RA_OBJ'],header['DEC_OBJ'],0)
    col,row=physical.all_pix2world(x,y,0)
    yy,xx=np.indices(shape);pcol,_=physical.all_pix2world(xx,yy,0)
    strap=np.isin(np.rint(pcol-44).astype(int),columns)
    prf=PRF.TESS_PRF(cam,ccd,sector,float(np.clip(col,45.001,2091.999)),float(np.clip(row,1.001,2047.999)),
        localdatadir=str(Path('tess_cache/prf_models')/('start_s0001' if sector<4 else 'start_s0004')))
    cat=positions(table,tic,header,midtime);target=cat[cat.is_target].iloc[0]
    near=cat.usable&cat.x.between(-6,shape[1]+5)&cat.y.between(-6,shape[0]+5)
    cat=cat[near|cat.is_target].copy()
    templates=templates_at(prf,cat[['x','y']].to_numpy(),shape)
    target_template=templates[cat.is_target.to_numpy()][0]
    cat['stamp_capture']=templates.sum(axis=(1,2));cat['aperture_capture']=templates[:,mask].sum(axis=1)
    cat['strap_capture']=templates[:,mask&strap].sum(axis=1)
    cat['relative_flux']=10**(-.4*(cat.tmag-target.tmag))
    cat['expected_stamp_contribution']=cat.stamp_capture*cat.relative_flux
    cat['expected_aperture_contribution']=cat.aperture_capture*cat.relative_flux
    cat=cat.sort_values(['is_target','expected_stamp_contribution'],ascending=[False,False])
    keep=[];rejected=[]
    for i,s in cat.iterrows():
        if not s.is_target and s.expected_stamp_contribution<.001:rejected.append(i);continue
        if keep and np.min(np.linalg.norm(cat.loc[keep,['x','y']].to_numpy()-s[['x','y']].to_numpy(float),axis=1))<.15:rejected.append(i);continue
        if len(keep)>=18:rejected.append(i);continue
        keep.append(i)
    selected=cat.loc[keep];assert selected.iloc[0].is_target
    expected=selected.expected_aperture_contribution.to_numpy();capture=float(selected.iloc[0].aperture_capture)
    crowd=float(expected[0]/expected.sum())
    if not np.isfinite(capture+crowd) or min(capture,crowd)<=0:raise ValueError('Invalid scene capture/crowding')
    neighbor=selected[~selected.is_target]
    neighbor_aperture_ratio=float(neighbor.expected_aperture_contribution.sum()/capture)
    neighbor_strap_ratio=float((neighbor.strap_capture*neighbor.relative_flux).sum()/capture)
    return dict(actual_cam=cam,actual_ccd=ccd,actual_col=float(col),actual_row=float(row),
        midtime=float(midtime),header_midpoint=midpoint,capture_term=float(np.log(capture/nominal_q)),
        crowding_term=float(np.log(nominal_c/crowd)),prf_capture=capture,catalog_scene_crowding=crowd,
        nominal_capture=nominal_q,nominal_crowding=nominal_c,overlap_0=float(target_template[mask&strap].sum()/capture),
        neighbor_aperture_ratio=neighbor_aperture_ratio,neighbor_strap_ratio=neighbor_strap_ratio,
        area_fraction=float((mask&strap).sum()/mask.sum()),aperture_pixels=int(mask.sum()),sources=len(keep),
        omitted_expected_aperture_flux_relative_target=float(cat.loc[rejected,'expected_aperture_contribution'].sum()/expected[0]) if rejected else 0.,
        target_native_gaia=bool(target.native_gaia),target_header_position_difference=float(np.hypot(target.x-x,target.y-y)),
        procver=str(primary.get('PROCVER','')),data_rel=int(primary.get('DATA_REL',-1)),
        prf_grid_clamp_pixels=float(np.hypot(np.clip(col,45.001,2091.999)-col,np.clip(row,1.001,2047.999)-row)))


def worker(task):
    tic,rows,columns=task;path=ROOT/'geometry_by_star'/f'tic_{tic}.parquet'
    if path.exists():return
    try:table=Table.read(ROOT/'catalogs'/f'tic_{tic}.ecsv');catalog_error=None
    except Exception as exc:table=None;catalog_error=str(exc)
    output=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for r in rows:
            result=dict(source_row=r['source_row'],tic_id=tic,sector=r['sector'],geometry_available=False)
            try:
                if table is None:raise ValueError('Catalog unavailable: '+catalog_error)
                if not r['available']:raise ValueError(r.get('error','Geometry unavailable'))
                with np.load(r['compact_path']) as z:
                    primary=fits.Header.fromstring(str(z['primary']),sep='');lc=fits.Header.fromstring(str(z['lightcurve']),sep='');header=fits.Header.fromstring(str(z['aperture_header']),sep='')
                    result.update(feature(primary,lc,header,z['aperture'],tic,r['sector'],table,columns),geometry_available=True)
            except Exception as exc:result['error']=str(exc)
            output.append(result)
    temp=path.with_suffix(f'.{os.getpid()}.tmp')
    pd.DataFrame(output).to_parquet(temp,index=False);temp.replace(path)


def parity():
    from evaluate_population_scene import metadata_feature
    from check_native_gaia_geometry import native_positions
    source=Path('eval/gaia_strap_multicamera_20261011')
    products=pd.read_csv(source/'products.csv').sort_values(['tic_id','sector'])
    selected=products.iloc[::max(1,len(products)//16)].head(16)
    columns=pd.read_csv(STRAPS).imaging_column.to_numpy();checks=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for r in selected.to_dict('records'):
            path=compact_geometry(r)
            with np.load(path) as z,fits.open(r['local_path']) as h:
                primary=fits.Header.fromstring(str(z['primary']),sep='');lc=fits.Header.fromstring(str(z['lightcurve']),sep='');header=fits.Header.fromstring(str(z['aperture_header']),sep='')
                assert primary.tostring()==h[0].header.tostring()
                assert lc.tostring()==h[1].header.tostring()
                assert header.tostring()==h[2].header.tostring()
                np.testing.assert_array_equal(z['aperture'],h[2].data)
                d=h[1].data;mid=float(np.median(d['TIME'][(d['QUALITY']==0)&np.isfinite(d['TIME'])]))
                table=Table.read(source/'catalogs'/f"tic_{r['tic_id']}.ecsv")
                exact=feature(primary,lc,header,z['aperture'],r['tic_id'],r['sector'],table,columns,mid)
                operational=feature(primary,lc,header,z['aperture'],r['tic_id'],r['sector'],table,columns)
            old=metadata_feature(r['local_path'],r['tic_id'],r['sector'],source/'catalogs',position_function=native_positions)
            keys=['capture_term','crowding_term','prf_capture','catalog_scene_crowding']
            np.testing.assert_allclose([exact[k] for k in keys],[old[k] for k in keys],atol=1e-12)
            checks.append(dict(tic_id=r['tic_id'],sector=r['sector'],midtime_delta_days=operational['midtime']-mid,
                max_feature_difference_same_time=max(abs(exact[k]-old[k]) for k in keys),
                capture_time_difference=operational['capture_term']-exact['capture_term'],crowding_time_difference=operational['crowding_term']-exact['crowding_term'],
                overlap_time_difference=operational['overlap_0']-exact['overlap_0']))
    pd.DataFrame(checks).to_csv(ROOT/'geometry_parity.csv',index=False)
    write_json(ROOT/'geometry_parity.json',dict(status='passed',remote_local_observations=len(checks),all_headers_masks_equal=True,
        max_implementation_difference=max(r['max_feature_difference_same_time'] for r in checks),
        max_header_midpoint_log_feature_difference=max(max(abs(r['capture_time_difference']),abs(r['crowding_time_difference'])) for r in checks),
        limitation='Midpoint check is a sample, not proof of negligible motion for every star; retain motion/provenance diagnostics.'))


def extract():
    (ROOT/'geometry_by_star').mkdir(exist_ok=True)
    while (ROOT/'stream_started.json').exists() and not (ROOT/'stream_complete.json').exists():time.sleep(5)
    f=pd.read_parquet(ROOT/'geometry_inventory.parquet')
    columns=pd.read_csv(STRAPS).imaging_column.to_numpy()
    tasks=[(int(tic),g.to_dict('records'),columns) for tic,g in f.groupby('tic_id')]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for i,fu in enumerate(as_completed(futures),1):
            fu.result()
            if i%50==0 or i==len(futures):log(f'Geometry extraction {i}/{len(futures)} TICs')
    result=pd.concat([pd.read_parquet(ROOT/'geometry_by_star'/f'tic_{tic}.parquet') for tic,_,_ in tasks],ignore_index=True)
    result.to_parquet(ROOT/'geometry_features.parquet',index=False)
    write_json(ROOT/'geometry_complete.json',dict(script_sha256=sha256(__file__),observations=len(result),available=int(result.geometry_available.sum()),
        geometry_sha256=sha256(ROOT/'geometry_features.parquet'),source_selection='Existing Tmag catalog-scene rule; Gaia RP remains the training target.'))


def stream():
    """Extract whole TICs once their metadata and catalog have arrived."""
    (ROOT/'geometry_by_star').mkdir(exist_ok=True)
    write_json(ROOT/'stream_started.json',dict(pid=os.getpid(),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    f=pd.read_parquet(ROOT/'products.parquet')
    columns=pd.read_csv(STRAPS).imaging_column.to_numpy()
    from fetch_geometry_training_transfer import CACHE
    tasks={int(tic):g.to_dict('records') for tic,g in f.groupby('tic_id')}
    inflight={};completed=0
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            while tasks or inflight:
                for tic in list(tasks):
                    if (ROOT/'geometry_by_star'/f'tic_{tic}.parquet').exists():tasks.pop(tic);continue
                    if len(inflight)>=4:break
                    if not (ROOT/'catalogs'/f'tic_{tic}.ecsv').exists():continue
                    rows=tasks[tic]
                    if any(pd.isna(r['version']) for r in rows):continue
                    paths=[CACHE/f"{tic}_{int(r['sector']):04d}_v{int(r['version'])}.npz" for r in rows]
                    if not all(p.exists() for p in paths):continue
                    ready=[{**r,'available':True,'compact_path':str(p)} for r,p in zip(rows,paths)]
                    fu=pool.submit(worker,(tic,ready,columns));inflight[fu]=tic;tasks.pop(tic)
                if inflight:
                    done,_=wait(inflight,timeout=5,return_when=FIRST_COMPLETED)
                    for fu in done:
                        fu.result();inflight.pop(fu);completed+=1
                        if completed%50==0:log(f'Streamed geometry {completed} TICs')
                else:
                    if (ROOT/'collection_complete.json').exists():break
                    time.sleep(5)
    finally:
        write_json(ROOT/'stream_complete.json',dict(completed_tics=completed,remaining_tics=len(tasks),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','extract','stream']);a=p.parse_args()
    {'parity':parity,'extract':extract,'stream':stream}[a.stage]()
