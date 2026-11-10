"""Frozen training/evaluation cohorts and compact, exact FITS geometry retrieval."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import threading
import time
import warnings
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.table import Table
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'eval'))
from hgb_reference_repair import sha256, write_json, log

ROOT=Path('eval/geometry_training_transfer_20261012')
CACHE=Path('tess_cache/geometry_training_transfer_20261012')
SOURCE=Path('eval/hgb_reference_repair_20261010/clean_features.parquet')
BANK=Path('eval/calibration_bank_study_20261011/r0_half_A')
SEED=202610129
LOCAL=threading.local()


class TimedSession(requests.Session):
    def request(self,*args,**kwargs):
        kwargs['timeout']=kwargs.get('timeout') or (15,90)
        return super().request(*args,**kwargs)


def mast_client(kind):
    if not hasattr(LOCAL,kind):
        from astroquery.mast import ObservationsClass,CatalogsClass
        client=(ObservationsClass if kind=='observations' else CatalogsClass)()
        for name in ['_portal_api_connection','_service_api_connection']:
            connection=getattr(client,name,None)
            if connection is not None:
                connection.TIMEOUT=90
                connection._session=TimedSession()
        setattr(LOCAL,kind,client)
    return getattr(LOCAL,kind)


def session():
    if not hasattr(LOCAL,'session'): LOCAL.session=requests.Session()
    return LOCAL.session


def prepare():
    ROOT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
    if (ROOT/'freeze.json').exists(): return
    f=pd.read_parquet(SOURCE,columns=['source_row','tic_id','sector','role','cam','ccd','col','row','tmag','crowdsap','ra','dec'])
    banks=pd.read_parquet('eval/calibration_bank_study_20261011/training_banks.parquet')
    b=np.sort(banks[(banks.replicate==0)&(banks.bank=='B')].tic_id.unique())
    cached_ids=set()
    for path in Path('tess_cache').rglob('*_lc.fits'):
        m=re.fullmatch(r'hlsp_tess-spoc_tess_phot_(\d+)-s(\d+)_tess_v(\d+)_lc.fits',path.name)
        if m: cached_ids.add(int(m[1]))
    v=np.sort(f[f.role.eq('validation')&~f.tic_id.isin(cached_ids)].tic_id.unique())
    rng=np.random.default_rng(SEED)
    train=rng.choice(b,600,replace=False);evaluation=rng.choice(v,400,replace=False)
    stars=pd.DataFrame(dict(tic_id=np.r_[train,evaluation],study_role=['calibration']*600+['evaluation']*400))
    stars['head_fold']=-1;stars.loc[:599,'head_fold']=np.arange(600)%3
    sample=f.merge(stars,on='tic_id',validate='many_to_one')
    assert not set(train)&set(evaluation)
    assert set(train)<=set(f[f.role.eq('train')].tic_id)
    sample.to_parquet(ROOT/'sample.parquet',index=False);stars.to_parquet(ROOT/'stars.parquet',index=False)
    write_json(ROOT/'freeze.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),seed=SEED,
        source_sha256=sha256(SOURCE),sample_sha256=sha256(ROOT/'sample.parquet'),script_sha256=sha256(__file__),
        eligibility=dict(calibration_bank_B=len(b),evaluation_uncached_validation=len(v)),
        cohorts={k:dict(stars=int(g.tic_id.nunique()),rows=len(g)) for k,g in sample.groupby('study_role')},
        reference_bank=str(BANK),excluded_from_baseline='Every calibration TIC belongs to disjoint bank B; baseline and color transfer use bank A only.',
        selection_uses_flux_or_residuals=False,retain_all_sectors=True,
        geometry='Matching LC primary/lightcurve/aperture headers and aperture mask via verified HTTP ranges; native Gaia astrometry at header midpoint; PRF interpolated at FITS position.',
        target='Absolute Gaia RP anchor, retained at every core round and in physical-head fitting; no within-TIC target centering.',
        candidates=['scene','scene_strap','geometry_context'],head_alphas=[.01,.1,1.],head_log_clip=.03,
        selection='Three-fold whole-TIC CV among qualified calibration rows only; freeze all fitted heads before evaluation outcomes.',
        baseline=dict(rounds=5,max_leaf_nodes=1023,max_iter=250,seed=20261012),
        missing_geometry='Zero extra correction, retain observations; report attrition and missing anchors.',
        limit='Evaluation is a random subset of historically examined validation stars without cached LC geometry at selection; not pristine or fully population representative. No test labels used.'))
    (ROOT/'PROTOCOL.md').write_text('''# Training-star geometry transfer

Freeze 600 random original training TICs from bank B and 400 random validation
TICs without cached LC geometry, seed 202610129. Selection uses no flux outcome.
Keep every retained sector and do not replace failures. The baseline uses only
disjoint bank A. Refit its absolute Gaia core with fully nested training-only
color calibration; predict both the geometry-calibration and evaluation stars.
All KNN references exclude query, validation and test TIC labels. Core parameters
are fixed before fitting. No calibration TIC enters baseline fitting.

Retrieve exact primary/lightcurve/aperture headers and aperture masks using HTTP
byte ranges. Preserve product URI/version, identity, raw header text and transport
hashes. Use actual FITS camera/CCD/WCS and native Gaia coordinates propagated to
the header observation midpoint. Compare midpoint geometry with cached full-LC
geometry before evaluation. Catalog scene uses the existing source-selection
rule; no target flux is fit or normalized across sectors. Gaia RP remains the
learning target; catalog Tmag supplies the existing scene contrast model.

Fit qualified calibration rows using equal TIC weights. Compare scene, scene plus
PRF-weighted nominal straps, and geometry-context interactions. Three-fold
whole-TIC calibration CV chooses among Huber alpha 0.01, 0.1, 1.0 and these
families. Center predictors, never Gaia targets. Bound extra log corrections at
0.03. Freeze heads before joining evaluation flux outcomes. Missing geometry or
anchors retain zero extra correction. Keep all rows for matched all-sector and
individually TARS-qualified views, with attrition and paired TIC intervals.

The evaluation population is new for detailed geometry but belongs to the
historically examined validation role. It excludes cached-geometry TICs and is
not a pristine or full-population benchmark. No new test score, automatic default
promotion, or irreducible noise-floor claim. Fixed-feature injection checks do
not certify upstream PDC or extraction response.
''')
    log(f'Frozen {sample.tic_id.nunique()} stars / {len(sample)} observations')


def inventory_batch(task):
    start,ids,keys=task;path=ROOT/'inventory'/f'batch_{start:04d}.parquet'
    if path.exists(): return pd.read_parquet(path)
    Observations=mast_client('observations')
    for attempt in range(3):
        try:
            obs=Observations.query_criteria(provenance_name='TESS-SPOC',target_name=[str(t) for t in ids])
            obs=obs[[(int(r['target_name']),int(r['sequence_number'])) in keys for r in obs]]
            products=Observations.get_product_list(obs);rows=[]
            for r in products:
                name=str(r['productFilename']);m=re.fullmatch(r'hlsp_tess-spoc_tess_phot_(\d+)-s(\d+)_tess_v(\d+)_lc.fits',name)
                if not m or (int(m[1]),int(m[2])) not in keys:continue
                rows.append(dict(tic_id=int(m[1]),sector=int(m[2]),version=int(m[3]),filename=name,data_uri=str(r['dataURI']),size_bytes=int(r['size'])))
            f=pd.DataFrame(rows).sort_values(['tic_id','sector','version','filename']).drop_duplicates(['tic_id','sector'],keep='last')
            f.to_parquet(path,index=False);return f
        except Exception:
            if attempt==2:raise
            time.sleep(3*(attempt+1))


def inventory():
    if (ROOT/'inventory_complete.json').exists():return
    sample=pd.read_parquet(ROOT/'sample.parquet');keys=set(zip(sample.tic_id,sample.sector));ids=sorted(sample.tic_id.unique())
    (ROOT/'inventory').mkdir(exist_ok=True)
    tasks=[(start,ids[start:start+25],keys) for start in range(0,len(ids),25)]
    frames=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,f in enumerate(pool.map(inventory_batch,tasks),1):
            frames.append(f);log(f'Inventory {i}/{len(tasks)} batches')
    products=pd.concat(frames,ignore_index=True)
    f=sample.merge(products,on=['tic_id','sector'],how='left',validate='one_to_one')
    f.to_parquet(ROOT/'products.parquet',index=False)
    write_json(ROOT/'inventory_complete.json',dict(rows=len(f),products=int(f.data_uri.notna().sum()),missing=int(f.data_uri.isna().sum()),products_sha256=sha256(ROOT/'products.parquet')))


def header_from_bytes(data,offset=0):
    for end in range(offset,len(data)-79,80):
        if data[end:end+8]==b'END     ':
            span=((end+80-offset+2879)//2880)*2880
            if offset+span>len(data):raise ValueError('Incomplete padded header')
            return fits.Header.fromstring(data[offset:offset+span].decode('ascii'),sep=''),span
    raise ValueError('Missing complete FITS header')


def byte_range(uri,start,end):
    # The official public cloud mirror serves the identical versioned products.
    suffix=uri.split('tess-spoc/',1)[1]
    url='https://stpubdata.s3.amazonaws.com/mast/hlsp/tess-spoc/'+suffix
    response=session().get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,40),stream=True)
    if response.status_code==404:
        response.close()
        response=session().get('https://mast.stsci.edu/api/v0.1/Download/file',params={'uri':uri},headers={'Range':f'bytes={start}-{end}'},timeout=(15,40),stream=True)
    with response:
        if response.status_code!=206:raise ValueError(f'Expected partial response, got {response.status_code}')
        cr=response.headers['Content-Range'];m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
        if not m or int(m[1])!=start:raise ValueError(f'Wrong byte range {cr}')
        data=response.content
        if len(data)!=int(m[2])-start+1:raise ValueError('Partial byte range length mismatch')
        return data,int(m[3]),response.headers.get('ETag','')


def compact_geometry(rec):
    path=CACHE/f"{int(rec['tic_id'])}_{int(rec['sector']):04d}_v{int(rec['version'])}.npz"
    if path.exists():
        try:
            with np.load(path) as saved:
                assert str(saved['data_uri'])==rec['data_uri']
                assert saved['aperture'].ndim==2
            return str(path)
        except Exception:
            path.rename(path.with_suffix('.invalid.npz'))
    first,total,etag=byte_range(rec['data_uri'],0,28799)
    h0,n0=header_from_bytes(first);h1,n1=header_from_bytes(first,n0)
    assert int(h0['TICID'])==int(rec['tic_id']) and int(h0['SECTOR'])==int(rec['sector'])
    assert h1['XTENSION'].strip()=='BINTABLE'
    data_bytes=h1['NAXIS1']*h1['NAXIS2']+h1.get('PCOUNT',0)
    aperture_offset=n0+n1+((data_bytes+2879)//2880)*2880
    last,total2,etag2=byte_range(rec['data_uri'],aperture_offset,min(total-1,aperture_offset+57599))
    assert total==total2 and etag==etag2
    h2,n2=header_from_bytes(last)
    if h2.get('EXTNAME')!='APERTURE':raise ValueError('Unexpected extension after light curve')
    aperture=fits.ImageHDU.fromstring(last).data.copy()
    if aperture is None or aperture.ndim!=2:raise ValueError('Missing aperture mask')
    temp=path.with_suffix('.part.npz')
    np.savez_compressed(temp,primary=h0.tostring(),lightcurve=h1.tostring(),aperture_header=h2.tostring(),aperture=aperture,
        data_uri=rec['data_uri'],filename=rec['filename'],etag=etag,size_bytes=total,
        prefix_sha256=hashlib.sha256(first).hexdigest(),aperture_sha256=hashlib.sha256(last).hexdigest(),
        bytes_transferred=len(first)+len(last))
    temp.replace(path)
    return str(path)


def collect_one(rec):
    out=dict(source_row=int(rec['source_row']),tic_id=int(rec['tic_id']),sector=int(rec['sector']),available=False)
    if pd.isna(rec['data_uri']):return {**out,'error':'No matching LC product'}
    for attempt in range(3):
        try:
            path=compact_geometry(rec);return {**out,'available':True,'compact_path':path}
        except Exception as exc:
            if attempt==2:return {**out,'error':str(exc)}
            time.sleep(2*(attempt+1))


def cone(rec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    Catalogs=mast_client('catalogs')
    path=ROOT/'catalogs'/f"tic_{int(rec['tic_id'])}.ecsv"
    if path.exists():return dict(tic_id=int(rec['tic_id']),available=True)
    for attempt in range(3):
        try:
            t=Catalogs.query_region(SkyCoord(rec['ra'],rec['dec'],unit='deg'),radius=240*u.arcsec,catalog='Tic')
            assert str(int(rec['tic_id'])) in np.asarray(t['ID']).astype(str)
            temp=path.with_suffix('.tmp');t.write(temp,format='ascii.ecsv',overwrite=True);temp.replace(path)
            return dict(tic_id=int(rec['tic_id']),available=True,sources=len(t))
        except Exception as exc:
            if attempt==2:return dict(tic_id=int(rec['tic_id']),available=False,error=str(exc))
            time.sleep(2*(attempt+1))


def collect():
    (ROOT/'catalogs').mkdir(exist_ok=True)
    products=pd.read_parquet(ROOT/'products.parquet');outputs=[]
    # Independent catalog requests run beside geometry range reads.
    stars=products.groupby('tic_id').first().reset_index().to_dict('records')
    with ThreadPoolExecutor(max_workers=4) as catalogs, ThreadPoolExecutor(max_workers=16) as geometry:
        cf=[catalogs.submit(cone,r) for r in stars]
        futures=[geometry.submit(collect_one,r) for r in products.to_dict('records')]
        for i,fu in enumerate(as_completed(futures),1):
            outputs.append(fu.result())
            if i%200==0 or i==len(futures):
                pd.DataFrame(outputs).to_parquet(ROOT/'download_progress.parquet',index=False)
                log(f'Compact geometry {i}/{len(futures)}; available={sum(r["available"] for r in outputs)}')
        catalog_outputs=[fu.result() for fu in cf]
    pd.DataFrame(outputs).to_parquet(ROOT/'geometry_inventory.parquet',index=False)
    pd.DataFrame(catalog_outputs).to_parquet(ROOT/'catalog_inventory.parquet',index=False)
    write_json(ROOT/'collection_complete.json',dict(observations=len(outputs),geometry_available=sum(r['available'] for r in outputs),catalogs_available=sum(r['available'] for r in catalog_outputs),
        cache_bytes=sum(p.stat().st_size for p in CACHE.glob('*.npz'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','inventory','collect','all']);a=p.parse_args()
    for name,fn in [('prepare',prepare),('inventory',inventory),('collect',collect)]:
        if a.stage in [name,'all']:fn()
