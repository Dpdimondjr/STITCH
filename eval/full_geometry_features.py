"""Extract the verified geometry definition from lossless full-population batches."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/stitch_mpl')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from glob import glob
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from extract_geometry_training_transfer import feature, STRAPS, PRF
from hgb_reference_repair import write_json, sha256, log
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data'))
from collect_full_geometry import ROOT, STUDY, atomic_parquet, space_check, committed_batches

OriginalPRF=PRF.TESS_PRF


@lru_cache(maxsize=32)
def grid(localdatadir,cam,ccd):
    paths=[p for p in glob(os.path.join(localdatadir,f'cam{cam}_ccd{ccd}','*.fits')) if 'phot' not in p]
    return paths,np.array([int(p[-9:-5]) for p in paths]),np.array([int(p[-17:-13]) for p in paths])


@lru_cache(maxsize=128)
def tile(path):
    with fits.open(path,memmap=False) as h:return h[0].data.copy()


class CachedPRF(OriginalPRF):
    """Identical arithmetic and vendor locate(), caching only immutable FITS grids."""
    def __init__(self,cam,ccd,sector,colnum,rownum,localdatadir=None):
        self.cam,self.ccd,self.sector,self.colnum,self.rownum=cam,ccd,sector,colnum,rownum
        self.prfnsamp=9
        paths,cols,rows=grid(localdatadir,int(cam),int(ccd))
        subsets=[np.where(mask)[0] for mask in [(rows<rownum)&(cols<colnum),(rows>rownum)&(cols<colnum),
                                               (rows<rownum)&(cols>colnum),(rows>rownum)&(cols>colnum)]]
        dist=np.sqrt((rows-rownum)**2.+(cols-colnum)**2.)
        inds=[s[np.argmin(dist[s])] for s in subsets]
        points=sorted([(cols[i],rows[i],tile(paths[i])) for i in inds])
        (x1,y1,q11),(_x1,y2,q12),(x2,_y1,q21),(_x2,_y2,q22)=points
        self.prf=(q11*(x2-colnum)*(y2-rownum)+q21*(colnum-x1)*(y2-rownum)+
                  q12*(x2-colnum)*(rownum-y1)+q22*(colnum-x1)*(rownum-y1))/((x2-x1)*(y2-y1)+0.)
        reshaped=np.zeros((11,11,13,13))
        for i in range(9):
            for j in range(9):reshaped[i+1,j+1,:,:]=self.prf[8-i::9,8-j::9]
        for j in range(1,10):
            reshaped[j,0,:,:]=np.append(reshaped[j,-2,:,1:],np.zeros((13,1)),axis=1)
            reshaped[j,-1,:,:]=np.append(np.zeros((13,1)),reshaped[j,1,:,:-1],axis=1)
        for i in range(11):
            reshaped[0,i,:,:]=np.append(reshaped[-2,i,1:,:],np.zeros((1,13)),axis=0)
            reshaped[-1,i,:,:]=np.append(np.zeros((1,13)),reshaped[1,i,:-1,:],axis=0)
        self.reshaped=reshaped


def unpack(rec):
    h=[fits.Header.fromstring(rec[k],sep='') for k in ['primary','lightcurve','aperture_header']]
    mask=np.frombuffer(rec['aperture_bytes'],dtype=rec['aperture_dtype']).reshape(int(rec['aperture_height']),int(rec['aperture_width']))
    return h,mask


def header_features(h):
    h0,h1,h2=h;out={}
    names={'CROWDSAP':'header_crowdsap','FLFRCSAP':'header_flfrcsap','PDCVAR':'header_pdcvar',
           'PR_WGHT2':'header_pr_wght2','PDC_NOI':'header_pdc_noi','PDC_CORP':'header_pdc_corp',
           'PDC_TOTP':'header_pdc_totp','CDPP0_5':'header_cdpp0_5','CDPP1_0':'header_cdpp1_0',
           'CDPP2_0':'header_cdpp2_0','PDC_TOT':'header_pdc_tot','PDC_COR':'header_pdc_cor'}
    for key,name in names.items():
        try:out[name]=float(h1[key])
        except (TypeError,ValueError,KeyError):out[name]=np.nan
    try:out['header_teff']=float(h0['TEFF'])
    except (TypeError,ValueError,KeyError):out['header_teff']=np.nan
    out['pipeline_method']=str(h1.get('PDCMETHD',''))
    out['procver']=str(h0.get('PROCVER',''));out['data_rel']=int(h0.get('DATA_REL',-1))
    return out


def parity():
    sample=pd.read_parquet(ROOT/'lossless_storage_sample.parquet').sample(24,random_state=812)
    columns=pd.read_csv(STRAPS).imaging_column.to_numpy();checks=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for r in sample.to_dict('records'):
            h,mask=unpack(r);t=Table.read(STUDY/'catalogs'/f"tic_{r['tic_id']}.ecsv")
            PRF.TESS_PRF=OriginalPRF;start=time.monotonic();a=feature(*h,mask,r['tic_id'],r['sector'],t,columns);ta=time.monotonic()-start
            PRF.TESS_PRF=CachedPRF;start=time.monotonic();b=feature(*h,mask,r['tic_id'],r['sector'],t,columns);tb=time.monotonic()-start
            for k in a:
                if isinstance(a[k],str):assert a[k]==b[k]
                else:np.testing.assert_allclose(a[k],b[k],rtol=1e-13,atol=1e-13,equal_nan=True)
            checks.append(dict(source_row=r['source_row'],original_seconds=ta,cached_seconds=tb))
    PRF.TESS_PRF=OriginalPRF
    write_json(ROOT/'extractor_parity.json',dict(status='passed',observations=len(checks),
        tolerance=1e-13,all_features_agree=True,script_sha256=sha256(__file__),checks=checks,
        limitation='Same PRF and scene assumptions as the study; cache optimization is not a new physical model.'))
    print('Cached PRF/packed feature parity passed',flush=True)


CATALOG_MAP=None


def initialize():
    global CATALOG_MAP,feature
    ids=np.sort(pd.read_parquet(ROOT/'manifest.parquet',columns=['tic_id']).tic_id.unique())
    CATALOG_MAP={int(tic):i//128 for i,tic in enumerate(ids)}
    PRF.TESS_PRF=CachedPRF
    from fast_aperture_geometry import feature as fast_feature
    feature=fast_feature


@lru_cache(maxsize=4)
def catalog_batch(number):
    path=ROOT/'catalogs'/f'batch_{number:04d}.parquet'
    return pd.read_parquet(path) if path.exists() else None


def worker(path):
    dest=ROOT/'features'/Path(path).name
    if dest.exists():return
    columns=pd.read_csv(STRAPS).imaging_column.to_numpy();f=pd.read_parquet(path);out=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tic,g in f.groupby('tic_id',sort=False):
            batch=catalog_batch(CATALOG_MAP[int(tic)])
            cat=None if batch is None else batch[batch.query_tic_id.eq(tic)]
            t=Table.from_pandas(cat.drop(columns=['query_tic_id','catalog_source'])) if cat is not None and len(cat) else None
            for r in g.to_dict('records'):
                row={k:int(r[k]) for k in ['source_row','tic_id','sector']};row['geometry_available']=False;row['headers_available']=False
                try:
                    if not r['available']:raise ValueError(r.get('error','Metadata unavailable'))
                    h,mask=unpack(r);row.update(header_features(h),headers_available=True)
                    if t is None:raise ValueError('Catalog query unavailable')
                    row.update(feature(*h,mask,int(tic),r['sector'],t,columns),geometry_available=True)
                except Exception as exc:row['error']=str(exc)
                out.append(row)
    atomic_parquet(pd.DataFrame(out),dest)
    write_json(dest.with_suffix('.json'),dict(rows=len(out),available=sum(r['geometry_available'] for r in out),
        input_sha256=sha256(path),output_sha256=sha256(dest)))


def extract(workers):
    assert json.loads((ROOT/'extractor_parity.json').read_text())['status']=='passed'
    fast_check=json.loads((ROOT/'fast_geometry_final_parity.json').read_text())
    assert fast_check['status']=='passed' and fast_check['script_sha256']==sha256('eval/fast_aperture_geometry.py')
    # Schedule a metadata batch as soon as all of its catalog requests finish.
    # This overlaps extraction with collection without ever caching absent files.
    manifest=pd.read_parquet(ROOT/'manifest.parquet',columns=['source_row','tic_id']).sort_values('source_row')
    ids=np.sort(manifest.tic_id.unique());catalog_index={int(t):i//128 for i,t in enumerate(ids)}
    needed={f'batch_{i//1024:04d}.parquet':set(manifest.iloc[i:i+1024].tic_id.map(catalog_index))
            for i in range(0,len(manifest),1024)}
    (ROOT/'features').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        while True:
            pending=[p for p in committed_batches(ROOT/'metadata') if not (ROOT/'features'/p.name).exists()]
            pending=[p for p in pending if all((ROOT/'catalogs'/f'batch_{j:04d}.json').exists() for j in needed[p.name])]
            for start in range(0,len(pending),workers*2):
                space_check()
                futures=[pool.submit(worker,p) for p in pending[start:start+workers*2]]
                for fu in as_completed(futures):fu.result()
                log(f'Full geometry: {len(committed_batches(ROOT/"features"))} batches complete')
            if (ROOT/'metadata_complete.json').exists():
                remaining=[p for p in committed_batches(ROOT/'metadata') if not (ROOT/'features'/p.name).exists()]
                if not remaining:break
            time.sleep(10)
    write_json(ROOT/'extraction_complete.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        batches=len(committed_batches(ROOT/'features')),script_sha256=sha256(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','extract']);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    if a.stage=='parity':parity()
    else:extract(a.workers)
