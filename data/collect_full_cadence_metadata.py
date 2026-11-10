"""Stream matched light curves into compact background/pointing summaries.

Raw LC bodies are not persisted. Labels are only recorded in explicit lineage
columns for subsequent provenance checking, never used to choose acquisitions.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from collect_full_geometry import ROOT, STUDY, atomic_parquet, activation, space_check, committed_batches
from fetch_geometry_training_transfer import session
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from hgb_reference_repair import write_json,sha256,log

DEFAULT_BITMASK=17087
CADENCE_DIR='cadence_v2'


def local_inventory():
    path=ROOT/'local_lc_inventory.parquet'
    if path.exists():return
    files=[]
    # Exact filenames identify TIC/sector/version; contents are checked later.
    for p in Path('tess_cache').rglob('*_lc.fits'):
        if p.name.startswith('hlsp_tess-spoc_tess_phot_'):
            files.append(dict(filename=p.name,path=str(p),bytes=p.stat().st_size))
    f=pd.DataFrame(files,columns=['filename','path','bytes']).sort_values('path').drop_duplicates('filename')
    atomic_parquet(f,path)
    write_json(ROOT/'local_lc_inventory.json',dict(files=len(f),bytes=int(f.bytes.sum()),sha256=sha256(path)))


def fetch_body(rec,local):
    if local:
        data=Path(local).read_bytes();origin=local
    else:
        suffix=rec['data_uri'].split('tess-spoc/',1)[1]
        url='https://stpubdata.s3.amazonaws.com/mast/hlsp/tess-spoc/'+suffix
        response=session().get(url,stream=True,timeout=(15,75))
        if response.status_code==404:
            response.close()
            response=session().get('https://mast.stsci.edu/api/v0.1/Download/file',params={'uri':rec['data_uri']},stream=True,timeout=(15,75))
        with response:
            response.raise_for_status()
            if response.status_code!=200:raise ValueError('Unexpected status for complete LC body')
            data=response.content
        origin='MAST public archive; full body streamed into memory'
    if len(data)!=int(rec['size_bytes']):raise ValueError('LC length differs from frozen metadata product')
    if hashlib.sha256(data[:28800]).hexdigest()!=rec['prefix_sha256']:
        raise ValueError('LC prefix differs from the frozen metadata byte hash')
    return data,origin


def summarize(data,rec,origin):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with fits.open(io.BytesIO(data),memmap=False) as h:
            assert int(h[0].header['TICID'])==rec['tic_id'] and int(h[0].header['SECTOR'])==rec['sector']
            for name,i in [('primary',0),('lightcurve',1),('aperture_header',2)]:
                if h[i].header.tostring()!=rec[name]:raise ValueError('LC header differs from frozen metadata')
            bits=np.frombuffer(rec['aperture_bytes'],dtype=rec['aperture_dtype']).reshape(int(rec['aperture_height']),int(rec['aperture_width']))
            np.testing.assert_array_equal(h[2].data,bits)
            d=h[1].data;q=np.asarray(d['QUALITY'],np.int64);t=np.asarray(d['TIME'],float)
            b=np.asarray(d['SAP_BKG'],float);pc1=np.asarray(d['POS_CORR1'],float);pc2=np.asarray(d['POS_CORR2'],float)
            out=dict(cadence_available=True,n_cadences=len(q),quality_zero_fraction=float(np.mean(q==0)),
                default_quality_fraction=float(np.mean((q&DEFAULT_BITMASK)==0)),
                scatter_flag_frac=float(np.mean((q&4096)!=0)),
                full_file_sha256=hashlib.sha256(data).hexdigest(),cadence_source=origin,
                procver=str(h[0].header.get('PROCVER','')),data_rel=int(h[0].header.get('DATA_REL',-1)),
                transferred_bytes=0 if origin.startswith('tess_cache/') else len(data))
            for label,mask in [('allfinite',np.ones(len(q),bool)),('default',(q&DEFAULT_BITMASK)==0),('q0',q==0)]:
                good=mask&np.isfinite(b);out['n_background_'+label]=int(good.sum())
                for name,value in [('median_sap_bkg',np.median),('p90_sap_bkg',lambda x:np.quantile(x,.9)),('bkg_rms',np.std)]:
                    out[name+'_'+label]=float(value(b[good])) if good.any() else np.nan
                for name,values in [('pos_corr1',pc1),('pos_corr2',pc2)]:
                    goodp=mask&np.isfinite(values);v=values[goodp]
                    out[name+'_median_'+label]=float(np.median(v)) if len(v) else np.nan
                    out[name+'_rms_'+label]=float(np.std(v)) if len(v) else np.nan
                    out[name+'_p05_'+label]=float(np.quantile(v,.05)) if len(v) else np.nan
                    out[name+'_p95_'+label]=float(np.quantile(v,.95)) if len(v) else np.nan
                out['jitter_rms_'+label]=float(np.hypot(out['pos_corr1_rms_'+label],out['pos_corr2_rms_'+label]))
            pdc=np.asarray(d['PDCSAP_FLUX'],float)
            valid=np.isfinite(pdc)&np.isfinite(t)&((q&DEFAULT_BITMASK)==0)
            out['lineage_pdcsap_median_default']=float(np.median(pdc[valid])) if valid.any() else np.nan
            out['lineage_n_pdcsap_default']=int(valid.sum())
            # Formal errors remain measurement diagnostics, never a total DC floor.
            for col in ['SAP_FLUX_ERR','PDCSAP_FLUX_ERR','SAP_BKG_ERR']:
                v=np.asarray(d[col],float);good=np.isfinite(v)&(v>0)&(q==0)
                out[col.lower()+'_median_q0']=float(np.median(v[good])) if good.any() else np.nan
            # Flux-weighted centroids are diagnostics, distinct from ensemble
            # pointing POS_CORR and from the incompletely documented legacy perr.
            for col in ['MOM_CENTR1','MOM_CENTR2','MOM_CENTR1_ERR','MOM_CENTR2_ERR']:
                v=np.asarray(d[col],float)
                for label,mask in [('allfinite',np.ones(len(q),bool)),('default',(q&DEFAULT_BITMASK)==0),('q0',q==0)]:
                    good=mask&np.isfinite(v)
                    out[col.lower()+'_median_'+label]=float(np.median(v[good])) if good.any() else np.nan
                    out[col.lower()+'_mean_'+label]=float(np.mean(v[good])) if good.any() else np.nan
                    if not col.endswith('_ERR'):
                        out[col.lower()+'_rms_'+label]=float(np.std(v[good])) if good.any() else np.nan
            good=(q==0)&np.isfinite(t)&np.isfinite(b)
            if good.sum()>1:
                tt=t[good];bb=b[good];first=tt<np.median(tt)
                out['background_half_shift']=float(np.median(bb[~first])-np.median(bb[first])) if first.any() and (~first).any() else np.nan
            else:out['background_half_shift']=np.nan
            return out


def one(task):
    rec,local=task;out={k:int(rec[k]) for k in ['source_row','tic_id','sector']};out['cadence_available']=False
    if not rec['available']:return {**out,'error':'Frozen geometry product unavailable'}
    for attempt in range(3):
        try:
            data,origin=fetch_body(rec,local)
            return {**out,**summarize(data,rec,origin)}
        except Exception as exc:
            # A stale historical cache must not prevent querying the exact product.
            if local:local=None
            if attempt==2:return {**out,'error':str(exc)}
            time.sleep(2*(attempt+1))


LOCAL_LC=None


def initialize_worker():
    global LOCAL_LC
    LOCAL_LC=pd.read_parquet(ROOT/'local_lc_inventory.parquet').set_index('filename').path.to_dict()


def collect_batch(task):
    path,workers=task;path=Path(path);dest=ROOT/CADENCE_DIR/path.name
    if dest.exists():return
    space_check();f=pd.read_parquet(path)
    tasks=[(r,LOCAL_LC.get(r.get('filename',''))) for r in f.to_dict('records')]
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(one,tasks))
    atomic_parquet(pd.DataFrame(rows),dest)
    write_json(dest.with_suffix('.json'),dict(rows=len(rows),available=sum(r['cadence_available'] for r in rows),
        input_sha256=sha256(path),sha256=sha256(dest),transferred_bytes=sum(r.get('transferred_bytes',0) for r in rows)))
    log(f'Full cadence v2 summaries: {len(committed_batches(ROOT/CADENCE_DIR))} batches; latest {sum(r["cadence_available"] for r in rows)}/{len(rows)} available')


def collect(workers,processes):
    activation();local_inventory()
    (ROOT/CADENCE_DIR).mkdir(exist_ok=True)
    if not (ROOT/'cadence_v2_protocol.json').exists():write_json(ROOT/'cadence_v2_protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        script_sha256=sha256(__file__),version=2,
        reason='Include explicitly defined flux-weighted centroid measurements while each LC body is available; legacy perr definitions are not assumed.',
        previous='Initial cadence-only summaries and their source remain preserved under cadence/ and source_snapshot/collect_full_cadence_metadata_v1.py.',
        selection='All retained rows; no outcome-based acquisition or correction scoring.',
        quality_views=['allfinite','Lightkurve default bitmask 17087','QUALITY=0'],
        limitation='Centroids can depend on variable target/neighbor light. They are recorded diagnostics, not automatically adopted signal-preserving predictors.'))
    run=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),script_sha256=sha256(__file__),processes=processes,threads_per_process=workers,measurements_unchanged=True)
    write_json(ROOT/f'cadence_run_{int(time.time())}.json',run)
    with ProcessPoolExecutor(max_workers=processes,initializer=initialize_worker) as pool:
        while True:
            paths=[p for p in committed_batches(ROOT/'metadata') if not (ROOT/CADENCE_DIR/p.name).exists()]
            for start in range(0,len(paths),processes):
                futures=[pool.submit(collect_batch,(str(p),workers)) for p in paths[start:start+processes]]
                for fu in as_completed(futures):fu.result()
            if (ROOT/'metadata_complete.json').exists():
                remaining=[p for p in committed_batches(ROOT/'metadata') if not (ROOT/CADENCE_DIR/p.name).exists()]
                if not remaining:break
            time.sleep(10)
    write_json(ROOT/'cadence_complete.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        batches=len(committed_batches(ROOT/CADENCE_DIR)),default_quality_bitmask=DEFAULT_BITMASK,version=2,
        script_sha256=sha256(__file__),raw_lightcurves_saved=False))


def parity():
    from collect_full_geometry import packed_cached
    lineage=STUDY/'lineage_checks'
    files=list(lineage.rglob('*_lc.fits'))[:8]
    if not files:raise ValueError('No local lineage FITS files')
    products=pd.read_parquet(STUDY/'products.parquet').set_index('filename');checks=[]
    for p in files:
        r=products.loc[p.name].to_dict();r['filename']=p.name
        cache=Path('tess_cache/geometry_training_transfer_20261012')/f"{r['tic_id']}_{r['sector']:04d}_v{r['version']}.npz"
        r.update(packed_cached(r,cache));data=p.read_bytes();a=summarize(data,r,str(p))
        with fits.open(p,memmap=False) as h:
            q=h[1].data['QUALITY'];b=h[1].data['SAP_BKG'];v=b[(q==0)&np.isfinite(b)].astype(float)
            np.testing.assert_allclose([a['median_sap_bkg_q0'],a['p90_sap_bkg_q0'],a['bkg_rms_q0']],
                [np.median(v),np.quantile(v,.9),np.std(v)],atol=1e-12)
        checks.append(dict(filename=p.name,sha256=sha256(p),available=a['cadence_available']))
    write_json(ROOT/'cadence_parity.json',dict(status='passed',observations=len(checks),checks=checks,
        script_sha256=sha256(__file__),diagnostics_are_not_noise_floor=True))
    print('Cadence summary parity passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','collect']);p.add_argument('--workers',type=int,default=4);p.add_argument('--processes',type=int,default=3);a=p.parse_args()
    if a.stage=='parity':parity()
    else:
        assert json.loads((ROOT/'cadence_parity.json').read_text())['status']=='passed'
        collect(a.workers,a.processes)
