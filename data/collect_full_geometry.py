"""Resumable, lossless metadata collection for the frozen full STITCH population.

Bulk scripts are parsed as archive listings, never executed. No outcome column
is loaded by acquisition. Existing products and research artifacts are untouched.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from fetch_geometry_training_transfer import byte_range, header_from_bytes, session, mast_client
from audit_full_geometry_collection import ROOT, STUDY, SOURCE
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from hgb_reference_repair import sha256, write_json, log

BATCH_SIZE = 1024
RESERVE = 2**30
CATALOG_FIELDS = ['ID', 'ra', 'dec', 'RA_orig', 'Dec_orig', 'POSflag', 'pmRA',
                  'pmDEC', 'Tmag', 'gaiarp', 'gaiabp', 'GAIA', 'disposition',
                  'contratio', 'numcont', 'e_Tmag', 'e_gaiarp', 'e_gaiabp',
                  'e_RA', 'e_Dec', 'e_pmRA', 'e_pmDEC', 'Teff']
PRODUCT_RE = re.compile(r'hlsp_tess-spoc_tess_phot_(\d+)-s(\d+)_tess_v(\d+)_lc\.fits')


def space_check():
    if shutil.disk_usage(ROOT).free < RESERVE:
        raise RuntimeError('Below 1 GiB free-space reserve; completed batches are resumable.')


def atomic_parquet(frame, path):
    space_check();path.parent.mkdir(exist_ok=True, parents=True)
    temp=path.with_suffix('.part.parquet')
    frame.to_parquet(temp,index=False,compression='zstd',compression_level=9)
    # Ensure a complete readable parquet footer before committing a batch.
    import pyarrow.parquet as pq
    assert pq.ParquetFile(temp).metadata.num_rows==len(frame)
    temp.replace(path)


def committed_batches(folder):
    """Discover only final batch names with their accompanying commit receipt."""
    return [p for p in sorted(Path(folder).glob('batch_*.parquet'))
        if re.fullmatch(r'batch_\d+\.parquet',p.name) and p.with_suffix('.json').is_file()]


def activation(wait=False):
    required=[STUDY/'results.json',STUDY/'full_bank_transfer/results.json',
              STUDY/'verification.json',STUDY/'strength_control/results.json']
    while not all(p.exists() for p in required):
        if not wait:raise RuntimeError('Frozen transfer and verification have not finished.')
        time.sleep(15)
    primary,full,verification,control=[json.loads(p.read_text()) for p in required]
    selected=primary['selected']['family']
    a=next(r for r in primary['metrics'] if r['selected'] and r['view']=='qualified')
    b=next(r for r in full['metrics'] if r['view']=='qualified')
    checks=dict(primary_positive_95=a['gain95_pp'][0]>0,
                full_transfer_positive_95=b['gain95_pp'][0]>0,
                verification_passed=verification['status']=='passed',
                geometry_beats_strength_control=control['qualified_geometry_increment_passes'])
    decision=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),selected=selected,
        passed=all(checks.values()),checks=checks,
        evidence=[dict(path=str(p),sha256=sha256(p)) for p in required],
        protocol_sha256=sha256(ROOT/'activation_protocol.json'),
        strength_protocol_sha256=sha256(STUDY/'strength_control/freeze.json'))
    path=ROOT/'activation_decision.json'
    if path.exists():
        prior=json.loads(path.read_text())
        stable=['selected','passed','checks','evidence','protocol_sha256','strength_protocol_sha256']
        if any(prior.get(key)!=decision[key] for key in stable):
            raise RuntimeError('Frozen acquisition evidence changed. The existing decision is preserved; use a new experiment directory for a different protocol or result.')
        decision=prior
    else:write_json(path,decision)
    if not decision['passed']:raise RuntimeError('The frozen scale-up condition did not pass. See activation_decision.json; no bulk collection started.')
    return decision


def inventory_sector(task):
    sector,ids=task;path=ROOT/'inventory'/f's{sector:04d}.parquet'
    receipt=path.with_suffix('.json')
    if path.exists() and receipt.exists():return
    url=f'https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/hlsp_tess-spoc_tess_phot_s{sector:04d}_tess_v1_dl-lc.sh'
    for attempt in range(3):
        try:
            rows=[];digest=hashlib.sha256();count=0;downloaded=0;buffer=b''
            with session().get(url,stream=True,timeout=(15,90)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(256*1024):
                    digest.update(chunk);downloaded+=len(chunk);buffer+=chunk
                    lines=buffer.split(b'\n');buffer=lines.pop()
                    for line in lines:
                        text=line.decode('ascii');m=PRODUCT_RE.search(text)
                        if not m:continue
                        count+=1;tic=int(m[1]);sec=int(m[2]);version=int(m[3])
                        if tic not in ids or sec!=sector:continue
                        uri=re.search(r'uri=(mast:HLSP/tess-spoc/[^\s\'\"]+)',text)
                        if not uri:raise ValueError('Matching listing line lacks an archive URI')
                        rows.append(dict(tic_id=tic,sector=sector,version=version,
                            filename=m[0],data_uri=uri[1],listing_line=text))
            if buffer.strip():raise ValueError('Listing lacks its terminal newline; reject possible truncation.')
            if count<1000:raise ValueError('Implausibly short sector listing')
            f=pd.DataFrame(rows,columns=['tic_id','sector','version','filename','data_uri','listing_line'])
            f=f.sort_values(['tic_id','sector','version','filename']).drop_duplicates(['tic_id','sector'],keep='last')
            atomic_parquet(f,path)
            write_json(receipt,dict(url=url,retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                listing_sha256=digest.hexdigest(),listing_bytes=downloaded,
                listed_products=count,matched_products=len(f),expected_observations=len(ids),
                subset_sha256=sha256(path),scripts_executed=False))
            return
        except Exception:
            if attempt==2:raise
            time.sleep(3*(attempt+1))


def inventory():
    (ROOT/'inventory').mkdir(exist_ok=True)
    if (ROOT/'inventory_complete.json').exists():return
    m=pd.read_parquet(ROOT/'manifest.parquet')
    tasks=[(int(s),set(g.tic_id)) for s,g in m.groupby('sector')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(inventory_sector,t) for t in tasks]
        for i,fu in enumerate(as_completed(futures),1):fu.result();log(f'Full inventory: {i}/{len(tasks)} sectors')
    p=pd.concat([pd.read_parquet(ROOT/'inventory'/f's{s:04d}.parquet') for s,_ in tasks],ignore_index=True)
    p=p.drop(columns='listing_line')
    f=m.merge(p,on=['tic_id','sector'],how='left',validate='one_to_one')
    # Same URI/version as the independently queried 17,680-observation study.
    known=pd.read_parquet(STUDY/'products.parquet')
    matched=f[['source_row','data_uri']].merge(known[['source_row','data_uri']],on='source_row',suffixes=('_bulk','_mast'),validate='one_to_one')
    assert matched.data_uri_bulk.eq(matched.data_uri_mast).all()
    atomic_parquet(f,ROOT/'products.parquet')
    write_json(ROOT/'inventory_complete.json',dict(rows=len(f),available=int(f.data_uri.notna().sum()),
        missing=int(f.data_uri.isna().sum()),independent_api_matches=len(matched),
        products_sha256=sha256(ROOT/'products.parquet')))


def packed_cached(rec,path):
    with np.load(path) as z:
        if str(z['data_uri'])!=rec['data_uri']:raise ValueError('Cached URI mismatch')
        d={k:str(z[k]) for k in ['primary','lightcurve','aperture_header','data_uri','filename','etag','prefix_sha256','aperture_sha256']}
        a=z['aperture'];d.update(aperture_bytes=a.tobytes(),aperture_dtype=a.dtype.str,
            aperture_height=a.shape[0],aperture_width=a.shape[1],size_bytes=int(z['size_bytes']),
            bytes_transferred=int(z['bytes_transferred']),acquisition_source=str(path))
        return d


def packed_remote(rec):
    first,total,etag=byte_range(rec['data_uri'],0,28799)
    h0,n0=header_from_bytes(first);h1,n1=header_from_bytes(first,n0)
    if int(h0['TICID'])!=rec['tic_id'] or int(h0['SECTOR'])!=rec['sector']:raise ValueError('FITS identity mismatch')
    data_bytes=h1['NAXIS1']*h1['NAXIS2']+h1.get('PCOUNT',0)
    offset=n0+n1+((data_bytes+2879)//2880)*2880
    last,total2,etag2=byte_range(rec['data_uri'],offset,min(total-1,offset+57599))
    if (total,etag)!=(total2,etag2):raise ValueError('Product changed between range reads')
    h2,_=header_from_bytes(last)
    if h2.get('EXTNAME')!='APERTURE':raise ValueError('Wrong image extension')
    a=fits.ImageHDU.fromstring(last).data.copy()
    if a.ndim!=2:raise ValueError('Invalid aperture dimensions')
    return dict(primary=h0.tostring(),lightcurve=h1.tostring(),aperture_header=h2.tostring(),
        data_uri=rec['data_uri'],filename=rec['filename'],etag=etag,
        prefix_sha256=hashlib.sha256(first).hexdigest(),aperture_sha256=hashlib.sha256(last).hexdigest(),
        aperture_bytes=a.tobytes(),aperture_dtype=a.dtype.str,aperture_height=a.shape[0],
        aperture_width=a.shape[1],size_bytes=total,bytes_transferred=len(first)+len(last),
        acquisition_source='MAST public archive: verified byte ranges')


def collect_one(rec):
    out={k:int(rec[k]) for k in ['source_row','tic_id','sector']};out['available']=False
    if pd.isna(rec['data_uri']):return {**out,'error':'No matching product in official sector LC listing'}
    path=Path('tess_cache/geometry_training_transfer_20261012')/f"{rec['tic_id']}_{rec['sector']:04d}_v{int(rec['version'])}.npz"
    for attempt in range(3):
        try:
            d=packed_cached(rec,path) if path.exists() else packed_remote(rec)
            h0=fits.Header.fromstring(d['primary'],sep='')
            assert int(h0['TICID'])==rec['tic_id'] and int(h0['SECTOR'])==rec['sector']
            return {**out,**d,'available':True}
        except Exception as exc:
            if attempt==2:return {**out,'error':str(exc)}
            time.sleep(2*(attempt+1))


def metadata(workers=16):
    if workers<1:raise ValueError('Metadata worker count must be positive')
    f=pd.read_parquet(ROOT/'products.parquet').sort_values('source_row')
    (ROOT/'metadata').mkdir(exist_ok=True)
    write_json(ROOT/f'metadata_run_{pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")}.json',
        dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),workers=workers,script_sha256=sha256(__file__)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(f),BATCH_SIZE):
            path=ROOT/'metadata'/f'batch_{start//BATCH_SIZE:04d}.parquet'
            if path.exists():continue
            space_check();part=f.iloc[start:start+BATCH_SIZE]
            rows=list(pool.map(collect_one,part.to_dict('records')))
            atomic_parquet(pd.DataFrame(rows),path)
            write_json(path.with_suffix('.json'),dict(rows=len(rows),available=sum(r['available'] for r in rows),
                sha256=sha256(path),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
            log(f'Full metadata: {start+len(part)}/{len(f)} rows')
    write_json(ROOT/'metadata_complete.json',dict(rows=len(f),batches=(len(f)+BATCH_SIZE-1)//BATCH_SIZE))


def catalog_one(rec):
    tic=int(rec['tic_id']);known=STUDY/'catalogs'/f'tic_{tic}.ecsv'
    for attempt in range(3):
        try:
            if known.exists():t=Table.read(known);source=str(known)
            else:
                t=mast_client('catalogs').query_region(SkyCoord(rec['ra'],rec['dec'],unit='deg'),radius=240*u.arcsec,catalog='Tic');source='MAST TIC cone: 240 arcsec'
            assert str(tic) in np.asarray(t['ID']).astype(str)
            cols=[c for c in CATALOG_FIELDS if c in t.colnames]
            g=t[cols].to_pandas();g['query_tic_id']=tic;g['catalog_source']=source
            for c in ['ID','GAIA','POSflag','disposition']:
                if c in g:g[c]=g[c].fillna('').astype(str)
            return g,dict(tic_id=tic,available=True,sources=len(t),catalog_source=source)
        except Exception as exc:
            if attempt==2:return None,dict(tic_id=tic,available=False,error=str(exc))
            time.sleep(3*(attempt+1))


def catalogs():
    f=pd.read_parquet(ROOT/'manifest.parquet',columns=['tic_id','ra','dec']).drop_duplicates('tic_id').sort_values('tic_id')
    (ROOT/'catalogs').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(f),128):
            path=ROOT/'catalogs'/f'batch_{start//128:04d}.parquet';receipt=path.with_suffix('.json')
            if path.exists() and receipt.exists():continue
            space_check();results=list(pool.map(catalog_one,f.iloc[start:start+128].to_dict('records')))
            frames=[g for g,r in results if g is not None]
            atomic_parquet(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame({'query_tic_id':[]}),path)
            write_json(receipt,dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                results=[r for g,r in results],sha256=sha256(path),radius_arcsec=240,
                saved_columns=CATALOG_FIELDS,full_response_columns_not_retained=True))
            log(f'Full scene catalogs: {min(start+128,len(f))}/{len(f)} TICs')
    write_json(ROOT/'catalogs_complete.json',dict(stars=len(f)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['activation','inventory','metadata','catalogs','all']);p.add_argument('--wait',action='store_true');p.add_argument('--metadata-workers',type=int,default=16);a=p.parse_args()
    activation(a.wait)
    for name,fn in [('inventory',inventory),('metadata',metadata),('catalogs',catalogs)]:
        if a.stage in [name,'all']:
            if name=='metadata':fn(a.metadata_workers)
            else:fn()
