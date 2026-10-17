"""Bounded, resumable LC acquisition; measured outputs survive raw-byte release."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import gzip,hashlib,io,json,shutil
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from ..io import sha256,write_json
from .measure import measure,summarize

MAX_BODY=64*1024**2


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def load_body(rec):
    cached=rec.get('cached_path')
    if isinstance(cached,str) and Path(cached).is_file():
        if Path(cached).stat().st_size>MAX_BODY:raise ValueError('Cached file exceeds byte limit')
        packed=Path(cached).read_bytes()
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) if packed.startswith(b'\x1f\x8b') else io.BytesIO(packed) as stream:
            body=stream.read(MAX_BODY+1)
        if len(body)>MAX_BODY:raise ValueError('Decompressed FITS exceeds byte limit')
        return body,'existing_cache',0
    uri=rec['data_uri']
    if uri.startswith('mast:'):
        url='https://mast.stsci.edu/api/v0.1/Download/file';params={'uri':uri}
    elif uri.startswith('https://'):url=uri;params=None
    else:raise ValueError('Require a MAST URI or HTTPS archive URL')
    chunks=[];size=0
    with requests.get(url,params=params,stream=True,timeout=(15,60)) as response:
        response.raise_for_status()
        if response.status_code!=200:raise ValueError('Expected complete archive response')
        for chunk in response.iter_content(256*1024):
            size+=len(chunk)
            if size>MAX_BODY:raise ValueError('FITS exceeds byte limit')
            chunks.append(chunk)
    return b''.join(chunks),'archive_memory',size


def collect(requests_path,out,workers=2,resume=False,reserve_gib=4,output_limit_gib=4):
    if workers not in [1,2]:raise ValueError('Use one or two simultaneous downloads')
    if reserve_gib<4:raise ValueError('Retain at least 4 GiB free disk')
    out=Path(out);source=Path(requests_path).resolve()
    rows=pd.read_csv(source) if source.suffix=='.csv' else pd.read_parquet(source)
    if set(['tic_id','sector','data_uri'])-set(rows):raise ValueError('Requests need tic_id, sector, data_uri; optional cached_path and expected_raw_sha256')
    if 'source_row' not in rows:rows['source_row']=np.arange(len(rows))
    if rows.empty or not rows.source_row.is_unique or rows.duplicated(['tic_id','sector']).any():raise ValueError('Require nonempty unique source_row and TIC/sector requests')
    for col in ['source_row','tic_id','sector']:
        if not np.equal(rows[col],rows[col].astype('int64')).all() or rows[col].lt(0).any():raise ValueError(f'Invalid {col}')
    signature=sha256(source)
    if out.exists():
        if not resume:raise FileExistsError('Use a new output directory or explicit --resume')
        if json.loads((out/'protocol.json').read_text())['requests_sha256']!=signature:raise ValueError('Resume requests changed')
    else:
        out.mkdir(parents=True)
        write_json(out/'protocol.json',dict(requests_sha256=signature,workers=workers,reserve_gib=reserve_gib,output_limit_gib=output_limit_gib,quality_policy='Persist quality flags and separate QUALITY=0/default bitmask 17087 measurements for both SAP and PDCSAP',raw_cache=False))
    for name in ['measurements','cadences','metadata','receipts','failures']:(out/name).mkdir(exist_ok=True)
    completed=[];pending=[]
    for rec in rows.to_dict('records'):
        receipt=out/'receipts'/f"{int(rec['source_row'])}.json"
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            for name,digest in saved['output_sha256'].items():
                if sha256(out/name)!=digest:raise ValueError(f'Resume output changed: {name}')
            completed.append(saved)
        else:pending.append(rec)
    def one(rec):
        key=int(rec['source_row']);body,origin,transferred=load_body(rec)
        measurements=measure(body,rec)
        with fits.open(io.BytesIO(body),memmap=False) as hdus:
            headers=[hdus[i].header for i in range(3)];bits=hdus[2].data
            metadata=dict(tic_id=int(rec['tic_id']),sector=int(rec['sector']),primary=headers[0].tostring(),lightcurve=headers[1].tostring(),aperture_header=headers[2].tostring(),aperture_bytes=bits.tobytes(),aperture_dtype=str(bits.dtype),aperture_height=bits.shape[0],aperture_width=bits.shape[1])
            cadence=summarize(body,metadata,origin)
            metadata.pop('aperture_bytes');metadata['aperture_pixels']=bits.tolist();metadata['cadence']=clean(cadence)
            metadata['systematic_score']=clean(rec.get('systematic_score',np.nan))
            metadata['product_family']='tess-spoc' if 'tess-spoc' in rec['data_uri'] else 'unverified'
            metadata['data_uri']=rec['data_uri'];metadata['raw_sha256']=measurements['raw_sha256']
            data=hdus[1].data
            curves=pd.DataFrame({k.lower():np.asarray(data[k]).astype(np.int64 if k=='QUALITY' else float) for k in ['TIME','QUALITY','SAP_FLUX','SAP_FLUX_ERR','PDCSAP_FLUX','PDCSAP_FLUX_ERR']})
            curves.insert(0,'source_row',key);curves['tic_id']=int(rec['tic_id']);curves['sector']=int(rec['sector'])
        paths=[out/'measurements'/f'{key}.parquet',out/'cadences'/f'{key}.parquet',out/'metadata'/f'{key}.json']
        for frame,path in [(pd.DataFrame([dict(source_row=key,tic_id=int(rec['tic_id']),sector=int(rec['sector']),**measurements)]),paths[0]),(curves,paths[1])]:
            tmp=path.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(path)
        write_json(paths[2],metadata)
        receipt=dict(source_row=key,tic_id=int(rec['tic_id']),sector=int(rec['sector']),data_uri=rec['data_uri'],raw_sha256=measurements['raw_sha256'],raw_bytes=len(body),origin=origin,network_bytes=transferred,measured_utc=datetime.now(timezone.utc).isoformat(),output_sha256={str(p.relative_to(out)):sha256(p) for p in paths})
        write_json(out/'receipts'/f'{key}.json',receipt)
        return receipt
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(pending),workers):
            if shutil.disk_usage(out).free<reserve_gib*1024**3+workers*MAX_BODY:raise RuntimeError('Disk reserve reached; safe to resume later')
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>=output_limit_gib*1024**3:raise RuntimeError('Output limit reached; use a smaller request batch')
            batch=pending[start:start+workers];futures=[pool.submit(one,r) for r in batch]
            for rec,future in zip(batch,futures):
                try:completed.append(future.result())
                except Exception as exc:write_json(out/'failures'/f"{int(rec['source_row'])}.json",dict(error=str(exc),tic_id=int(rec['tic_id']),sector=int(rec['sector'])))
            print(f'{len(completed)}/{len(rows)} products verified',flush=True)
    write_json(out/'status.json',dict(requested=len(rows),verified=len(completed),remaining=len(rows)-len(completed),new_fits_cache_files=0))
    if len(completed)!=len(rows):raise RuntimeError('Some products failed; inspect failures/ and resume')
