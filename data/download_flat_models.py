"""Fetch the 16 public TESS flat fields (MAST models/tess2018323-{cam}-{ccd}-flat.fits) used by eval/flat_imprint.py.
At most two simultaneous downloads; existing files are reused; receipts (URL, bytes, sha256) go to receipts.json."""
import concurrent.futures as cf,hashlib,json,pathlib,urllib.request

OUT=pathlib.Path('tess_cache/flat_models');BASE='https://archive.stsci.edu/missions/tess/models/'


def get(job):
    cam,ccd=job;name=f'tess2018323-{cam}-{ccd}-flat.fits';path=OUT/name
    if not path.exists():
        req=urllib.request.Request(BASE+name,headers={'User-Agent':'stitch-calibration-study'})
        path.write_bytes(urllib.request.urlopen(req,timeout=300).read())
    data=path.read_bytes();return dict(name=name,url=BASE+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with cf.ThreadPoolExecutor(2) as pool:rows=list(pool.map(get,[(c,k) for c in range(1,5) for k in range(1,5)]))
    (OUT/'receipts.json').write_text(json.dumps(rows,indent=1)+'\n');print(len(rows),'files',sum(r['bytes'] for r in rows)/1e6,'MB')
