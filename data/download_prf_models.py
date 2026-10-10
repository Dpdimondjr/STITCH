"""Download a bounded official TESS PRF grid into the ignored local cache."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',type=int,required=True,choices=range(1,5))
    p.add_argument('--ccd',type=int,required=True,choices=range(1,5))
    p.add_argument('--epoch',default='start_s0004',choices=['start_s0001','start_s0004'])
    p.add_argument('--out',default='tess_cache/prf_models')
    a=p.parse_args()
    base=f'https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/{a.epoch}/cam{a.camera}_ccd{a.ccd}/'
    target=Path(a.out)/a.epoch/f'cam{a.camera}_ccd{a.ccd}';target.mkdir(parents=True,exist_ok=True)
    response=requests.get(base,timeout=60);response.raise_for_status()
    names=sorted(set(re.findall(r'href=["\']([^"\']+\.fits)["\']',response.text)))
    names=[n for n in names if 'phot' not in n]
    if len(names)!=25:raise ValueError(f'Expected 25 PRF grid files, found {len(names)}')
    manifest=[]
    for name in names:
        url=urljoin(base,name)
        if urlparse(url).netloc!='archive.stsci.edu':raise ValueError('Unexpected download host')
        path=target/Path(urlparse(url).path).name
        if not path.exists():
            r=requests.get(url,timeout=60);r.raise_for_status()
            if len(r.content)>2_000_000 or not r.content.startswith(b'SIMPLE'):
                raise ValueError('Unexpected PRF FITS response')
            temp=path.with_suffix('.part');temp.write_bytes(r.content);temp.replace(path)
        manifest.append({'file':path.name,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path.name,flush=True)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
