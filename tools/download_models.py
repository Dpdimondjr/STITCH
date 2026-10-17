"""Download immutable model assets listed in the release manifest, with hashes."""
import argparse,hashlib,json,shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import requests


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def fetch(url,path,expected):
    if path.exists():
        if digest(path)!=expected:raise ValueError(f'Existing artifact has a different hash: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part')
    try:
        with requests.get(url,stream=True,timeout=(15,120)) as response:
            response.raise_for_status()
            with tmp.open('wb') as output:
                for block in response.iter_content(1024**2):
                    if shutil.disk_usage(path.parent).free<4*1024**3:raise RuntimeError('Keep 4 GiB free disk reserve')
                    output.write(block)
        if digest(tmp)!=expected:raise ValueError(f'Download hash mismatch: {path}')
        tmp.replace(path)
    finally:
        if tmp.exists():tmp.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',choices=['hgb','cnf','cnf_gaia','geometry'],action='append',required=True);p.add_argument('--out',type=Path,default=Path('artifacts'));p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'models/manifest.json');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());a.out.mkdir(parents=True,exist_ok=True)
    root_url=f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['asset_commit']}"
    for name in dict.fromkeys(a.bundle):
        info=manifest['bundles'][name];root=(a.out/name).resolve()
        if not root.exists() and shutil.disk_usage(a.out).free<4*1024**3+info['bytes']:raise RuntimeError('Insufficient disk reserve for bundle')
        fetch(f'{root_url}/{name}/bundle.json',root/'bundle.json',info['manifest_sha256'])
        bundle=json.loads((root/'bundle.json').read_text());items=[]
        for member,expected in bundle['sha256'].items():
            dest=(root/member).resolve()
            if root not in dest.parents:raise ValueError('Invalid bundle member path')
            items.append((f'{root_url}/{name}/{quote(member,safe="/")}',dest,expected))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0,len(items),2):
                for future in [pool.submit(fetch,*item) for item in items[start:start+2]]:future.result()
        print(f'{name}: verified {len(items)} assets in {root}')


if __name__=='__main__':main()
