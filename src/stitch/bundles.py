"""Verify versioned model bundles before loading serialized estimators."""
import json
from pathlib import Path
from .io import sha256,write_json


def verify_bundle(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'bundle.json').read_text())
    if manifest.get('schema_version')!=1:
        raise ValueError('Unsupported bundle schema')
    checked={}
    for name,digest in manifest['sha256'].items():
        p=(root/name).resolve()
        if root not in p.parents or not p.is_file():
            raise ValueError(f'Invalid or absent bundle member: {name}')
        actual=sha256(p)
        if actual!=digest:raise ValueError(f'Bundle hash mismatch: {name}')
        checked[name]=actual
    return checked


def seal_bundle(root,metadata):
    root=Path(root)
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='bundle.json')
    write_json(root/'bundle.json',dict(schema_version=1,**metadata,sha256={str(p.relative_to(root)):sha256(p) for p in paths}))
