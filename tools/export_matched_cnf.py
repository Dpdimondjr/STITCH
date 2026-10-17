"""Package a matched 120-input Gaia CNF using the exact HGB reference assets."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from stitch.bundles import verify_bundle,seal_bundle
from stitch.features.context import FEATURES
from stitch.io import write_json,sha256


def export(checkpoint,hgb_bundle,out,protocol):
    checkpoint=Path(checkpoint);hgb_bundle=Path(hgb_bundle);out=Path(out)
    if out.exists():raise FileExistsError('Use a new bundle directory')
    verify_bundle(hgb_bundle)
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if sha256(protocol)!=c['protocol_sha256']:raise ValueError('Training protocol mismatch')
    inputs=json.loads(Path(protocol).read_text())['input_sha256']
    for name in ['reference_bank.parquet','cohort_manifest.parquet',*[f'color_fold_{i}.pkl' for i in range(5)]]:
        expected=[digest for path,digest in inputs.items() if Path(path).name==name]
        if expected!=[sha256(hgb_bundle/name)]:raise ValueError(f'HGB assets do not match training: {name}')
    if c['target']!='gaia_rp' or c['prior']!='pred_knn' or c['feature_cols']!=FEATURES:
        raise ValueError('Expected matched Gaia-RP CNF with HGB physical inputs')
    prep=c['preprocessing']
    if prep['features']!=FEATURES:raise ValueError('Preprocessing feature order mismatch')
    out.mkdir(parents=True)
    runtime=dict(state=c['state'],config=c['config'],feature_cols=FEATURES,
                 **{key:prep[key] for key in ['medians','means','stds','y_median','y_std']})
    torch.save(runtime,out/'model.pt')
    shutil.copy2(protocol,out/'training_protocol.json')
    write_json(out/'model.json',dict(target='gaia_rp',reference_family='hgb',prior='pred_knn',
        beta=c['beta'],clip=c['clip'],estimator='conditional median',epoch=c['epoch'],
        source_checkpoint_sha256=sha256(checkpoint),protocol_sha256=c['protocol_sha256'],
        calibration_metrics=c['metrics'],historical_checkpoint=False,new_gaia_cnf=True,
        training_rows=457123,training_stars=36507,default_model_promoted=False,
        long_baseline_preservation_validated=False))
    for name in ['reference_bank.parquet','cohort_manifest.parquet','identity_catalog.parquet','thermal.parquet',
                 *[f'color_fold_{i}.pkl' for i in range(5)]]:
        shutil.copy2(hgb_bundle/name,out/name)
    seal_bundle(out,dict(kind='cnf',target='gaia_rp',reference_family='hgb',
        status='Matched full-data research CNF; no automatic promotion',
        source_checkpoint_sha256=sha256(checkpoint),hgb_bundle_sha256=sha256(hgb_bundle/'bundle.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--hgb-bundle',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--protocol',required=True,type=Path)
    a=p.parse_args();export(a.checkpoint,a.hgb_bundle,a.out,a.protocol)
