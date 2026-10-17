"""Public commands; all science inputs and output locations are explicit."""
import argparse,json,os
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(prog='stitch',description='STITCH TESS calibration research workflows')
    p.add_argument('--threads',type=int,default=1)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect',help='Bounded LC acquisition and verified SAP/PDCSAP measurements')
    c.add_argument('--requests',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--workers',type=int,default=2);c.add_argument('--resume',action='store_true');c.add_argument('--output-limit-gib',type=float,default=4)
    c=sub.add_parser('features',help='Physical features from collected metadata and explicit catalog/PRF assets')
    for name in ['collection','catalog-dir','thermal','prf-dir','strap-path','out']:c.add_argument('--'+name,type=Path,required=True)
    c=sub.add_parser('curves',help='Export retained QUALITY=0 PDCSAP cadences from collection outputs')
    c.add_argument('--collection',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    c=sub.add_parser('prepare',help='Source-safe references and nested Gaia RP passband transfer')
    for name in ['metadata','labels','splits','out']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--family',choices=['hgb','cnf'],default='hgb')
    c=sub.add_parser('train-hgb',help='Five-round Gaia RP HGB')
    for name in ['prepared','out','identity-catalog']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--config',type=Path)
    c=sub.add_parser('train-cnf',help='Conditional NSF; new training defaults to Gaia RP')
    c.add_argument('--prepared',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--identity-catalog',type=Path,required=True);c.add_argument('--epochs',type=int,default=60);c.add_argument('--batch-size',type=int,default=2048);c.add_argument('--target',choices=['gaia_rp','relative'],default='gaia_rp')
    for name in ['predict-hgb','predict-cnf']:
        c=sub.add_parser(name,help='Apply verified model to physical metadata or prepared features')
        for arg in ['model','features','out']:c.add_argument('--'+arg,type=Path,required=True)
        c.add_argument('--prepared',action='store_true',help='Features already include the correct frozen reference statistics')
        c.add_argument('--catalog-dir',type=Path,help='Required identity catalogs for new-source reference attachment')
        c.add_argument('--curves',type=Path)
    c=sub.add_parser('dashboard',help='Original star grid plus two full-light-curve panels')
    for name in ['predictions','curves','out']:c.add_argument('--'+name,type=Path,required=True)
    c.add_argument('--model-spec',type=Path)
    c=sub.add_parser('verify-model',help='Check all frozen bundle members')
    c.add_argument('model',type=Path)
    return p


def predict(args):
    import numpy as np
    import pandas as pd
    from astropy.table import Table
    from .io import write_json,sha256
    from .hgb.model import FrozenPhysicalModel
    if args.out.exists():raise FileExistsError('Use a new inference directory')
    f=pd.read_parquet(args.features)
    if f.empty or f.duplicated(['tic_id','sector']).any():raise ValueError('Require nonempty unique TIC/sector inputs')
    if args.command=='predict-hgb':model=FrozenPhysicalModel(args.model)
    else:
        from .cnf.model import CNFModel
        model=CNFModel(args.model)
    if not args.prepared:
        if args.catalog_dir is None:raise ValueError('New-source inference requires --catalog-dir; use --prepared only for verified prepared inputs')
        from types import SimpleNamespace
        cohort=pd.read_parquet(args.model/'cohort_manifest.parquet')
        frames=[]
        for tic,g in f.groupby('tic_id',sort=False):
            table=Table.read(args.catalog_dir/f'{int(tic)}_cone.ecsv');targets=table[np.asarray(table['ID']).astype(str)==str(int(tic))]
            if len(targets)!=1:raise ValueError('Ambiguous target catalog identity')
            FrozenPhysicalModel.check_new_identity(SimpleNamespace(root=args.model,cohort=cohort),int(tic),targets[0])
            frames.append(model.attach_references(g,int(tic)) if args.command=='predict-hgb' else model.attach_references(g))
        f=pd.concat(frames,ignore_index=True)
    f['correction']=model.predict(f)
    if args.command=='predict-cnf':f['pred_knn']=f[model.prior_column]
    if 'has_gaia_anchor' not in f:f['has_gaia_anchor']=np.isfinite(f.gaiarp)&np.isfinite(f.bp_rp)
    if 'systematic_score' not in f:f['systematic_score']=np.nan
    f['qualified']=f.systematic_score.gt(.95);f['missing_anchor_fallback']=~f.has_gaia_anchor
    columns=model.features if args.command=='predict-hgb' else model.c['feature_cols']
    f['imputed_feature_count']=(~np.isfinite(f[columns].to_numpy(float))).sum(axis=1)
    count='ref_count' if args.command=='predict-hgb' else model.reference_count_column
    f['reference_supported']=f[count].fillna(0).gt(0)
    if 'product_family' not in f:f['product_family']='unknown'
    f['experimental_product_transfer']=f.product_family.ne('tess-spoc')
    if args.command=='predict-hgb':
        for name,(lo,hi) in model.ranges.items():f[name+'_outside_training_range']=~f[name].between(lo,hi)
        f['science_supported']=f.reference_supported&f.has_gaia_anchor&~f.tmag_outside_training_range&~f.bp_rp_outside_training_range&~f.sector_outside_training_range&~f.experimental_product_transfer
    args.out.mkdir(parents=True);f.to_parquet(args.out/'predictions.parquet',index=False)
    if args.curves:
        c=pd.read_parquet(args.curves).drop(columns=['correction','corrected'],errors='ignore')
        c=c.merge(f[['tic_id','sector','correction']],on=['tic_id','sector'],validate='many_to_one',how='left')
        if c.correction.isna().any():raise ValueError('Unmatched cadence sectors')
        c['corrected']=c.raw/c.correction
        if 'raw_error' in c:c['corrected_error']=c.raw_error/c.correction
        c.to_parquet(args.out/'curves.parquet',index=False)
    coverage=[]
    for view,g in [('all',f),('qualified',f[f.qualified])]:
        coverage.append(dict(view=view,rows=len(g),stars=int(g.tic_id.nunique()),lost_rows=len(f)-len(g),missing_anchors=int((~g.has_gaia_anchor).sum()),unknown_scores=int(g.systematic_score.isna().sum()),missing_references=int((~g.reference_supported).sum()),unverified_product_transfer_rows=int(g.experimental_product_transfer.sum())))
    write_json(args.out/'provenance.json',dict(model=str(args.model),model_bundle_sha256=sha256(args.model/'bundle.json'),features_sha256=sha256(args.features),prepared_inputs=bool(args.prepared),coverage=coverage,error_policy='Formal cadence errors exclude extraction/calibration uncertainty',no_refitting=True))


def main():
    args=parser().parse_args()
    if args.threads<1:raise ValueError('Positive thread count required')
    os.environ['OMP_NUM_THREADS']=str(args.threads);os.environ['OPENBLAS_NUM_THREADS']=str(args.threads)
    if 'cnf' in args.command:
        import torch
        torch.set_num_threads(args.threads)
    if args.command=='collect':
        from .collection.collect import collect
        collect(args.requests,args.out,args.workers,args.resume,output_limit_gib=args.output_limit_gib)
    elif args.command=='features':
        import pandas as pd
        from .features.physical import from_collection
        if args.out.exists():raise FileExistsError('Use a new feature output path')
        f=from_collection(args.collection,args.catalog_dir,pd.read_parquet(args.thermal),prf_dir=args.prf_dir,strap_path=args.strap_path)
        args.out.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(args.out,index=False)
    elif args.command=='curves':
        import numpy as np
        import pandas as pd
        from .io import sha256
        if args.out.exists():raise FileExistsError('Use a new curve output path')
        frames=[]
        for receipt_path in sorted((args.collection/'receipts').glob('*.json')):
            r=json.loads(receipt_path.read_text());name=f"cadences/{r['source_row']}.parquet"
            if sha256(args.collection/name)!=r['output_sha256'][name]:raise ValueError('Cadence checksum changed')
            c=pd.read_parquet(args.collection/name);ok=c.quality.eq(0)&np.isfinite(c.time)&np.isfinite(c.pdcsap_flux)&c.pdcsap_flux.gt(0)
            c=c[ok].rename(columns={'pdcsap_flux':'raw','pdcsap_flux_err':'raw_error'});frames.append(c)
        if not frames:raise ValueError('No verified cadences')
        args.out.parent.mkdir(parents=True,exist_ok=True);pd.concat(frames,ignore_index=True).to_parquet(args.out,index=False)
    elif args.command=='prepare':
        from .reference.prepare import prepare
        prepare(args.metadata,args.labels,args.splits,args.out,args.family)
    elif args.command=='train-hgb':
        from .hgb.train import train
        train(args.prepared,args.out,args.identity_catalog,args.config)
    elif args.command=='train-cnf':
        from .cnf.train import train
        train(args.prepared,args.out,args.identity_catalog,args.epochs,args.target,args.batch_size)
    elif args.command.startswith('predict-'):predict(args)
    elif args.command=='dashboard':
        from .visualization.render import render
        print(json.dumps(render(args.predictions,args.curves,args.out,args.model_spec),indent=2))
    elif args.command=='verify-model':
        from .bundles import verify_bundle
        print(f'Verified {len(verify_bundle(args.model))} model assets')


if __name__=='__main__':main()
