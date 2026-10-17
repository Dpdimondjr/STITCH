"""Export verified historical models without copying reports or training datasets."""
import argparse,json,shutil,importlib.metadata
from pathlib import Path
import pandas as pd
from stitch.io import sha256,write_json
from stitch.bundles import seal_bundle


def main():
    p=argparse.ArgumentParser()
    for name in ['hgb-source','cnf-checkpoint','cnf-results','cnf-reference','cnf-selection','identity-catalog','prf-dir','strap-path','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a new bundle directory')
    manifest=json.loads((a.hgb_source/'selected_model.json').read_text())
    for name,digest in manifest['model_sha256'].items():
        if sha256(a.hgb_source/'physical'/name)!=digest:raise ValueError(f'HGB checkpoint changed: {name}')
    selected=json.loads(a.cnf_selection.read_text())
    if sha256(a.cnf_checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Selected CNF checkpoint changed')
    a.out.mkdir(parents=True);h=a.out/'hgb';h.mkdir();(h/'physical').mkdir()
    for name in manifest['model_sha256']:shutil.copy2(a.hgb_source/'physical'/name,h/'physical'/name)
    for name in ['selected_model.json','reference_bank.parquet','cohort_manifest.parquet',*manifest['color_models']]:shutil.copy2(a.hgb_source/name,h/name)
    metadata=pd.read_parquet(a.hgb_source/'features.parquet',columns=['fit_row','tmag','bp_rp','sector','cam','ccd','focal_plane_temp','focal_plane_temp_std'])
    tr=metadata[metadata.fit_row]
    write_json(h/'support.json',dict(ranges={name:[float(tr[name].min()),float(tr[name].max())] for name in ['tmag','bp_rp','sector']}))
    metadata[['sector','cam','ccd','focal_plane_temp','focal_plane_temp_std']].drop_duplicates().to_parquet(h/'thermal.parquet',index=False)
    identity=pd.read_parquet(a.identity_catalog,columns=['ID','GAIA','ra','dec']);identity.to_parquet(h/'identity_catalog.parquet',index=False)
    environment={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','joblib','torch','zuko','astropy','pyarrow']}
    write_json(h/'environment.json',environment)
    seal_bundle(h,dict(kind='hgb',target='gaia_rp',release_designation='Frozen physical research model; historical default_model_promoted=false retained',original_model_manifest_sha256=sha256(a.hgb_source/'selected_model.json')))
    c=a.out/'cnf';c.mkdir();shutil.copy2(a.cnf_checkpoint,c/'model.pt');shutil.copy2(a.cnf_reference,c/'reference_bank.parquet');shutil.copy2(h/'cohort_manifest.parquet',c/'cohort_manifest.parquet');shutil.copy2(h/'identity_catalog.parquet',c/'identity_catalog.parquet')
    result=json.loads(a.cnf_results.read_text())
    write_json(c/'model.json',dict(target='historical_relative',beta=result['beta'],clip=[.85,1.15],estimator='exact conditional median',historical_checkpoint=True,selection='Selected among four auditable archived flows on eight calibration stars; not a global-best or Gaia-RP-trained claim',checkpoint_sha256=selected['checkpoint_sha256'],application_reference='Verified training-only bank intersected with archived pairwise training membership; held-out source groups excluded',training_reference_limit='Application reference repair cannot repair historical training preprocessing.'))
    shutil.copy2(h/'thermal.parquet',c/'thermal.parquet')
    write_json(c/'environment.json',environment)
    seal_bundle(c,dict(kind='cnf',target='historical_relative',archived_selection_sha256=sha256(a.cnf_selection),archived_results_sha256=sha256(a.cnf_results)))
    geom=a.out/'geometry';geom.mkdir();shutil.copytree(a.prf_dir,geom/'prf',ignore=shutil.ignore_patterns('.DS_Store','*.tmp','*.part'))
    shutil.copy2(a.strap_path,geom/'strap_columns.csv')
    seal_bundle(geom,dict(kind='geometry',description='Immutable PRF grids and strap-column geometry; independent of LC downloads'))
    print(a.out)


if __name__=='__main__':main()
