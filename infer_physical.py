"""Apply the frozen Gaia-RP physical HGB to locally acquired exact FITS products.

Example (run from any directory; relative arguments use the caller's cwd):
  python3 infer_physical.py --products study/products.parquet \
    --catalog-dir study --tars study/full_tars_lookup.parquet --out new_run

Catalog directory contains <TIC>_cone.ecsv. Product manifest columns:
tic_id, sector, path, sha256. Paths are relative to the repository root.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from stitch.frozen import REPO, DEFAULT_MODEL, FrozenPhysicalModel, legacy_imports, sha256, verify_inventory


def run(products_path, catalog_dir, out, tars=None, tics=None, product_family='tess-spoc', model_dir=DEFAULT_MODEL):
    import numpy as np
    import pandas as pd
    from astropy.table import Table
    from astropy.io import fits
    legacy_imports()
    from stitch._physical_features import physical_rows, PRF
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {out}')
    model = FrozenPhysicalModel(model_dir)
    products = pd.read_parquet(products_path)
    if tics:
        products = products[products.tic_id.isin(tics)].copy()
        if set(tics) - set(products.tic_id):
            raise ValueError('Requested TIC absent from products manifest')
    if products.empty or products.duplicated(['tic_id', 'sector']).any():
        raise ValueError('Require nonempty unique TIC/sector products from one family')
    scores = pd.read_parquet(tars) if tars else None
    thermal = pd.read_parquet(model.root / 'features.parquet', columns=[
        'sector', 'cam', 'ccd', 'focal_plane_temp', 'focal_plane_temp_std'])
    out.mkdir(parents=True)
    provenance = dict(model_dir=str(model.root), model_sha256=model.hashes,
                      products_sha256=sha256(products_path), product_family=product_family,
                      tars_sha256=sha256(tars) if tars else None,
                      no_training=True, metadata_context='all supplied sectors; transductive',
                      error_policy='Scaled formal cadence errors exclude calibration/extraction uncertainty',
                      support_policy='science_supported is a basic range/reference screen, not validated accuracy')
    (out / 'protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')
    predictions, attrition, identities = [], [], []
    for tic, group in products.groupby('tic_id', sort=True):
        tic = int(tic)
        path = Path(catalog_dir) / f'{tic}_cone.ecsv'
        table = Table.read(path)
        targets = table[np.asarray(table['ID']).astype(str) == str(tic)]
        if len(targets) != 1:
            raise ValueError(f'Expected one exact catalog identity for TIC {tic}')
        identities.append(dict(**model.check_new_identity(tic, targets[0]), catalog_sha256=sha256(path)))
        valid = []
        for p in group.sort_values('sector').itertuples():
            if sha256(p.path) != p.sha256:
                raise ValueError(f'Product hash mismatch: {p.path}')
            with fits.open(p.path, memmap=False) as h:
                if int(h[0].header['TICID']) != tic or int(h[0].header['SECTOR']) != p.sector:
                    raise ValueError(f'FITS identity mismatch: {p.path}')
                primary = 'hlsp_tess-spoc_' in Path(p.path).name
                if primary != (product_family == 'tess-spoc'):
                    raise ValueError(f'Product family disagrees with filename: {p.path}')
                d = h[1].data
                good = (d['QUALITY'] == 0) & np.isfinite(d['TIME']) & np.isfinite(d['PDCSAP_FLUX']) & (d['PDCSAP_FLUX'] > 0)
                if product_family == 'spoc120':
                    seconds = float(h[1].header.get('TIMEDEL', np.nan)) * 86400
                    if not np.isclose(seconds, 120, atol=1):
                        raise ValueError(f'Expected standard 120-second integration: {seconds}')
                attrition.append(dict(tic_id=tic, sector=int(p.sector), total_cadences=len(d),
                                      accepted_cadences=int(good.sum()), rejected_cadences=int((~good).sum()),
                                      status='retained' if good.any() else 'no_valid_cadences'))
                if good.any():
                    valid.append(p._asdict())
        if not valid:
            continue
        old_prf = PRF.TESS_PRF
        try:
            frame, curves = physical_rows(pd.DataFrame(valid), table, targets[0], thermal, tic)
        finally:
            PRF.TESS_PRF = old_prf
        curves['tic_id'] = tic
        frame = model.attach_references(frame, tic)
        p, c = model.annotate(frame, curves, scores, product_family)
        target_out = out / str(tic)
        target_out.mkdir()
        frame.to_parquet(target_out / 'features.parquet', index=False)
        p.to_parquet(target_out / 'predictions.parquet', index=False)
        c.to_parquet(target_out / 'curves.parquet', index=False)
        predictions.append(p)
        print(f'TIC {tic}: {len(p)} sectors, {len(c):,} cadences', flush=True)
    all_predictions = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    all_predictions.to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(attrition).to_csv(out / 'cadence_attrition.csv', index=False)
    products.to_parquet(out / 'input_products.parquet', index=False)
    (out / 'identity_audit.json').write_text(json.dumps(identities, indent=2) + '\n')
    summary = []
    for tic, p in all_predictions.groupby('tic_id') if len(all_predictions) else []:
        for view, mask in [('all', np.ones(len(p), bool)), ('qualified', p.qualified.to_numpy())]:
            z = p[mask]
            summary.append(dict(tic_id=int(tic), view=view, sectors=len(z),
                                lost_sectors=len(p)-len(z), unknown_TARS=int(p.systematic_score.isna().sum()),
                                missing_anchor=int(z.missing_anchor_fallback.sum()),
                                missing_references=int((~z.reference_supported).sum()),
                                outside_basic_support=int((~z.science_supported).sum())))
    pd.DataFrame(summary).to_csv(out / 'coverage.csv', index=False)
    verify_inventory(model.root)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for source in [Path(__file__), REPO/'stitch/frozen.py', REPO/'stitch/_physical_features.py']:
        shutil.copy2(source, snapshot/source.name)
    receipts = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
    (out / 'complete.json').write_text(json.dumps(dict(
        rows=len(all_predictions), source_count=len(predictions), archived_artifacts_verified=len(model.hashes),
        missing_anchor_rows=int(all_predictions.missing_anchor_fallback.sum()) if len(all_predictions) else 0,
        artifact_sha256=receipts, no_retraining=True), indent=2) + '\n')
    return all_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--products', type=Path, required=True)
    parser.add_argument('--catalog-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tars', type=Path)
    parser.add_argument('--tics', type=int, nargs='+')
    parser.add_argument('--product-family', choices=['tess-spoc', 'spoc120'], default='tess-spoc')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    for name in ['products', 'catalog_dir', 'out', 'tars', 'model_dir']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    os.chdir(REPO)
    run(args.products, args.catalog_dir, args.out, args.tars, args.tics, args.product_family, args.model_dir)


if __name__ == '__main__':
    main()
