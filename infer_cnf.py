"""Apply the frozen pdcsap-branch CNF102 (with per-sector sigma) and the point-source flat correction to
locally acquired exact FITS products.

Example (run from the repository root; the first line is needed once):
  python3 data/download_flat_models.py
  python3 infer_cnf.py --products study/products.parquet --catalog-dir study \
      --tars study/tars.parquet --tics 230384382 --out new_run

Inputs follow infer_physical.py: a product manifest (tic_id, sector, path, sha256; one TESS-SPOC
light-curve product per TIC/sector) and <catalog-dir>/<TIC>_cone.ecsv scene catalogs. Outputs:
  predictions.parquet   one row per sector: pred_knn, hgb_pdcsap, cnf102 (point), cnf102_sigma_pct,
                        16/84 and 2.5/97.5 percentile factors (model only), sigma_keep80/50 flags,
                        support flags, flat_phi_pct, flat_factor = exp(-phi/100),
                        total_factor = cnf102 x flat_factor
  <TIC>/curves.parquet  accepted cadences with raw, corrected = raw / total_factor, formal corrected_error,
                        and level_sigma = corrected x cnf102_log_sd (per-sector level uncertainty)
  <TIC>/features.parquet, coverage.csv, identity_audit.json, protocol.json, complete.json
The sigma is the uncertainty of the model's sector-level correction only; it excludes stellar variability
and the flat term. --no-flat-correction restores corrected = raw / cnf102, the behaviour before 2026-11-05.

--hgb-cnf102-mean (optional, not the default) replaces the CNF102 factor with the frozen, untrained blend
blend_factor = exp((ln hgb_pdcsap + ln cnf102) / 2), so total_factor = blend_factor x flat_factor
(eval/ensemble_truth_tests_20261106/REPORT.md, arm MEAN). The blend has no validated uncertainty:
level_sigma is NaN, and the cnf102_* sigma, percentile and sigma_keep columns describe CNF102 alone.
Without the flag, outputs are unchanged.
"""
import argparse
import json
import shutil
from pathlib import Path

from stitch.frozen import REPO, FrozenPhysicalModel, legacy_imports, sha256, verify_inventory
from stitch.cnf import PdcsapCNF
from stitch.flat import FlatCorrection


def run(products_path, catalog_dir, out, tars=None, tics=None, seed=None, flat_correction=True, hgb_cnf102_mean=False):
    import numpy as np
    import pandas as pd
    from astropy.table import Table
    legacy_imports()
    from stitch._physical_features import physical_rows, PRF
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {out}')
    flat = FlatCorrection() if flat_correction else None
    physical = FrozenPhysicalModel()
    cnf = PdcsapCNF()
    products = pd.read_parquet(products_path)
    if tics:
        products = products[products.tic_id.isin(tics)].copy()
        if set(tics) - set(products.tic_id):
            raise ValueError('Requested TIC absent from products manifest')
    if products.empty or products.duplicated(['tic_id', 'sector']).any():
        raise ValueError('Require nonempty unique TIC/sector products')
    scores = pd.read_parquet(tars) if tars else None
    thermal = pd.read_parquet(physical.root / 'features.parquet',
                              columns=['sector', 'cam', 'ccd', 'focal_plane_temp', 'focal_plane_temp_std'])
    out.mkdir(parents=True)
    protocol = dict(
        cnf=cnf.describe(), physical_model_dir=str(physical.root), physical_sha256=physical.hashes,
        products_sha256=sha256(products_path), tars_sha256=sha256(tars) if tars else None,
        seed=seed, no_training=True, metadata_context='all supplied sectors of a source; transductive',
        flat_correction=flat.describe() if flat else 'disabled',
        error_policy='corrected_error scales the formal cadence error; level_sigma is the sampled correction sigma; '
                     'neither includes stellar variability, the flat term or extraction error beyond the model')
    if hgb_cnf102_mean:
        protocol.update(
            correction_model='hgb_cnf102_mean: blend_factor = exp((ln hgb_pdcsap + ln cnf102) / 2), frozen models, no training',
            error_policy='corrected_error scales the formal cadence error; level_sigma is NaN because the blend has no '
                         'validated uncertainty; cnf102_* sigma columns describe CNF102 alone')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    predictions, attrition, identities = [], [], []
    for tic, group in products.groupby('tic_id', sort=True):
        tic = int(tic)
        cone = Path(catalog_dir) / f'{tic}_cone.ecsv'
        table = Table.read(cone)
        targets = table[np.asarray(table['ID']).astype(str) == str(tic)]
        if len(targets) != 1:
            raise ValueError(f'Expected one exact catalog identity for TIC {tic}')
        identities.append(dict(**physical.check_new_identity(tic, targets[0]), catalog_sha256=sha256(cone)))
        valid = []
        for p in group.sort_values('sector').itertuples():
            if sha256(p.path) != p.sha256:
                raise ValueError(f'Product hash mismatch: {p.path}')
            from astropy.io import fits
            with fits.open(p.path, memmap=False) as h:
                if int(h[0].header['TICID']) != tic or int(h[0].header['SECTOR']) != p.sector:
                    raise ValueError(f'FITS identity mismatch: {p.path}')
                d = h[1].data
                good = (d['QUALITY'] == 0) & np.isfinite(d['TIME']) & np.isfinite(d['PDCSAP_FLUX']) & (d['PDCSAP_FLUX'] > 0)
                attrition.append(dict(tic_id=tic, sector=int(p.sector), total_cadences=len(d), accepted_cadences=int(good.sum()),
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
        frame = physical.attach_references(frame, tic)   # fold, physical group, HGB-branch anchor (overwritten below)
        p, q = cnf.apply(frame, seed=seed if seed is not None else 20261102 + tic % 100000)
        p['flat_phi_pct'] = flat.phi_pct(p.cam, p.ccd, p.col, p.row) if flat else 0.
        p['flat_factor'] = np.exp(-p.flat_phi_pct / 100)
        if hgb_cnf102_mean:
            p['blend_factor'] = np.exp((np.log(p.hgb_pdcsap) + np.log(p.cnf102)) / 2)
            if (np.isfinite(p.cnf102) & ~np.isfinite(p.blend_factor)).any():
                raise ValueError(f'TIC {tic}: HGB factor missing where CNF102 exists; blend undefined')
            p['total_factor'] = p.blend_factor * p.flat_factor
        else:
            p['total_factor'] = p.cnf102 * p.flat_factor
        p['systematic_score'] = np.nan
        if scores is not None:
            s = scores.rename(columns={'TICID': 'tic_id'})[['tic_id', 'sector', 'systematic_score']].drop_duplicates(['tic_id', 'sector'])
            p = p.drop(columns='systematic_score').merge(s, on=['tic_id', 'sector'], how='left', validate='one_to_one')
        p['qualified'] = p.systematic_score.gt(.95)
        p['qualification_status'] = np.where(p.systematic_score.isna(), 'unknown', np.where(p.qualified, 'qualified', 'nonqualified'))
        curves['tic_id'] = tic
        c = curves.merge(p[['sector', 'cnf102', 'flat_factor', 'total_factor', 'cnf102_log_sd', 'qualified', 'science_supported']],
                         on='sector', how='left', validate='many_to_one')
        if c.cnf102.isna().any():
            raise ValueError('Unmatched cadence sectors')
        c['corrected'] = c.raw / c.total_factor
        c['corrected_error'] = c.raw_error / c.total_factor
        c['level_sigma'] = np.nan if hgb_cnf102_mean else c.corrected * c.cnf102_log_sd
        medians = c.groupby('sector').agg(raw_median=('raw', 'median'), corrected_median=('corrected', 'median')).reset_index()
        p = p.merge(medians, on='sector', validate='one_to_one')
        dest = out / str(tic)
        dest.mkdir()
        q.to_parquet(dest / 'features.parquet', index=False)
        p.to_parquet(dest / 'predictions.parquet', index=False)
        c.to_parquet(dest / 'curves.parquet', index=False)
        predictions.append(p)
        print(f'TIC {tic}: {len(p)} sectors, median sigma {p.cnf102_sigma_pct.median():.2f} %, '
              f'keep80 {int(p.sigma_keep80.sum())}/{len(p)}'
              + (' (CNF102 sigma; correction is the HGB/CNF102 blend)' if hgb_cnf102_mean else ''), flush=True)
    allp = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    allp.to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(attrition).to_csv(out / 'cadence_attrition.csv', index=False)
    products.to_parquet(out / 'input_products.parquet', index=False)
    (out / 'identity_audit.json').write_text(json.dumps(identities, indent=2) + '\n')
    rows = []
    for tic, p in (allp.groupby('tic_id') if len(allp) else []):
        for view, m in [('all', np.ones(len(p), bool)), ('qualified', p.qualified.to_numpy()),
                        ('qualified_sigma_keep80', (p.qualified & p.sigma_keep80).to_numpy())]:
            z = p[m]
            rows.append(dict(tic_id=int(tic), view=view, sectors=len(z), lost_sectors=len(p) - len(z),
                             median_sigma_pct=float(z.cnf102_sigma_pct.median()) if len(z) else np.nan,
                             missing_anchor=int((~z.has_gaia_anchor).sum()), missing_references=int((~z.reference_supported).sum()),
                             outside_basic_support=int((~z.science_supported).sum())))
    pd.DataFrame(rows).to_csv(out / 'coverage.csv', index=False)
    verify_inventory(physical.root)
    snap = out / 'source_snapshot'
    snap.mkdir()
    for s in [Path(__file__), REPO / 'stitch/cnf.py', REPO / 'stitch/flat.py', REPO / 'stitch/frozen.py', REPO / 'stitch/_physical_features.py']:
        shutil.copy2(s, snap / s.name)
    receipts = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
    (out / 'complete.json').write_text(json.dumps(dict(rows=len(allp), source_count=len(predictions),
                                                         artifact_sha256=receipts, no_retraining=True), indent=2) + '\n')
    return allp


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--products', required=True)
    a.add_argument('--catalog-dir', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--tars')
    a.add_argument('--tics', nargs='*', type=int)
    a.add_argument('--seed', type=int)
    a.add_argument('--no-flat-correction', action='store_true', help='apply the CNF102 factor only (behaviour before 2026-11-05)')
    a.add_argument('--hgb-cnf102-mean', action='store_true',
                   help='optional: use the frozen mean of log HGB and log CNF102 factors (no validated uncertainty; level_sigma NaN)')
    args = a.parse_args()
    run(args.products, args.catalog_dir, args.out, args.tars, args.tics, args.seed,
        flat_correction=not args.no_flat_correction, hgb_cnf102_mean=args.hgb_cnf102_mean)


if __name__ == '__main__':
    main()
