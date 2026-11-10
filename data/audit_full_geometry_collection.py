"""Outcome-independent inventory and storage feasibility for full STITCH geometry."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from hgb_reference_repair import sha256, write_json

ROOT = Path('eval/full_geometry_collection_20261012')
STUDY = Path('eval/geometry_training_transfer_20261012')
SOURCE = Path('eval/hgb_reference_repair_20261010/clean_features.parquet')
SCORES = Path('eval/gaia_selection_20261011/tars_scores.parquet')
FIELDS = ['gaiarp', 'gaiabp', 'bp_rp', 'crowdsap', 'flfrcsap', 'teff',
          'pr_wght2', 'pdc_noi', 'pdc_corp', 'pdc_totp', 'pdcvar',
          'median_sap_bkg', 'p90_sap_bkg', 'bkg_rms', 'scatter_flag_frac',
          'jitter_rms', 'perr_col', 'perr_row', 'focal_plane_temp',
          'focal_plane_temp_std', 'contratio', 'numcont']


def audit():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT / 'activation_protocol.json').exists():
        assert not (STUDY / 'results.json').exists()
        write_json(ROOT / 'activation_protocol.json', dict(
            created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
            user_authorization='Collect the full dataset and missing information if the frozen geometry transfer works.',
            activation='The calibration-selected primary head and its unchanged full-bank secondary transfer each have a positive qualified median-CV gain with paired 95% lower bound above zero; scientific verification passes.',
            selection='No choice of head from evaluation outcomes; all-sector results and harms reported even if qualified results pass.',
            acquisition='Every retained source_row, preserving role/TIC/sector/split and TARS status. Geometry/cadence metadata acquisition uses no flux outcome ranking.',
            storage='Keep historical files. Reuse verified caches. Store lossless packed headers/masks and compact catalog/provenance; stream cadence data without retaining full raw light curves. Stop new downloads below a 1 GiB free-space reserve.',
            scope='Exact product geometry and scene/strap response inputs; missing FITS metadata, background and pointing statistics; report unavailable fields and pixel-level information outside this compact collection.',
            source_sha256=sha256(SOURCE), study_freeze_sha256=sha256(STUDY / 'freeze.json')))
    f = pd.read_parquet(SOURCE, columns=['source_row', 'tic_id', 'sector', 'role',
        'reference_fold', 'cam', 'ccd', 'col', 'row', 'ra', 'dec', 'tmag'] + FIELDS)
    scores = pd.read_parquet(SCORES)
    f = f.merge(scores, on=['tic_id', 'sector'], how='left', validate='one_to_one')
    assert f.source_row.is_unique and not f.duplicated(['tic_id', 'sector']).any()
    assert f.groupby('tic_id').role.nunique().max() == 1
    manifest_cols = ['source_row', 'tic_id', 'sector', 'role', 'reference_fold',
                     'cam', 'ccd', 'col', 'row', 'ra', 'dec', 'tmag', 'gaiarp',
                     'gaiabp', 'bp_rp', 'systematic_score']
    f[manifest_cols].to_parquet(ROOT / 'manifest.parquet', index=False, compression='zstd')
    coverage = []
    attrition = []
    for role, g in f.groupby('role'):
        for view in ['all', 'qualified']:
            v = g if view == 'all' else g[g.systematic_score.gt(.95)]
            attrition.append(dict(role=role, view=view, observations=len(v),
                stars_any=int(v.tic_id.nunique()), metric_stars=int(v.groupby('tic_id').sector.nunique().ge(2).sum()),
                score_missing=int(v.systematic_score.isna().sum())))
            for field in FIELDS:
                valid = np.isfinite(v[field].to_numpy(float))
                coverage.append(dict(role=role, view=view, field=field,
                    observations=len(v), available=int(valid.sum()), missing=int((~valid).sum())))
    pd.DataFrame(coverage).to_csv(ROOT / 'existing_field_coverage.csv', index=False)
    pd.DataFrame(attrition).to_csv(ROOT / 'cohort_attrition.csv', index=False)
    recovered = pd.read_parquet('eval/calibration_bank_study_20261011/header_recovery/cached_headers.parquet')
    z = f.merge(recovered, on=['tic_id', 'sector'], how='left', suffixes=('', '_recovered'), validate='one_to_one')
    recovery = []
    for field in recovered.columns.difference(['tic_id', 'sector']):
        for role, g in z.groupby('role'):
            old = g[field].notna(); new = old | g[field+'_recovered'].notna()
            recovery.append(dict(role=role, field=field, observations=len(g),
                available_before=int(old.sum()), available_with_legacy_recovery=int(new.sum()),
                missing_after=int((~new).sum()), limitation='Legacy cached metadata lack exact product/version provenance.'))
    pd.DataFrame(recovery).to_csv(ROOT / 'legacy_header_recovery_coverage.csv', index=False)
    p = pd.read_parquet(STUDY / 'products.parquet')
    estimate = dict(sample_observations=len(p), sample_raw_bytes=int(p.size_bytes.sum()),
        projected_full_raw_lc_gib=float(p.size_bytes.mean()*len(f)/2**30),
        caveat='Projection from frozen 1000-TIC study, not an archive census.')
    write_json(ROOT / 'audit.json', dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        rows=len(f), stars=int(f.tic_id.nunique()), sectors=sorted(int(x) for x in f.sector.unique()),
        manifest_sha256=sha256(ROOT / 'manifest.parquet'), source_sha256=sha256(SOURCE),
        scores_sha256=sha256(SCORES), script_sha256=sha256(__file__),
        attrition=attrition, raw_lightcurve_storage=estimate,
        disk_free_gib=shutil.disk_usage(ROOT).free/2**30, outcomes_loaded=False))
    print(json.dumps({'rows':len(f), 'stars':int(f.tic_id.nunique()), **estimate}), flush=True)


def compression():
    """Lossless storage benchmark; sampled products remain saved and reproducible."""
    sample = pd.read_parquet(STUDY / 'geometry_inventory.parquet').sample(1024, random_state=20261012)
    rows=[]; original=0
    for r in sample.itertuples():
        path=Path(r.compact_path);original+=path.stat().st_size
        with np.load(path) as z:
            d={k:str(z[k]) for k in ['primary','lightcurve','aperture_header','data_uri','filename','etag','prefix_sha256','aperture_sha256']}
            a=z['aperture'];d.update(source_row=r.source_row,tic_id=r.tic_id,sector=r.sector,
                aperture_bytes=a.tobytes(),aperture_dtype=a.dtype.str,
                aperture_height=a.shape[0],aperture_width=a.shape[1],
                size_bytes=int(z['size_bytes']),bytes_transferred=int(z['bytes_transferred']))
            rows.append(d)
    t=pd.DataFrame(rows).sort_values(['tic_id','sector'])
    output=ROOT/'lossless_storage_sample.parquet'
    t.to_parquet(output,index=False,compression='zstd',compression_level=9)
    reread=pd.read_parquet(output)
    pd.testing.assert_frame_equal(t.reset_index(drop=True),reread.reset_index(drop=True))
    audit=json.loads((ROOT/'audit.json').read_text())
    result=dict(observations=len(t),original_npz_bytes=original,
        lossless_packed_bytes=output.stat().st_size,
        projected_full_packed_gib=output.stat().st_size/len(t)*audit['rows']/2**30,
        max_memory_batch_rows=1024,exact_roundtrip=True,sha256=sha256(output))
    write_json(ROOT/'storage_benchmark.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    audit();compression()
