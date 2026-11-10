"""Physical extraction with the October 12 application arithmetic preserved.

Internal adapter; run from the repository root with local PRF/strap assets.
All historical extraction modules and artifacts remain untouched.
"""
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
import full_physical_rebuild as rebuild
from fast_aperture_geometry import feature
from full_geometry_features import header_features, CachedPRF, PRF
from extract_geometry_training_transfer import STRAPS
from hgb_reference_repair import sha256
from fit_geometry_training_transfer import design
from collect_full_cadence_metadata import summarize

def physical_rows(products, table, target, thermal, tic):
    columns = pd.read_csv(STRAPS).imaging_column.to_numpy()
    PRF.TESS_PRF = CachedPRF
    rows, curves = [], []
    for i,p in products.reset_index(drop=True).iterrows():
        path = Path(p.path)
        assert sha256(path) == p.sha256
        with fits.open(path, memmap=False) as hdus:
            h = [hdus[j].header.copy() for j in range(3)]
            bits = hdus[2].data.copy()
            assert h[0]['TICID'] == tic and h[0]['SECTOR'] == p.sector
            record = dict(tic_id=tic, sector=int(p.sector),
                          primary=h[0].tostring(), lightcurve=h[1].tostring(),
                          aperture_header=h[2].tostring(), aperture_bytes=bits.tobytes(),
                          aperture_dtype=str(bits.dtype), aperture_height=bits.shape[0],
                          aperture_width=bits.shape[1])
            payload = path.read_bytes()
            if payload.startswith(b"\x1f\x8b"):
                payload = gzip.decompress(payload)
            cadence = summarize(payload, record, str(path))
            geometry = feature(*h, bits, tic, int(p.sector), table, columns)
            row = dict(source_row=i, tic_id=tic, sector=int(p.sector),
                       physical_group_id=str(tic), role='external_science', **geometry)
            row.update({c:cadence[c] for c in rebuild.CADENCE})
            row.update({dest:cadence[src] for src,dest in rebuild.CADENCE_MAP.items()})
            row.update({dest:header_features(h)[src] for src,dest in rebuild.HEADER_MAP.items()})
            for dest,src in [('median_sap_bkg','median_sap_bkg_default'),
                             ('p90_sap_bkg','p90_sap_bkg_default')]:
                row[dest] = cadence[src]
            row.update(cam=geometry['actual_cam'], ccd=geometry['actual_ccd'],
                       col=geometry['actual_col'], row=geometry['actual_row'],
                       tmag=float(h[0]['TESSMAG']), gaiarp=float(target['gaiarp']),
                       bp_rp=float(target['gaiabp'])-float(target['gaiarp']))
            for name in ['contratio','numcont']:
                row[name] = float(target[name]) if not np.ma.is_masked(target[name]) else np.nan
            # Legacy perr has no verified acquisition definition. Do not replace
            # it with POS_CORR or flux-weighted centroid errors.
            row['perr_col'] = row['perr_row'] = np.nan
            temp = thermal[(thermal.sector == p.sector) &
                           (thermal.cam == row['cam']) & (thermal.ccd == row['ccd'])]
            for name in ['focal_plane_temp','focal_plane_temp_std']:
                vals = temp[name].dropna().unique()
                assert len(vals) <= 1
                row[name] = float(vals[0]) if len(vals) else np.nan
            for axis,j in [('col',1),('row',2)]:
                row['delta_sub_'+axis] = (row[axis]+cadence[f'pos_corr{j}_median_default']) % 1.
            data = hdus[1].data
            valid = ((data['QUALITY'] == 0) & np.isfinite(data['TIME']) &
                     np.isfinite(data['PDCSAP_FLUX']) & (data['PDCSAP_FLUX'] > 0))
            curves.append(pd.DataFrame(dict(source_row=i, sector=int(p.sector),
                         time=data['TIME'][valid].astype(float),
                         raw=data['PDCSAP_FLUX'][valid].astype(float),
                         raw_error=data['PDCSAP_FLUX_ERR'][valid].astype(float),
                         sap=data['SAP_FLUX'][valid].astype(float))))
            rows.append(row)
    f = pd.DataFrame(rows)
    f['temp_x_col'] = f.focal_plane_temp*f.col/2200.
    f['temp_x_row'] = f.focal_plane_temp*f.row/2200.
    f['temp_x_crowdsap'] = f.focal_plane_temp*f.crowdsap
    f['pdctotp_x_crowdsap'] = f.pdc_totp*f.crowdsap
    f['n_sectors_obs'] = f.groupby('tic_id').sector.transform('nunique').astype(float)
    rebuild.deviations(f)
    rpflux = 10.**(-.4*(f.gaiarp.fillna(f.tmag)-10))
    for src,dest in [('median_sap_bkg','log_background_rp'),
                     ('bkg_rms','log_background_rms_rp'),
                     ('p90_sap_bkg','log_background_p90_rp')]:
        f[dest] = np.log1p(f[src].clip(lower=0)/rpflux)
    f['background_half_shift_rp'] = f.background_half_shift/rpflux
    f['capture_log'] = np.log(f.flfrcsap.clip(.05,1.5))
    f['crowding_log'] = np.log(f.crowdsap.clip(.05,1.5))
    f['log_aperture_pixels'] = np.log1p(f.aperture_pixels)
    f['log_neighbor_aperture_ratio'] = np.log1p(f.neighbor_aperture_ratio.clip(lower=0))
    f['log_neighbor_strap_ratio'] = np.log1p(f.neighbor_strap_ratio.clip(lower=0))
    f['log_omitted_catalog_light'] = np.log1p(f.omitted_expected_aperture_flux_relative_target.clip(lower=0))
    for name in [c[4:] for c in rebuild.PHYSICAL_DERIVED if c.startswith('dev_')]:
        f['dev_'+name] = f[name]-f.groupby('tic_id')[name].transform('median')
    f['log_correction'] = 0.
    matrix,_ = design(f,'geometry_context')
    for i,name in enumerate(rebuild.GEOM_CENTERED):
        f[name] = matrix[:,i]
    return f.drop(columns='log_correction'), pd.concat(curves,ignore_index=True)

