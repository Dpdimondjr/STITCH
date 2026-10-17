"""Metadata context and fixed feature definitions; no brightness centering."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ..reference._hgb_math import SECTOR,SPATIAL,STRIPE
from ..io import log

DEV_SOURCE = ['crowdsap', 'jitter_rms', 'pdcvar', 'focal_plane_temp']


DEV = ['dev_crowdsap_loo', 'dev_jitter_rms_loo', 'dev_pdcvar_loo', 'dev_fptemp_loo']


PARAMS = dict(loss='absolute_error', learning_rate=.06, max_iter=250,
              max_leaf_nodes=1023, min_samples_leaf=40, l2_regularization=.1,
              early_stopping=False, verbose=0)


def per_star(frame, prediction):
    z = frame[['tic_id', 'sector', 'sector_median']].copy()
    z['flux'] = z.sector_median.to_numpy()/prediction
    g = z.groupby('tic_id')
    cv = 100*g.flux.std(ddof=0)/g.flux.mean()
    return cv[g.sector.nunique()>=2]


def group_median(values, ids):
    return pd.Series(values).groupby(pd.Series(ids)).transform('median').to_numpy()
def center(f,value):
    z=pd.Series(np.asarray(value),index=f.index)
    return (z-z.groupby(f.tic_id).transform('mean')).fillna(0).to_numpy()


def design(f,family):
    values=[f.capture_term,f.crowding_term,f.log_correction]
    names=['capture','crowding','baseline_strength']
    if family in ['scene_strap','geometry_context']:
        values.append(.05*f.overlap_0);names.append('strap_prf_overlap')
    if family=='geometry_context':
        color=f.bp_rp.fillna(1)-1
        values.extend([f.capture_term*color,f.crowding_term*color,.05*f.overlap_0*color,.05*f.area_fraction,.02*np.log(f.aperture_pixels.clip(lower=1))])
        names.extend(['capture_color','crowding_color','strap_color','strap_aperture_area','log_aperture_area'])
        values.extend([.05*f.neighbor_strap_ratio,.05*f.neighbor_aperture_ratio*f.overlap_0])
        names.extend(['neighbor_light_on_straps','neighbor_light_times_target_overlap'])
        for cam in [1,2,3]:
            values.append(.05*f.overlap_0*f.actual_cam.eq(cam));names.append(f'strap_camera_{cam}')
    return np.column_stack([np.clip(center(f,v),-.04,.04) for v in values]),names
def color_design(color):
    c=np.clip(np.asarray(color,float),-.6,3.5)-.9
    return np.column_stack([c,c*c,c*c*c])
SEED=2026101201


REFCOLS=SECTOR+SPATIAL+STRIPE+['pred_knn']


GEOMETRY=['capture_term','crowding_term','prf_capture','catalog_scene_crowding',
    'nominal_capture','nominal_crowding','overlap_0','neighbor_aperture_ratio',
    'neighbor_strap_ratio','area_fraction','aperture_pixels','sources',
    'omitted_expected_aperture_flux_relative_target','target_header_position_difference',
    'prf_grid_clamp_pixels']


CADENCE=['quality_zero_fraction','default_quality_fraction','n_cadences',
    'pos_corr1_median_default','pos_corr2_median_default','pos_corr1_rms_default',
    'pos_corr2_rms_default','mom_centr1_rms_default','mom_centr2_rms_default',
    'background_half_shift']


HEADER_MAP={'header_crowdsap':'crowdsap','header_flfrcsap':'flfrcsap','header_pdcvar':'pdcvar',
    'header_pr_wght2':'pr_wght2','header_pdc_noi':'pdc_noi','header_pdc_corp':'pdc_corp',
    'header_pdc_totp':'pdc_totp','header_cdpp1_0':'cdpp1_0','header_teff':'teff'}


CADENCE_MAP={'median_sap_bkg_default':'median_sap_bkg','p90_sap_bkg_default':'p90_sap_bkg',
    'bkg_rms_default':'bkg_rms','jitter_rms_default':'jitter_rms','scatter_flag_frac':'scatter_flag_frac'}


FLAGS=['physical_group_id','training_reference_role_conflict','systematic_score',
    'exact_product_identity','geometry_available','cadence_available','headers_available',
    'suspect_catalog_alias','training_reference_eligible','gaia_catalog_anchor_available']


GEOM_CENTERED=[f'geometry_centered_{i}' for i in range(14)]


PHYSICAL_DERIVED=['capture_log','crowding_log','log_background_rp','log_background_rms_rp',
    'log_background_p90_rp','background_half_shift_rp','log_aperture_pixels',
    'log_neighbor_aperture_ratio','log_neighbor_strap_ratio','log_omitted_catalog_light',
    'dev_capture_term','dev_crowding_term','dev_overlap_0','dev_area_fraction',
    'dev_log_aperture_pixels','dev_log_background_rp','dev_log_background_rms_rp',
    'dev_background_half_shift_rp','dev_pos_corr1_rms_default','dev_pos_corr2_rms_default',
    'dev_mom_centr1_rms_default','dev_mom_centr2_rms_default']


EXTRA=GEOMETRY+CADENCE+['flfrcsap']+PHYSICAL_DERIVED+GEOM_CENTERED


def hash_bucket(value,n,salt):
    import hashlib
    return int.from_bytes(hashlib.blake2b(f'{SEED}:{salt}:{value}'.encode(),digest_size=8).digest(),'big')%n


def deviations(f):
    """Recompute metadata-only LOO deviations using all retained observations."""
    groups=f.groupby('tic_id',sort=False).indices
    for source,dest in zip(DEV_SOURCE,DEV):
        result=np.zeros(len(f),np.float32)
        source_values=f[source].to_numpy(float)
        for indices in groups.values():
            n=len(indices)
            if n<3:continue
            v=source_values[indices]
            if np.isfinite(v).sum()<3:continue
            other=np.broadcast_to(v,(n,n)).copy();np.fill_diagonal(other,np.nan)
            med=np.nanmedian(other,axis=1);width=1.4826*np.nanmedian(np.abs(other-med[:,None]),axis=1)
            sd=np.nanstd(other,axis=1);width=np.where(width<1e-9,np.maximum(sd,1e-9),width)
            z=(v-med)/width;z[np.isfinite(other).sum(axis=1)<2]=0
            result[indices]=np.nan_to_num(z,nan=0,posinf=0,neginf=0)
        f[dest]=result
        log(f'Rebuilt within-star predictor {dest}')


def inner_baseline(ref):
    prediction=pd.Series(1.,index=ref.index)
    for _,g in ref.groupby(['sector','cam','ccd']):
        if len(g)<2:continue
        ids=g.physical_group_id.to_numpy();labels=g.flux_offset_loo.to_numpy()
        k=min(100+int(g.physical_group_id.value_counts().max()),len(g))
        _,ix=cKDTree(g[['col','row']].to_numpy(float)).query(g[['col','row']].to_numpy(float),k=k)
        ix=np.asarray(ix).reshape(len(g),-1)
        ok=ids[ix]!=ids[:,None];ok &= np.cumsum(ok,axis=1)<=100
        pred=np.nanmedian(np.where(ok,labels[ix],np.nan),axis=1)
        prediction.loc[g.index]=np.where(np.isfinite(pred),pred,1.)
    return prediction

BASE = ['col', 'row', 'delta_sub_col', 'delta_sub_row', 'sector', 'cam', 'ccd', 'tmag', 'crowdsap', 'cdpp1_0', 'pdcvar', 'jitter_rms', 'pdc_noi', 'pr_wght2', 'gaiarp', 'focal_plane_temp', 'focal_plane_temp_std', 'temp_x_col', 'temp_x_row', 'temp_x_crowdsap', 'pdc_corp', 'pdc_totp', 'pdctotp_x_crowdsap', 'teff', 'bkg_rms', 'scatter_flag_frac', 'n_sectors_obs', 'sector_mean_loo', 'sector_std_loo', 'sector_frac_extreme', 'knn5_mean', 'knn5_median', 'knn5_mad', 'knn5_radius', 'knn10_mean', 'knn10_median', 'knn10_mad', 'knn10_radius', 'knn20_mean', 'knn20_median', 'knn20_mad', 'knn20_radius', 'knn50_mean', 'knn50_median', 'knn50_mad', 'knn50_radius', 'ref_count', 'contratio', 'numcont', 'bp_rp', 'col_destripe_loo', 'row_destripe_loo', 'perr_col', 'perr_row', 'dev_crowdsap_loo', 'dev_jitter_rms_loo', 'dev_pdcvar_loo', 'dev_fptemp_loo']
FEATURES = BASE + EXTRA
