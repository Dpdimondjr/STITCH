"""Apply the frozen Gaia-RP HGB without experiment-directory dependencies."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ..io import sha256
from ..bundles import verify_bundle

def validate_reference_bank(bank, cohort):
    if not bank.source_row.is_unique or not cohort.source_row.is_unique:
        raise ValueError('Duplicate source rows in reference provenance')
    joined = bank.merge(cohort, on='source_row', suffixes=('', '_cohort'),
                        how='left', validate='one_to_one')
    if not joined.role.eq('train').all() or not joined.fit_row.eq(True).all():
        raise ValueError('Reference bank contains non-fitting or non-training rows')
    if not joined.systematic_score.gt(.95).all():
        raise ValueError('Reference bank contains unqualified sectors')
    for name in ['tic_id', 'physical_group_id', 'reference_fold', 'sector']:
        if not joined[name].astype(str).eq(joined[name + '_cohort'].astype(str)).all():
            raise ValueError(f'Reference provenance mismatch: {name}')
    forbidden = set(cohort.loc[~cohort.role.eq('train'), 'physical_group_id'].astype(str))
    if forbidden & set(bank.physical_group_id.astype(str)):
        raise ValueError('Reference bank overlaps validation/test physical groups')


class FrozenPhysicalModel:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = verify_bundle(self.root)
        self.manifest = json.loads((self.root / 'selected_model.json').read_text())
        if (self.manifest['selected'] != 'physical' or
                self.manifest['target'] != 'gaia_rp' or
                self.manifest['rounds'] != 5 or self.manifest['beta'] != 1 or
                not self.manifest['external_anchor_retained']):
            raise ValueError('Expected the Gaia-anchored physical model')
        self.features = self.manifest['features']
        core = self.root / self.manifest['core_directory']
        for name, digest in self.manifest['model_sha256'].items():
            if sha256(core / name) != digest:
                raise ValueError(f'Model manifest mismatch: {name}')
        self.imputer = joblib.load(core / 'imputation.pkl')
        self.models = [joblib.load(core / f'round_{i}.pkl')
                       for i in range(self.manifest['rounds'])]
        self.bank = pd.read_parquet(self.root / 'reference_bank.parquet')
        self.cohort = pd.read_parquet(self.root / 'cohort_manifest.parquet')
        validate_reference_bank(self.bank, self.cohort)
        self.ranges = json.loads((self.root / 'support.json').read_text())['ranges']

    def predict(self, frame):
        absent = set(self.features + ['pred_knn']) - set(frame.columns)
        if absent:
            raise ValueError(f'Missing feature columns (do not silently invent them): {sorted(absent)}')
        prior = frame.pred_knn.to_numpy(float)
        if not np.all(np.isfinite(prior) & (prior > 0)):
            raise ValueError('Invalid reference correction')
        x = frame[self.features].apply(pd.to_numeric, errors='raise')
        x = x.replace([np.inf, -np.inf], np.nan).fillna(self.imputer).to_numpy(np.float32)
        logfactor = np.log(prior) + sum(model.predict(x) for model in self.models)
        factor = np.exp(logfactor)
        if not np.all(np.isfinite(factor) & (factor > 0)):
            raise ValueError('Invalid predicted correction')
        return factor

    def check_new_identity(self, tic, target):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        if str(target['disposition']) in ['DUPLICATE', 'ARTIFACT']:
            raise ValueError('Target catalog marks this source as duplicate/artifact')
        catalog = pd.read_parquet(self.root / 'identity_catalog.parquet',
                                  columns=['ID', 'GAIA', 'ra', 'dec'])
        separation = SkyCoord(catalog.ra.to_numpy() * u.deg, catalog.dec.to_numpy() * u.deg).separation(
            SkyCoord(float(target['ra']) * u.deg, float(target['dec']) * u.deg)).arcsec
        gaia = str(target['GAIA'])
        collision = catalog.ID.astype(str).eq(str(tic)).to_numpy() | (separation < 2)
        if gaia not in ['', '--', 'nan', 'None']:
            collision |= catalog.GAIA.astype(str).eq(gaia).to_numpy()
        if np.any(collision) or int(tic) in set(self.cohort.tic_id):
            raise ValueError(f'TIC {tic} overlaps the historical physical-source cohort; use saved historical predictions')
        return dict(tic_id=int(tic), gaia_id=gaia, cohort_collision=False,
                    nearest_cohort_arcsec=float(np.nanmin(separation)),
                    policy='Same TIC, Gaia or <2 arcsec; reject all cohort matches before frozen color application')

    def attach_references(self, frame, tic):
        from ..features import context as rebuild
        from ..reference.safe import hgb_features as reference_features
        from ..features.context import color_design
        if int(tic) in set(self.cohort.tic_id):
            raise ValueError('New-source preparation cannot use a historical cohort TIC')
        fold = rebuild.hash_bucket(str(tic), 5, 'reference')
        groups = set(frame.physical_group_id.astype(str))
        ref = self.bank[self.bank.reference_fold.ne(fold) &
                        ~self.bank.tic_id.eq(int(tic)) &
                        ~self.bank.physical_group_id.astype(str).isin(groups)]
        f = frame.drop(columns=rebuild.REFCOLS, errors='ignore').reset_index(drop=True).copy()
        f = pd.concat([f, reference_features(ref, f)], axis=1)
        f['reference_fold'] = fold
        good = np.isfinite(f.gaiarp) & np.isfinite(f.bp_rp)
        f['has_gaia_anchor'] = good
        f['gaia_log_anchor'] = 0.
        color = joblib.load(self.root / f'color_fold_{fold}.pkl')
        if good.any():
            f.loc[good, 'gaia_log_anchor'] = (-.4 * np.log(10) * (f.loc[good, 'gaiarp'] - 10) +
                                            color.predict(color_design(f.loc[good, 'bp_rp'])))
        # Missing anchors use frozen input imputation/reference prediction only;
        # the query flux never supplies an anchor or an inference-time target.
        f['missing_anchor_fallback'] = ~good
        return f

    def annotate(self, f, curves, scores, product_family):
        keys = ['source_row', 'tic_id', 'sector']
        if f.duplicated(keys).any() or f.duplicated(['tic_id', 'sector']).any():
            raise ValueError('Run each extraction family separately, one product per TIC/sector')
        columns = keys + ['physical_group_id', 'cam', 'ccd', 'col', 'row', 'tmag', 'bp_rp',
                          'ref_count', 'knn5_radius', 'knn50_radius', 'reference_fold',
                          'has_gaia_anchor', 'gaia_log_anchor', 'pred_knn']
        p = f[columns].copy()
        p['correction'] = self.predict(f)
        p['missing_anchor_fallback'] = ~p.has_gaia_anchor
        p['reference_supported'] = p.ref_count.fillna(0).gt(0)
        p['product_family'] = product_family
        p['experimental_product_transfer'] = product_family != 'tess-spoc'
        for name, (lo, hi) in self.ranges.items():
            p[name + '_outside_training_range'] = ~p[name].between(lo, hi)
        missing = ~np.isfinite(f[self.features].to_numpy(float))
        p['imputed_feature_count'] = missing.sum(axis=1)
        p['imputed_features'] = [';'.join(np.asarray(self.features)[row]) for row in missing]
        p['systematic_score'] = np.nan
        if scores is not None:
            s = scores[['tic_id', 'sector', 'systematic_score']]
            if s.duplicated(['tic_id', 'sector']).any():
                raise ValueError('Ambiguous TARS scores')
            p = p.drop(columns='systematic_score').merge(s, on=['tic_id', 'sector'], how='left', validate='one_to_one')
        p['qualified'] = p.systematic_score.gt(.95)
        p['qualification_status'] = np.where(p.systematic_score.isna(), 'unknown',
                                             np.where(p.qualified, 'qualified', 'nonqualified'))
        p['qualification_provenance'] = ('primary_TARS_catalog' if product_family == 'tess-spoc'
                                          else 'joined_primary_TARS_not_native_certification')
        p['science_supported'] = (p.reference_supported & p.has_gaia_anchor &
                                   ~p.tmag_outside_training_range & ~p.bp_rp_outside_training_range &
                                   ~p.sector_outside_training_range & ~p.experimental_product_transfer)
        c = curves.drop(columns=['correction', 'corrected', 'corrected_error', 'qualified',
                                 'qualification_status', 'science_supported'], errors='ignore').merge(
            p[keys + ['correction', 'qualified', 'qualification_status', 'science_supported']],
            on=keys, how='left', validate='many_to_one')
        if c.correction.isna().any() or c.duplicated(['tic_id', 'sector', 'time']).any():
            raise ValueError('Unmatched or duplicate cadence keys')
        c['corrected'] = c.raw / c.correction
        if 'raw_error' in c:
            c['corrected_error'] = c.raw_error / c.correction
        medians = c.groupby(keys).raw.median().rename('raw_median').reset_index()
        p = p.merge(medians, on=keys, validate='one_to_one')
        p['corrected_median'] = p.raw_median / p.correction
        return p, c
