"""Regression and leakage controls against archived real applications."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from stitch.frozen import FrozenPhysicalModel, REPO, validate_reference_bank


class FrozenApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FrozenPhysicalModel()
        cls.study = REPO/'eval/prospective_ground_validation_20261012'

    def test_all_570_archived_predictions_replay(self):
        count = 0
        for root in [self.study, self.study/'spoc_controls']:
            saved = pd.read_parquet(root/'predictions.parquet')
            for tic, p in saved.groupby('tic_id'):
                f = pd.read_parquet(root/str(tic)/'features.parquet')
                expected = p.set_index('sector').correction.reindex(f.sector).to_numpy()
                np.testing.assert_array_equal(self.model.predict(f), expected)
                count += len(f)
        self.assertEqual(count, 570)

    def test_query_flux_cannot_change_prediction(self):
        f = pd.read_parquet(self.study/'230384382/features.parquet')
        expected = self.model.predict(f)
        f['sector_median'] = np.linspace(1., 1e9, len(f))
        f['raw'] = 1e20
        np.testing.assert_array_equal(self.model.predict(f), expected)

    def test_missing_schema_rejected(self):
        f = pd.read_parquet(self.study/'230384382/features.parquet')
        with self.assertRaisesRegex(ValueError, 'Missing feature'):
            self.model.predict(f.drop(columns='pdcvar'))

    def test_validation_label_rejected(self):
        bank = self.model.bank.head(2).copy()
        cohort = self.model.cohort[self.model.cohort.source_row.isin(bank.source_row)].copy()
        cohort.loc[cohort.index[0], 'role'] = 'val'
        with self.assertRaisesRegex(ValueError, 'non-training'):
            validate_reference_bank(bank, cohort)

    def test_physical_group_collision_rejected(self):
        bank = self.model.bank.head(2).copy()
        cohort = self.model.cohort[self.model.cohort.source_row.isin(bank.source_row)].copy()
        collision = cohort.iloc[[0]].copy()
        collision['source_row'] = -1
        collision['role'] = 'test'
        cohort = pd.concat([cohort, collision], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'physical groups'):
            validate_reference_bank(bank, cohort)

    def test_new_tic_alias_of_historical_source_is_rejected(self):
        catalog = pd.read_parquet(REPO/'eval/full_geometry_collection_20261012/target_catalog.parquet',
                                  columns=['ID','GAIA','ra','dec'])
        target = catalog.dropna(subset=['ra','dec']).iloc[0].to_dict()
        target['disposition'] = ''
        with self.assertRaisesRegex(ValueError, 'historical physical-source cohort'):
            self.model.check_new_identity(999999999999, target)

    def test_missing_anchor_is_flagged_without_flux_fallback(self):
        f = pd.read_parquet(self.study/'230384382/features.parquet').head(2).copy()
        f['gaiarp'] = np.nan
        a = self.model.attach_references(f, 230384382)
        f['sector_median'] = 1e20
        b = self.model.attach_references(f, 230384382)
        self.assertTrue(a.missing_anchor_fallback.all())
        np.testing.assert_array_equal(self.model.predict(a), self.model.predict(b))

    def test_unsupported_rows_are_retained_and_flagged(self):
        f = pd.read_parquet(self.study/'230384382/features.parquet').head(1).copy()
        # A sector with no spatial references follows the documented fallback.
        f['ref_count'] = 0.
        f['gaiarp'] = np.nan
        f['has_gaia_anchor'] = False
        f['gaia_log_anchor'] = 0.
        curves = pd.DataFrame(dict(source_row=[int(f.source_row.iloc[0])]*2,
                                   tic_id=[230384382]*2, sector=[int(f.sector.iloc[0])]*2,
                                   time=[1.,2.], raw=[10.,12.], raw_error=[.1,.2]))
        p,c = self.model.annotate(f,curves,None,'tess-spoc')
        self.assertEqual(len(p),1)
        self.assertEqual(len(c),2)
        self.assertTrue(p.missing_anchor_fallback.iloc[0])
        self.assertFalse(p.reference_supported.iloc[0])
        self.assertFalse(p.science_supported.iloc[0])
        self.assertEqual(p.qualification_status.iloc[0],'unknown')
        np.testing.assert_allclose(c.corrected/c.raw, c.corrected_error/c.raw_error)


if __name__ == '__main__':
    unittest.main()
