import unittest
import numpy as np
import pandas as pd
from robust_reference_features import robust_features


class RobustReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.bank = pd.DataFrame(dict(tic_id=[1,2,3,4], sector=[1]*4,
            cam=[1]*4, ccd=[1]*4, col=[0,0,20,30], row=[0]*4,
            flux_offset_loo=[1.01,1.02,1.03,1.05]))

    def test_filtered_rows_keep_identity(self):
        bank = self.bank.drop(index=1)
        result = robust_features(bank, self.bank)
        np.testing.assert_allclose(result.neighbor_median, [1.04,1.03,1.03,1.02])

    def test_tied_positions_exclude_subject(self):
        self.bank.loc[0,'flux_offset_loo'] = 100
        result = robust_features(self.bank, self.bank, k=1)
        self.assertAlmostEqual(result.loc[0,'neighbor_median'],1.02)

    def test_all_subject_records_excluded(self):
        duplicate = self.bank.iloc[[0]].copy()
        duplicate.flux_offset_loo = 100
        bank = pd.concat([self.bank,duplicate],ignore_index=True)
        result = robust_features(bank,self.bank,k=20)
        self.assertEqual(result.loc[0,'neighbor_count'],3)
        self.assertAlmostEqual(result.loc[0,'neighbor_mean'],(1.02+1.03+1.05)/3)

    def test_query_labels_cannot_change_features(self):
        query = self.bank.copy(); query.tic_id += 100
        original = robust_features(self.bank,query)
        query.flux_offset_loo *= 100
        pd.testing.assert_frame_equal(original,robust_features(self.bank,query))

    def test_unsupported_group_and_singleton_do_not_use_subject(self):
        singleton = self.bank.iloc[[0]].copy()
        result = robust_features(singleton,singleton)
        self.assertEqual(result.neighbor_count.iloc[0],0)
        self.assertTrue(np.isnan(result.neighbor_median.iloc[0]))
        singleton.sector = 999
        result = robust_features(self.bank,singleton)
        self.assertTrue(np.isnan(result.ccd_mean.iloc[0]))


if __name__ == '__main__':
    unittest.main()
