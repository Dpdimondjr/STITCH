"""Numerical checks for the comparison's reference and scatter calculations."""
import unittest
import numpy as np
import pandas as pd
from value_benchmark import attach_references, static_surface, cv
from checkpoint_value_benchmark import historical_features


class BenchmarkChecks(unittest.TestCase):
    def setUp(self):
        self.ref = pd.DataFrame(dict(tic_id=range(5),sector=[1]*5,cam=[1]*5,
            ccd=[1]*5,col=[100,200,300,400,500],row=[100]*5,
            loo=[1.02]*5,bm=[1.03]*5,gaia=[1.04]*5))
        self.query = pd.DataFrame(dict(tic_id=[99,99],sector=[1,2],cam=[1,1],
            ccd=[1,1],col=[150,150],row=[100,100],sector_median=[100,110]))

    def test_reference_features_do_not_use_subject_flux(self):
        baseline = attach_references(self.ref,self.query,20)
        changed = self.query.copy()
        changed.sector_median *= [2,5]
        injected = attach_references(self.ref,changed,20)
        pd.testing.assert_series_equal(baseline.spatial_knn_mean_loo,
                                       injected.spatial_knn_mean_loo)
        self.assertAlmostEqual(baseline.loc[0,'knn_loo_median'],1.02)
        self.assertEqual(baseline.reference_count.tolist(),[5,0])
        self.assertTrue(pd.isna(baseline.loc[1,'spatial_knn_mean_loo']))

    def test_constant_detector_surface(self):
        np.testing.assert_allclose(static_surface(self.ref,self.query,'bm',2),1.03)

    def test_relative_scatter_and_exact_correction(self):
        np.testing.assert_allclose(cv(self.query,np.ones(2)),
                                   cv(self.query,np.full(2,2)))
        self.assertLess(cv(self.query,np.array([1,1.1])).iloc[0],1e-12)

    def test_mean_and_median_reference_outlier_response(self):
        self.ref.loc[0,'loo'] = 10
        result = attach_references(self.ref,self.query,20)
        self.assertGreater(result.loc[0,'spatial_knn_mean_loo'],2)
        self.assertAlmostEqual(result.loc[0,'knn_loo_median'],1.02)

    def test_historical_ties_are_audited_and_baseline_excludes_self(self):
        self.ref.loc[0,'loo'] = 10
        self.ref.loc[1,'col'] = self.ref.loc[0,'col']
        result, errors = historical_features(self.ref,'loo')
        self.assertGreater(errors,0)
        self.assertAlmostEqual(result.loc[0,'knn_loo_corrected_mean'],1.02)
        self.assertAlmostEqual(result.loc[0,'knn_loo_median'],1.02)


if __name__ == '__main__':
    unittest.main()
