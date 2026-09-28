import unittest
import numpy as np
import pandas as pd
from matched_global_benchmark import references, per_star


class MatchedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.bank = pd.DataFrame([dict(tic_id=t,sector=s,cam=1,ccd=1,col=t,row=1,loo=1.02)
                                 for s in range(1,6) for t in range(5)])
        self.query = pd.DataFrame([dict(tic_id=99,sector=s,cam=1,ccd=1,col=2,row=1,
                                       sector_median=100+s,loo=1.) for s in range(1,6)])

    def test_reference_query_overlap_rejected(self):
        with self.assertRaises(ValueError):references(self.bank,self.bank)

    def test_query_flux_cannot_change_reference_features(self):
        a=references(self.bank,self.query)
        q=self.query.copy();q.sector_median *= 100;q.loo *= 20
        b=references(self.bank,q)
        for c in ['knn_mean','knn_median','knn_mad','ccd_mean']:
            np.testing.assert_allclose(a[c],b[c])
        np.testing.assert_allclose(a.knn_median,1.02)

    def test_metric_invariant_to_star_scale_and_exact_recovery(self):
        raw=per_star(self.query,np.ones(5))
        np.testing.assert_allclose(raw,per_star(self.query,np.full(5,2.)))
        self.assertLess(per_star(self.query,self.query.sector_median.to_numpy()/100).iloc[0],1e-12)

    def test_no_reference_support_is_not_filled_with_subject(self):
        q=self.query.copy();q.sector=999
        self.assertEqual(len(references(self.bank,q)),0)


if __name__=='__main__':unittest.main()
