import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import spoc_cohort_experiment as cohort


class TestCohortExperiment(unittest.TestCase):
    def test_predictors_exclude_flux_response_and_scatter_diagnostics(self):
        cols = {c.lower() for values in cohort.GROUPS.values() for c in values}
        forbidden = {"pdcsap_median", "sap_flux", "cdpp", "pdcvar", "label", "log_flux"}
        self.assertTrue(cols.isdisjoint(forbidden))

    def test_stable_hash(self):
        self.assertEqual(cohort.stable_u01(123, "split"), cohort.stable_u01(123, "split"))
        self.assertNotEqual(cohort.stable_u01(123, "split"), cohort.stable_u01(124, "split"))

    def test_perfect_log_offset_removes_sector_scatter(self):
        frame = pd.DataFrame({"tic_id": [1, 1, 1, 2, 2, 2],
                              "pdcsap_median": [90., 100., 110., 180., 200., 220.]})
        pred = np.log(frame.pdcsap_median) - np.log(frame.pdcsap_median).groupby(frame.tic_id).transform("mean")
        _, corrected = cohort.star_metric(frame, pred.to_numpy())
        self.assertTrue(np.allclose(corrected, 0, atol=1e-12))


if __name__ == "__main__": unittest.main()
