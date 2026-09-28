import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("extract_spoc_trial", Path(__file__).with_name("extract_spoc_trial.py"))
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestSpocFeatureExtraction(unittest.TestCase):
    def test_aperture_bit_and_geometry(self):
        mask = np.zeros((3, 4), dtype=np.int32)
        mask[1, 1:3] = 2
        mask[0, 0] = 1  # collected pixel, but not in the optimal aperture
        got = MOD.aperture_features(mask)
        self.assertEqual(got["ap_npix"], 2)
        self.assertEqual(got["ap_width"], 2)
        self.assertEqual(got["ap_height"], 1)
        self.assertEqual(got["ap_perimeter"], 6)

    def test_robust_stats_reject_nonfinite_and_use_mad_scale(self):
        got = MOD.robust_stats(np.array([1.0, 2.0, 3.0, np.nan]), "x")
        self.assertEqual(got["x_median"], 2.0)
        self.assertAlmostEqual(got["x_mad"], 1.4826)


if __name__ == "__main__":
    unittest.main()
