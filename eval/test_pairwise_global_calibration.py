"""Small correctness tests for the global/pairwise target construction."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pairwise_global_calibration import star_center


class PairwiseTargetTests(unittest.TestCase):
    def test_invariant_to_arbitrary_star_brightness(self):
        ids = np.array([1, 1, 1, 2, 2])
        y = np.array([.1, .2, -.1, .4, .5])
        shifted = y + np.array([12., 12., 12., -7., -7.])
        np.testing.assert_allclose(star_center(y, ids),
                                   star_center(shifted, ids), atol=1e-14)

    def test_centered_residuals_equal_centered_pairwise_information(self):
        ids = np.array([1, 1, 1, 2, 2])
        y = np.array([1., 3., 8., -2., 5.])
        centered = star_center(y, ids)
        for tic in np.unique(ids):
            take = ids == tic
            # Every pair difference is preserved after centering.
            np.testing.assert_allclose(
                centered[take, None] - centered[take][None, :],
                y[take, None] - y[take][None, :])


if __name__ == "__main__":
    unittest.main()
