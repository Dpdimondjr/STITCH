import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from pairwise_global_robust import robust_center

class RobustCenterTests(unittest.TestCase):
    def test_star_shift_invariance_with_outlier(self):
        ids=np.array([1,1,1,2,2,2])
        y=np.array([0.,1.,100.,-2.,0.,3.])
        shift=np.array([20.,20.,20.,-9.,-9.,-9.])
        np.testing.assert_allclose(robust_center(y,ids),robust_center(y+shift,ids))

if __name__=="__main__": unittest.main()
