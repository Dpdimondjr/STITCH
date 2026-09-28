import unittest
import numpy as np
import pandas as pd
from background_bias_diagnostic import susceptibility, partial_slope, center


class BackgroundDiagnosticTests(unittest.TestCase):
    def test_susceptibility_does_not_consume_target_flux(self):
        f = pd.DataFrame(dict(tmag=[10.,12.],ap_npix=[10.,20.],
            crowdsap=[.9,.8],flfrcsap=[.8,.7],pdcsap_median=[1.,2.]))
        a = susceptibility(f)
        f.pdcsap_median *= 100
        np.testing.assert_equal(a,susceptibility(f))
        f.ap_npix *= 2
        np.testing.assert_allclose(2*a,susceptibility(f))

    def test_fixed_effects_remove_star_and_sector_confounding(self):
        rng = np.random.default_rng(7)
        star = np.repeat(np.arange(40),8); sec = np.tile(np.arange(8),40)
        x = rng.uniform(.2,2,len(star))
        f = pd.DataFrame(dict(tic_id=star,sector=sec,camera=1,ccd=1,
            tmag=np.repeat(rng.uniform(8,13,40),8),crowdsap=rng.uniform(.8,1,len(star)),
            flfrcsap=rng.uniform(.7,1,len(star)),mom_col=rng.uniform(0,2000,len(star)),
            mom_row=rng.uniform(0,2000,len(star)),susceptibility=x*.001))
        f['signed_residual'] = .025*x + star*.1 + sec*.2 + .03*f.crowdsap
        result = partial_slope(f,draws=30)
        self.assertAlmostEqual(result['slope_fraction_per_1e_minus3_susceptibility'],.025,places=8)

    def test_center_invariant_to_star_flux_scale(self):
        ids = [1,1,1,2,2,2]; y=np.array([1.,2.,3.,4.,5.,6.])
        np.testing.assert_allclose(center(y,ids),center(y+[3,3,3,-2,-2,-2],ids))


if __name__=='__main__': unittest.main()
