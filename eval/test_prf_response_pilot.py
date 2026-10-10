import unittest
import numpy as np
from astropy.io import fits
from scipy.special import ndtr
from prf_response_pilot import positions,response


class IntegratedGaussian:
    def locate(self,x,y,shape):
        xx=np.arange(shape[1]);yy=np.arange(shape[0])
        return np.outer(ndtr(yy+.5-y)-ndtr(yy-.5-y),
                        ndtr(xx+.5-x)-ndtr(xx-.5-x))


class PRFChecks(unittest.TestCase):
    def test_physical_wcs_uses_reference_pixel_and_scale(self):
        h=fits.Header()
        for key,value in dict(CTYPE1='RA---TAN',CTYPE2='DEC--TAN',CRPIX1=4.,CRPIX2=5.,
            CRVAL1=30.,CRVAL2=40.,CDELT1=-.005,CDELT2=.005,RA_OBJ=30.,DEC_OBJ=40.,
            WCSAXESP=2,CTYPE1P='RAWX',CTYPE2P='RAWY',CRPIX1P=2.,CRPIX2P=3.,
            CRVAL1P=100.,CRVAL2P=200.,CDELT1P=2.,CDELT2P=3.).items():h[key]=value
        np.testing.assert_allclose(positions(h),[3,4,104,206],atol=1e-8)

    def test_centered_capture_and_directional_derivative(self):
        mask=np.zeros((13,13),bool);mask[5:8,5:8]=True
        r=response(IntegratedGaussian(),mask,6,6,np.zeros(5),np.zeros(5))
        expected=(ndtr(1.5)-ndtr(-1.5))**2
        self.assertAlmostEqual(r['prf_capture'],expected)
        self.assertAlmostEqual(r['prf_dlogq_dx'],0,places=8)
        self.assertEqual(r['prf_motion_span'],0)
        shifted=response(IntegratedGaussian(),mask,7,6,np.zeros(5),np.zeros(5))
        self.assertLess(shifted['prf_capture'],r['prf_capture'])
        self.assertLess(shifted['prf_dlogq_dx'],0)

    def test_small_stamp_capture_is_not_renormalized(self):
        r=response(IntegratedGaussian(),np.ones((3,3),bool),1,1,np.zeros(5),np.zeros(5))
        self.assertLess(r['prf_capture'],.9)


if __name__=='__main__':unittest.main()
