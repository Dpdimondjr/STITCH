"""Tests for stitch/flat.py. Run from the repository root: python3 eval/test_flat_correction.py"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitch.flat import FlatCorrection

FROZEN_PHI = Path('eval/flat_imprint_20261105/phi.parquet')


def synthetic(folder, flat, cam=1, ccd=1, corrupt=False):
    """Write one flat in the MAST layout with a receipt."""
    name = f'tess2018323-{cam}-{ccd}-flat.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(flat.astype(np.float32), name='FLAT_DATA')]).writeto(folder / name)
    digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    (folder / 'receipts.json').write_text(json.dumps([dict(name=name, sha256='0' * 64 if corrupt else digest)]))


class FlatCorrectionTests(unittest.TestCase):
    def test_reproduces_every_frozen_phi(self):
        """The production module must equal the frozen recipe scored on 796,969 visits."""
        d = pd.read_parquet(FROZEN_PHI)
        got = FlatCorrection().phi_pct(d.cam, d.ccd, d.col, d.row)
        self.assertLess(np.abs(got - d.phi.to_numpy()).max(), 1e-9)

    def test_high_flat_raises_stellar_flux(self):
        """Where the flat reads high the star reads low, so the correction must raise its flux."""
        flat = np.ones((200, 200))
        flat[100, 100] = 1.05
        with tempfile.TemporaryDirectory() as t:
            synthetic(Path(t), flat)
            c = FlatCorrection(t)
            on = c.phi_pct([1], [1], [101.], [101.])[0]      # FITS positions are 1-based: index 100 is position 101
            off = c.phi_pct([1], [1], [141.], [141.])[0]
        self.assertGreater(on, .5)
        self.assertLess(abs(off), .02)
        self.assertGreater(100 * np.exp(on / 100), 100)

    def test_uniform_flat_is_no_correction(self):
        with tempfile.TemporaryDirectory() as t:
            synthetic(Path(t), np.ones((100, 100)))
            self.assertAlmostEqual(FlatCorrection(t).phi_pct([1], [1], [50.], [50.])[0], 0., places=6)

    def test_missing_files_name_the_download_script(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(FileNotFoundError, 'download_flat_models'):
                FlatCorrection(t)

    def test_checksum_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as t:
            synthetic(Path(t), np.ones((100, 100)), corrupt=True)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                FlatCorrection(t).phi_pct([1], [1], [50.], [50.])

    def test_nonfinite_position_refused(self):
        with tempfile.TemporaryDirectory() as t:
            synthetic(Path(t), np.ones((100, 100)))
            with self.assertRaisesRegex(ValueError, 'finite'):
                FlatCorrection(t).phi_pct([1], [1], [np.nan], [50.])


if __name__ == '__main__':
    unittest.main()
