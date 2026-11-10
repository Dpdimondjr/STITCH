"""Point-source flat-field correction (added 2026-11-05).

SPOC divides every pixel by the 2018 flat. Its pixel-scale pattern appears in stellar flux with slope
-0.983 +- 0.025 on held-out confirmation stars, while calibrated sky carries almost none of it, so the flat
looks right for diffuse light and wrong for point sources. The physical cause is not established. Evidence,
limits and what the correction does not fix: eval/flat_imprint_20261105/REPORT.md.

    phi [percent] = 100 G_0.7 * (ln f - G_15 * ln f) at (row - 1, col - 1), bilinear
    corrected flux = flux * exp(phi / 100)

f is FLAT_DATA of tess2018323-{cam}-{ccd}-flat.fits with NaN -> 1, clipped to [0.8, 1.2]; G_s is a Gaussian of
sigma s pixels. The amplitude is fixed at 1. Nothing is fitted to any star and no stellar flux is read, so the
correction depends only on where the star landed and cannot respond to its variability.
"""
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage as ni

from .frozen import REPO, sha256

FLAT_DIR = REPO / 'tess_cache/flat_models'
HIGHPASS_SIGMA_PX = 15
FOOTPRINT_SIGMA_PX = .7
CLIP = (.8, 1.2)
EVIDENCE = 'eval/flat_imprint_20261105/REPORT.md'


class FlatCorrection:
    def __init__(self, flat_dir=FLAT_DIR):
        self.dir = Path(flat_dir)
        receipts = self.dir / 'receipts.json'
        if not receipts.exists():
            raise FileNotFoundError(f'No flat fields in {self.dir}. Run: python3 data/download_flat_models.py')
        self.receipts = {r['name']: r for r in json.loads(receipts.read_text())}
        self.maps = {}

    def map(self, cam, ccd):
        """Footprint-weighted high-pass log flat for one CCD, in percent. Checksum verified on first use."""
        key = (int(cam), int(ccd))
        if key not in self.maps:
            name = f'tess2018323-{key[0]}-{key[1]}-flat.fits'
            path = self.dir / name
            if name not in self.receipts or not path.exists():
                raise FileNotFoundError(f'Missing {path}. Run: python3 data/download_flat_models.py')
            if sha256(path) != self.receipts[name]['sha256']:
                raise ValueError(f'Flat checksum mismatch: {path}')
            flat = fits.getdata(path, 1).astype(float)
            log_flat = np.log(np.clip(np.nan_to_num(flat, nan=1.), *CLIP))
            highpass = log_flat - ni.gaussian_filter(log_flat, HIGHPASS_SIGMA_PX, mode='nearest')
            self.maps[key] = (100 * ni.gaussian_filter(highpass, FOOTPRINT_SIGMA_PX, mode='nearest')).astype(np.float32)
        return self.maps[key]

    def phi_pct(self, cam, ccd, col, row):
        """phi in percent for FITS physical (1-based) detector positions. Positions must be finite."""
        cam, ccd, col, row = [np.asarray(v, float) for v in (cam, ccd, col, row)]
        if not all(np.isfinite(v).all() for v in (cam, ccd, col, row)):
            raise ValueError('Flat correction needs a finite camera, CCD, column and row for every visit')
        out = np.empty(len(col))
        for c, k in {(int(a), int(b)) for a, b in zip(cam, ccd)}:
            m = (cam == c) & (ccd == k)
            out[m] = ni.map_coordinates(self.map(c, k), [row[m] - 1, col[m] - 1], order=1, mode='nearest')
        return out

    def describe(self):
        return dict(recipe='corrected flux = flux * exp(phi/100); phi = 100 G_0.7 * (ln f - G_15 * ln f); amplitude fixed at 1',
                    high_pass_sigma_px=HIGHPASS_SIGMA_PX, footprint_sigma_px=FOOTPRINT_SIGMA_PX, clip=list(CLIP),
                    fitted_parameters=0, uses_stellar_flux=False, evidence=EVIDENCE,
                    flat_sha256={n: r['sha256'] for n, r in sorted(self.receipts.items())})
