"""Compute TESS PRF capture-fraction features for each star-sector observation.

For each observation, downloads (or uses cached) TESS PRF model files from MAST
and computes:
  - prf_q: predicted aperture capture fraction at the star's subpixel position
  - prf_dq_dcol, prf_dq_drow: sensitivity to detector-plane shifts (flux loss gradient)
  - prf_d2q_dcol2, prf_d2q_drow2: second-order curvature
  - prf_boundary_frac: fraction of PRF flux within 0.5px of the aperture edge
  - prf_q_log_ratio: log(prf_q / flfrcsap) — discrepancy with SPOC pipeline estimate
  - prf_grid_cell: which 5×5 interpolation cell the star falls in (0–24)

Physical motivation: identical jitter_rms can produce very different flux losses
depending on direction relative to the aperture boundary and PRF gradient.
These features expose that motion × aperture × PRF interaction.

PRF file source: https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0001/
File format: 117×117 float64 arrays (13 pixels × 9 subpixel oversampling per axis).
Column convention: tpf_col + 44 = prf_col (TPF uses 0-based science pixels;
  PRF grid starts at col=45, which corresponds to the first science column).

Usage:
    python3 data/compute_prf_features.py training_data_v3.parquet training_data_v3_prf.parquet
    python3 data/compute_prf_features.py --dry-run training_data_v3.parquet /tmp/test_prf.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

PRF_BASE_URL = (
    "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/start_s0001/"
    "cam{cam}_ccd{ccd}/"
)
# Exact grid positions in PRF coordinates (parsed from MAST filenames).
PRF_GRID_ROWS = [1, 513, 1025, 1536, 2048]
PRF_GRID_COLS = [45, 557, 1069, 1580, 2092]

PRF_OVERSAMPLE = 9    # subpixels per pixel
PRF_N_PIX = 13        # pixels per axis in PRF template
PRF_ARRAY_SIZE = PRF_N_PIX * PRF_OVERSAMPLE  # = 117

# Timestamp prefix varies by cam/ccd — discovered from MAST directory listings.
# cam1 all + cam2 ccd1-3 → 163600; cam2 ccd4 + cam3-4 all → 163601.
_PRF_TIMESTAMP = {
    (1, 1): "tess2018243163600", (1, 2): "tess2018243163600",
    (1, 3): "tess2018243163600", (1, 4): "tess2018243163600",
    (2, 1): "tess2018243163600", (2, 2): "tess2018243163600",
    (2, 3): "tess2018243163600", (2, 4): "tess2018243163601",
    (3, 1): "tess2018243163601", (3, 2): "tess2018243163601",
    (3, 3): "tess2018243163601", (3, 4): "tess2018243163601",
    (4, 1): "tess2018243163601", (4, 2): "tess2018243163601",
    (4, 3): "tess2018243163601", (4, 4): "tess2018243163601",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Input parquet (e.g. training_data_v3.parquet)")
    p.add_argument("output", help="Output parquet with prf_* columns added")
    p.add_argument("--prf-cache", default=str(Path.home() / ".cache" / "tess_prf"),
                   help="Directory to cache PRF FITS files (default: ~/.cache/tess_prf)")
    p.add_argument("--aperture-radius", type=float, default=None,
                   help="Override aperture radius in pixels; default: tmag-based")
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--dry-run", action="store_true",
                   help="Process first 500 rows only for testing")
    return p.parse_args()


def tpf_to_prf_col(col):
    """Convert TPF 0-based column to PRF coordinate (add 44)."""
    return col + 44.0


def aperture_radius_from_tmag(tmag):
    """Approximate SPOC aperture radius in pixels from Tmag.

    Bright stars get larger apertures; clipped to [1.5, 4.0] px.
    """
    return float(np.clip(2.5 - (tmag - 8.0) / 3.0, 1.5, 4.0))


def prf_filename(cam, ccd, row, col):
    ts = _PRF_TIMESTAMP.get((int(cam), int(ccd)), "tess2018243163600")
    return f"{ts}-prf-{cam}-{ccd}-row{int(row):04d}-col{int(col):04d}.fits"


def download_if_needed(local_path, url, timeout=30):
    """Download url to local_path if not already cached. Returns True on success."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return True
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"    WARNING: download failed {url}: {e}", flush=True)
        return False


class TessPRFGrid:
    """Loaded and bilinearly interpolatable TESS PRF for one cam/ccd.

    Holds up to 25 normalised 117×117 PRF templates at the 5×5 detector grid.
    """

    def __init__(self, cam, ccd, cache_dir, base_url=PRF_BASE_URL):
        self.cam = cam
        self.ccd = ccd
        self._templates = []   # list of (prf_col, prf_row, array)
        self._load(cam, ccd, cache_dir, base_url)

    def _load(self, cam, ccd, cache_dir, base_url):
        local_dir = Path(cache_dir) / f"cam{cam}_ccd{ccd}"
        local_dir.mkdir(parents=True, exist_ok=True)
        url_base = base_url.format(cam=cam, ccd=ccd)
        n_ok = 0
        for row_pos in PRF_GRID_ROWS:
            for col_pos in PRF_GRID_COLS:
                fname = prf_filename(cam, ccd, row_pos, col_pos)
                local_path = local_dir / fname
                url = url_base + fname
                if not download_if_needed(local_path, url):
                    continue
                try:
                    with fits.open(local_path) as hdul:
                        data = hdul[0].data.astype(np.float64)
                    norm = data.sum()
                    if norm > 0:
                        data = data / norm
                    self._templates.append((float(col_pos), float(row_pos), data))
                    n_ok += 1
                except Exception as e:
                    print(f"    WARNING: read error {local_path}: {e}", flush=True)
        if n_ok == 0:
            raise RuntimeError(f"No PRF files loaded for cam{cam} ccd{ccd}")
        print(f"  Loaded {n_ok}/25 PRF templates for cam{cam} ccd{ccd}", flush=True)

    def interpolate(self, prf_col, prf_row):
        """Return a normalised PRF array bilinearly interpolated to (prf_col, prf_row).

        Uses inverse-squared-distance weighting over all loaded grid points.
        """
        cols = np.array([t[0] for t in self._templates])
        rows = np.array([t[1] for t in self._templates])
        arrays = [t[2] for t in self._templates]
        dists2 = (cols - prf_col) ** 2 + (rows - prf_row) ** 2
        w = 1.0 / (dists2 + 1e-6)
        prf = sum(w[i] * arrays[i] for i in range(len(arrays))) / w.sum()
        norm = prf.sum()
        return prf / norm if norm > 0 else prf

    def shifted_prf(self, prf_col, prf_row, sub_col, sub_row):
        """Interpolated PRF shifted to the star's subpixel position.

        sub_col, sub_row: fractional pixel offsets ∈ [0, 1).
        Shift by rolling the oversampled array by round(sub * OVERSAMPLE) cells.
        """
        prf = self.interpolate(prf_col, prf_row)
        dc = int(round(sub_col * PRF_OVERSAMPLE)) % PRF_OVERSAMPLE
        dr = int(round(sub_row * PRF_OVERSAMPLE)) % PRF_OVERSAMPLE
        if dc != 0:
            prf = np.roll(prf, dc, axis=1)
        if dr != 0:
            prf = np.roll(prf, dr, axis=0)
        return prf

    def capture_fraction(self, prf_col, prf_row, sub_col, sub_row, radius_px):
        """Compute q = Σ_{p ∈ aperture} P_p.

        The 117×117 PRF array covers 13×13 pixels.  Each pixel's total
        contribution is the sum of its 9×9 subpixel block.
        Returns (q, boundary_frac) where boundary_frac is the fraction of q
        within 0.5 px of the aperture boundary (sensitivity indicator).
        """
        prf = self.shifted_prf(prf_col, prf_row, sub_col, sub_row)
        center = (PRF_N_PIX - 1) / 2.0  # = 6.0

        # Pre-compute per-pixel PRF sums.
        os_ = PRF_OVERSAMPLE
        n = PRF_N_PIX
        pixel_flux = np.array([[prf[i*os_:(i+1)*os_, j*os_:(j+1)*os_].sum()
                                 for j in range(n)] for i in range(n)])
        # Pixel distances from center.
        ii, jj = np.mgrid[0:n, 0:n]
        dists = np.sqrt((ii - center) ** 2 + (jj - center) ** 2)

        in_ap = dists <= radius_px
        near_edge = in_ap & (np.abs(dists - radius_px) <= 0.5)
        q = float(pixel_flux[in_ap].sum())
        boundary = float(pixel_flux[near_edge].sum())
        bf = (boundary / q) if q > 0 else np.nan
        return q, bf

    def features(self, tpf_col, tpf_row, sub_col, sub_row, radius_px, delta=1/9):
        """All PRF features for one observation.

        Returns dict: prf_q, prf_dq_dcol, prf_dq_drow, prf_d2q_dcol2,
        prf_d2q_drow2, prf_boundary_frac.

        delta: finite-difference step in subpixel units (default 1/9 px = 1 PRF
        oversampling step). The derivatives dq/d(sub_col) and dq/d(sub_row) measure
        how much the capture fraction changes when the star shifts within its pixel —
        the direct aperture sensitivity to pointing jitter. They are computed by
        varying sub_col/sub_row, NOT the PRF grid interpolation coordinate.
        """
        prf_col = tpf_to_prf_col(tpf_col)
        prf_row = float(tpf_row)
        q0, bf = self.capture_fraction(prf_col, prf_row, sub_col, sub_row, radius_px)
        # +/- one sub-pixel step in column direction (sub_col ∈ [0,1), wrap via mod)
        qpc, _ = self.capture_fraction(prf_col, prf_row, (sub_col + delta) % 1.0, sub_row, radius_px)
        qmc, _ = self.capture_fraction(prf_col, prf_row, (sub_col - delta) % 1.0, sub_row, radius_px)
        qpr, _ = self.capture_fraction(prf_col, prf_row, sub_col, (sub_row + delta) % 1.0, radius_px)
        qmr, _ = self.capture_fraction(prf_col, prf_row, sub_col, (sub_row - delta) % 1.0, radius_px)
        return dict(
            prf_q=q0,
            prf_dq_dcol=(qpc - qmc) / (2 * delta),
            prf_dq_drow=(qpr - qmr) / (2 * delta),
            prf_d2q_dcol2=(qpc - 2 * q0 + qmc) / (delta ** 2),
            prf_d2q_drow2=(qpr - 2 * q0 + qmr) / (delta ** 2),
            prf_boundary_frac=bf,
        )


def grid_cell_index(tpf_col, tpf_row):
    """Which 5×5 grid cell (0–24) the PRF position falls in."""
    prf_col = tpf_to_prf_col(tpf_col)
    ci = int(np.searchsorted(PRF_GRID_COLS, prf_col, side="right") - 1)
    ri = int(np.searchsorted(PRF_GRID_ROWS, tpf_row, side="right") - 1)
    return int(np.clip(ri, 0, 4)) * 5 + int(np.clip(ci, 0, 4))


PRF_COLS = [
    "prf_q", "prf_dq_dcol", "prf_dq_drow",
    "prf_d2q_dcol2", "prf_d2q_drow2",
    "prf_boundary_frac", "prf_q_log_ratio", "prf_grid_cell",
]


def process_chunk(chunk, prf_grids, radius_override):
    """Compute PRF features for one chunk. Returns DataFrame indexed like chunk."""
    rows = []
    for _, obs in chunk.iterrows():
        key = (int(obs.cam), int(obs.ccd))
        grid = prf_grids.get(key)
        if grid is None:
            rows.append({c: np.nan for c in PRF_COLS})
            rows[-1]["prf_grid_cell"] = -1
            continue

        r_px = (radius_override if radius_override is not None
                else aperture_radius_from_tmag(float(obs.tmag)))
        sub_col = float(obs.delta_sub_col) % 1.0
        sub_row = float(obs.delta_sub_row) % 1.0

        try:
            feat = grid.features(float(obs.col), float(obs.row), sub_col, sub_row, r_px)
        except Exception as e:
            feat = {c: np.nan for c in PRF_COLS if c not in ("prf_q_log_ratio", "prf_grid_cell")}

        q = feat.get("prf_q", np.nan)
        flfrcsap = getattr(obs, "flfrcsap", np.nan)
        if np.isfinite(q) and q > 0 and np.isfinite(flfrcsap) and flfrcsap > 0:
            feat["prf_q_log_ratio"] = np.log(q / flfrcsap)
        else:
            feat["prf_q_log_ratio"] = np.nan
        feat["prf_grid_cell"] = grid_cell_index(float(obs.col), float(obs.row))
        rows.append(feat)

    return pd.DataFrame(rows, index=chunk.index)


def main():
    args = parse_args()

    print(f"Loading {args.input} …")
    df = pd.read_parquet(args.input)
    if args.dry_run:
        df = df.head(500).copy()
        print(f"  Dry-run: processing first {len(df)} rows only")
    else:
        print(f"  {len(df):,} rows, {df.tic_id.nunique():,} unique stars")

    # Load PRF grids for all cam/ccd combinations.
    unique_ccds = df[["cam", "ccd"]].drop_duplicates().values.astype(int)
    prf_grids = {}
    for cam, ccd in unique_ccds:
        print(f"Preparing PRF grid cam{cam} ccd{ccd} …", flush=True)
        try:
            prf_grids[(cam, ccd)] = TessPRFGrid(cam, ccd, args.prf_cache)
        except RuntimeError as e:
            print(f"  Skipping cam{cam} ccd{ccd}: {e}", flush=True)

    if not prf_grids:
        print("ERROR: no PRF grids loaded.", file=sys.stderr)
        sys.exit(1)

    # Drop existing prf_* columns if re-running.
    for col in PRF_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Allocate output.
    prf_df = pd.DataFrame(np.nan, index=df.index, columns=PRF_COLS)
    prf_df["prf_grid_cell"] = -1

    n_total = len(df)
    for start in range(0, n_total, args.chunk_size):
        end = min(start + args.chunk_size, n_total)
        chunk = df.iloc[start:end]
        print(f"  Rows {start:,}–{end:,} / {n_total:,}", flush=True)
        result = process_chunk(chunk, prf_grids, args.aperture_radius)
        prf_df.loc[result.index, result.columns] = result.values

    df_out = pd.concat([df, prf_df], axis=1)
    print(f"\nSaving → {args.output}")
    df_out.to_parquet(args.output, index=False)

    print("\nPRF feature fill rates:")
    for col in PRF_COLS:
        if col == "prf_grid_cell":
            frac = (df_out[col] >= 0).mean()
        else:
            frac = df_out[col].notna().mean()
        print(f"  {col:25s}: {frac*100:.1f}%")
    print("Done.")


if __name__ == "__main__":
    main()
