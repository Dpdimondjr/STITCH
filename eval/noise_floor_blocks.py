"""Estimate the sector-median measurement floor from cached TESS light curves.

The estimator splits each sector into equal-duration blocks, removes a robust
low-order trend, and measures the scatter of block medians.  The result is a
lower-bound estimate of the uncertainty in a sector median; block-to-block
astrophysical variability and residual systematics can only make it larger.

Run from the repository root::

    python3 eval/noise_floor_blocks.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import lightkurve as lk


OUT = Path("eval/noise_floor_results")
CACHE_GLOB = str(Path.home() / ".lightkurve/cache/mastDownload/TESS/**/*_lc.fits")
N_BLOCKS = 8


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad if np.isfinite(mad) else np.nan


def block_floor(flux, time, n_blocks=N_BLOCKS):
    """Return raw and line-detrended block-median floor estimates in log units."""
    flux = np.asarray(flux, dtype=float)
    time = np.asarray(time, dtype=float)
    good = np.isfinite(flux) & np.isfinite(time) & (flux > 0)
    flux, time = flux[good], time[good]
    if len(flux) < 8 * n_blocks:
        return None
    order = np.argsort(time)
    flux, time = flux[order], time[order]
    y = np.log(flux / np.nanmedian(flux))
    # Robustly reject transients/cadence outliers before block summaries.
    med = np.nanmedian(y)
    scale = robust_scale(y)
    if np.isfinite(scale) and scale > 0:
        keep = np.abs(y - med) < 6.0 * scale
        y, time = y[keep], time[keep]
    if len(y) < 8 * n_blocks:
        return None
    edges = np.linspace(0, len(y), n_blocks + 1).astype(int)
    centers, raw, residual = [], [], []
    for i in range(n_blocks):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        centers.append(np.nanmedian(time[a:b]))
        raw.append(np.nanmedian(y[a:b]))
    centers = np.asarray(centers)
    raw = np.asarray(raw)
    if len(raw) < 4:
        return None
    # Remove only a linear long-timescale trend. This is deliberately modest:
    # it gives a lower bound without allowing a flexible spline to erase noise.
    x = centers - np.nanmean(centers)
    try:
        coef = np.polyfit(x, raw, 1)
        residual = raw - np.polyval(coef, x)
    except (np.linalg.LinAlgError, ValueError):
        residual = raw - np.nanmedian(raw)
    raw_floor = np.nanstd(raw, ddof=1) / np.sqrt(len(raw))
    detr_floor = np.nanstd(residual, ddof=1) / np.sqrt(len(residual))
    return {
        "raw_floor": float(raw_floor),
        "detrended_floor": float(detr_floor),
        "n_blocks": int(len(raw)),
        "n_cadences": int(len(y)),
        "block_median_abs": float(np.nanmedian(np.abs(raw - np.nanmedian(raw)))),
    }


def process(path):
    try:
        lc = lk.io.read(path)
        q = lc.quality.value.astype(int)
        good = q == 0
        t = lc.time.value.astype(float)
        out = {}
        for name in ("sap_flux", "pdcsap_flux"):
            r = block_floor(lc[name].value[good], t[good])
            if r is None:
                return None
            out[name] = r
        m = re.search(r"s(\d{4})-(\d+)-", str(path))
        out.update({
            "tic_id": int(m.group(2)) if m else None,
            "sector": int(m.group(1)) if m else None,
            "tmag": float(lc.meta.get("TESSMAG") or np.nan),
            "cam": int(lc.meta.get("CAMERA") or 0),
            "ccd": int(lc.meta.get("CCD") or 0),
            "flfrcsap": float(lc.meta.get("FLFRCSAP") or np.nan),
        })
        return out
    except Exception:
        return None


def summarize(df, field):
    vals = df[field].apply(lambda x: x["detrended_floor"] * 100).to_numpy()
    raw = df[field].apply(lambda x: x["raw_floor"] * 100).to_numpy()
    return {
        "n_sectors": int(len(vals)),
        "n_stars": int(df.tic_id.nunique()),
        "median_detrended_pct": float(np.nanmedian(vals)),
        "q75_detrended_pct": float(np.nanquantile(vals, .75)),
        "q90_detrended_pct": float(np.nanquantile(vals, .90)),
        "q95_detrended_pct": float(np.nanquantile(vals, .95)),
        "median_raw_pct": float(np.nanmedian(raw)),
        "trimmed_std_detrended_pct": float(np.nanstd(vals[vals <= np.nanquantile(vals, .99)], ddof=1)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = glob.glob(CACHE_GLOB, recursive=True)
    rows = [r for p in paths if (r := process(p)) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable cached light curves found")
    print(f"Processed {len(df)} sectors from {df.tic_id.nunique()} TICs")
    result = {"n_files_found": len(paths), "n_sectors": len(df), "n_stars": int(df.tic_id.nunique()), "n_blocks": N_BLOCKS}
    for field in ("sap_flux", "pdcsap_flux"):
        result[field] = summarize(df, field)
        print(field, json.dumps(result[field], indent=2))
    for lo, hi in [(7, 9), (9, 10), (10, 11), (11, 12), (12, 13)]:
        sub = df[df.tmag.between(lo, hi, inclusive="left")]
        if len(sub):
            result.setdefault("tmag_bins", []).append({"lo": lo, "hi": hi, "n": len(sub),
                "pdc": summarize(sub, "pdcsap_flux")})
    df.to_pickle(OUT / "block_floor_records.pkl")
    (OUT / "block_floor_results.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {OUT / 'block_floor_results.json'}")


if __name__ == "__main__":
    main()
