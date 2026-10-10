"""Within-sector orbit-split noise floor estimate.

TESS sectors are ~27 days = 2 orbits of ~13.7 days each. Splitting each sector
into first and second halves gives an independent estimate of measurement noise
that is free of cross-sector systematics. This gives a lower bound on the
irreducible scatter floor for SAP and PDCSAP flux.

Reads cached SPOC LC FITS files from the lightkurve cache — no new downloads.

Run from the repository root:
    python3 eval/noise_floor_orbit_split.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightkurve as lk
except ImportError:
    raise SystemExit("lightkurve required: pip install lightkurve")

OUT = Path("eval/noise_floor_results")
CACHE_GLOB = str(Path.home() / ".lightkurve/cache/mastDownload/TESS/**/*_lc.fits")


def half_medians(flux, time):
    """Split time-ordered flux into two halves, return medians."""
    order = np.argsort(time)
    f = flux[order]
    mid = len(f) // 2
    return np.nanmedian(f[:mid]), np.nanmedian(f[mid:])


def process_file(path):
    try:
        lc = lk.io.read(path)
        # Quality filter: good cadences only
        q = lc.quality.value.astype(int)
        good = (q == 0)
        if good.sum() < 100:
            return None
        t = lc.time.value[good]
        sap = lc["sap_flux"].value.astype(float)[good]
        pdc = lc["pdcsap_flux"].value.astype(float)[good]
        pc1 = lc["pos_corr1"].value.astype(float)[good]
        pc2 = lc["pos_corr2"].value.astype(float)[good]

        if not (np.isfinite(sap).mean() > 0.8 and np.isfinite(pdc).mean() > 0.8):
            return None

        # Orbit split
        sap_h1, sap_h2 = half_medians(sap, t)
        pdc_h1, pdc_h2 = half_medians(pdc, t)

        if not (sap_h1 > 0 and sap_h2 > 0 and pdc_h1 > 0 and pdc_h2 > 0):
            return None

        # Per-axis jitter
        pc1_rms = float(np.sqrt(np.nanvar(pc1)))
        pc2_rms = float(np.sqrt(np.nanvar(pc2)))
        jitter_total = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))

        m = re.search(r"s(\d{4})-(\d+)-", str(path))
        sector = int(m.group(1)) if m else None
        tic = int(m.group(2)) if m else None

        return {
            "tic_id": tic,
            "sector": sector,
            "n_good": int(good.sum()),
            "sap_log_ratio": float(np.log(sap_h1 / sap_h2)),
            "pdc_log_ratio": float(np.log(pdc_h1 / pdc_h2)),
            "sap_h1": sap_h1, "sap_h2": sap_h2,
            "pdc_h1": pdc_h1, "pdc_h2": pdc_h2,
            "pc1_rms": pc1_rms, "pc2_rms": pc2_rms, "jitter_total": jitter_total,
            "flfrcsap": float(lc.meta.get("FLFRCSAP") or np.nan),
            "tmag": float(lc.meta.get("TESSMAG") or np.nan),
            "cam": int(lc.meta.get("CAMERA") or 0),
            "ccd": int(lc.meta.get("CCD") or 0),
        }
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = glob.glob(CACHE_GLOB, recursive=True)
    print(f"Found {len(files)} cached LC files", flush=True)

    records = []
    for f in files:
        r = process_file(f)
        if r:
            records.append(r)
    df = pd.DataFrame(records)
    print(f"Processed {len(df)} files successfully", flush=True)

    # Floor estimate: std(log_ratio) / sqrt(2) for each flux type
    sap_floor = df.sap_log_ratio.std() / np.sqrt(2) * 100
    pdc_floor = df.pdc_log_ratio.std() / np.sqrt(2) * 100

    # Median absolute within-sector variation
    sap_mad = df.sap_log_ratio.abs().median() / np.sqrt(2) * 100
    pdc_mad = df.pdc_log_ratio.abs().median() / np.sqrt(2) * 100

    # By tmag bin
    print("\n=== Floor by tmag bin (PDCSAP) ===")
    tmag_rows = []
    for lo, hi in [(7, 9), (9, 10), (10, 11), (11, 12), (12, 13)]:
        sub = df[(df.tmag >= lo) & (df.tmag < hi)]
        if len(sub) < 3:
            continue
        floor = sub.pdc_log_ratio.std() / np.sqrt(2) * 100
        mad = sub.pdc_log_ratio.abs().median() / np.sqrt(2) * 100
        print(f"  tmag [{lo},{hi}): n={len(sub):3d}  floor(std)={floor:.4f}%  floor(MAD)={mad:.4f}%")
        tmag_rows.append({"tmag_lo": lo, "tmag_hi": hi, "n": len(sub),
                          "pdc_floor_std_pct": floor, "pdc_floor_mad_pct": mad})

    # Per-axis jitter asymmetry
    df["pc_anisotropy"] = df.pc1_rms / (df.pc2_rms + 1e-9)
    print(f"\n=== Per-axis jitter (POS_CORR) ===")
    print(f"  pc1_rms (col): median={df.pc1_rms.median():.4f} px")
    print(f"  pc2_rms (row): median={df.pc2_rms.median():.4f} px")
    print(f"  anisotropy pc1/pc2: median={df.pc_anisotropy.median():.3f}")
    print(f"  fraction col>row: {(df.pc1_rms > df.pc2_rms).mean()*100:.1f}%")

    # Does jitter anisotropy correlate with within-sector variation?
    r_aniso = df["pc_anisotropy"].corr(df.pdc_log_ratio.abs())
    r_total = df["jitter_total"].corr(df.pdc_log_ratio.abs())
    r_pc1 = df["pc1_rms"].corr(df.pdc_log_ratio.abs())
    r_pc2 = df["pc2_rms"].corr(df.pdc_log_ratio.abs())
    print(f"\n=== Jitter vs within-sector |PDC log ratio| ===")
    print(f"  r(jitter_total, |pdc_log_ratio|): {r_total:.4f}")
    print(f"  r(pc1_rms, |pdc_log_ratio|):      {r_pc1:.4f}")
    print(f"  r(pc2_rms, |pdc_log_ratio|):      {r_pc2:.4f}")
    print(f"  r(anisotropy, |pdc_log_ratio|):   {r_aniso:.4f}")

    print(f"\n=== Summary ===")
    print(f"  SAP within-sector floor (std):   {sap_floor:.4f}%")
    print(f"  SAP within-sector floor (MAD):   {sap_mad:.4f}%")
    print(f"  PDC within-sector floor (std):   {pdc_floor:.4f}%")
    print(f"  PDC within-sector floor (MAD):   {pdc_mad:.4f}%")
    print(f"  Note: STITCH best corrected CV = 0.931% (random-TIC split)")
    print(f"        If PDC within-sector floor < 0.931%, headroom exists")

    df.to_csv(OUT / "orbit_split.csv", index=False)
    result = {
        "n_files": len(df),
        "sap_floor_std_pct": sap_floor,
        "sap_floor_mad_pct": sap_mad,
        "pdc_floor_std_pct": pdc_floor,
        "pdc_floor_mad_pct": pdc_mad,
        "jitter_pc1_rms_median_px": float(df.pc1_rms.median()),
        "jitter_pc2_rms_median_px": float(df.pc2_rms.median()),
        "r_pc1_vs_within_sector": float(r_pc1),
        "r_pc2_vs_within_sector": float(r_pc2),
        "r_jitter_total_vs_within_sector": float(r_total),
        "tmag_bins": tmag_rows,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
