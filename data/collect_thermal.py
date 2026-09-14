"""
Download TESS engineering FITS files, extract per-cam/CCD focal plane
temperature (ALCU sensor median), and join into the training parquet.

Usage:
  python3 data/collect_thermal.py [parquet] [s_lo] [s_hi]
  python3 data/collect_thermal.py training_data_v3.parquet 1 30
"""

import io, os, re, sys, warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
from astropy.io import fits

PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")),
               "training_data_v3.parquet")
args = [s for s in sys.argv[1:] if not s.endswith(".parquet")]
S_LO = int(args[0]) if len(args) > 0 else 1
S_HI = int(args[1]) if len(args) > 1 else 30

MAST_ENG = "https://archive.stsci.edu/missions/tess/engineering/"

# ── Find all sector eng.fits URLs ──────────────────────────────────────────────
print("Scanning MAST engineering directory …")
r = requests.get(MAST_ENG, timeout=30)
all_files = re.findall(r'href="(tess[^"]+_sector\d+-eng\.fits)"', r.text)

sector_urls = {}
for fname in all_files:
    m = re.search(r'_sector(\d+)-eng\.fits', fname)
    if m:
        sec = int(m.group(1))
        sector_urls[sec] = MAST_ENG + fname

print(f"  Found {len(sector_urls)} sector eng.fits files")
missing = [s for s in range(S_LO, S_HI+1) if s not in sector_urls]
if missing:
    print(f"  WARNING: no eng.fits for sectors {missing}")

# ── Extract median focal plane temp per cam/CCD ────────────────────────────────
records = []

for sec in range(S_LO, S_HI + 1):
    if sec not in sector_urls:
        print(f"S{sec:02d}: no URL, skipping")
        continue

    url = sector_urls[sec]
    mb = requests.head(url, timeout=10).headers.get("Content-Length", "?")
    mb_str = f"{int(mb)/1024/1024:.0f} MB" if mb != "?" else "? MB"
    print(f"S{sec:02d}: downloading {mb_str} … ", end="", flush=True)

    try:
        resp = requests.get(url, timeout=300)
    except Exception as e:
        print(f"FAILED ({e})")
        continue

    print(f"parsing … ", end="", flush=True)

    try:
        with fits.open(io.BytesIO(resp.content)) as hdul:
            hdu_names = [h.name for h in hdul]
            for cam in range(1, 5):
                for ccd in range(1, 5):
                    sensor = f"S_CAM{cam}_ALCU_sensor_CCD{ccd}"
                    if sensor not in hdu_names:
                        continue
                    data = hdul[sensor].data
                    col_name = "COOKED" if "COOKED" in data.dtype.names else "VALUE"
                    cooked = data[col_name].astype(float)
                    cooked = cooked[np.isfinite(cooked)]
                    if len(cooked) == 0:
                        continue
                    records.append({
                        "sector": sec,
                        "cam":    cam,
                        "ccd":    ccd,
                        "focal_plane_temp": float(np.median(cooked)),
                        "focal_plane_temp_std": float(cooked.std()),
                        "focal_plane_temp_n":   len(cooked),
                    })
    except Exception as e:
        print(f"ERROR ({e})")
        continue

    print(f"done ({len([r for r in records if r['sector']==sec])} panels)")

# ── Summary ────────────────────────────────────────────────────────────────────
temp_df = pd.DataFrame(records)
print(f"\nExtracted {len(temp_df)} cam/CCD/sector temperature readings")
if len(temp_df):
    print(temp_df.groupby("cam")["focal_plane_temp"].describe().round(2))

    # ── Join onto parquet ──────────────────────────────────────────────────────
    print(f"\nLoading {PARQUET} …")
    df = pd.read_parquet(PARQUET)

    # Merge new readings on top of any existing focal_plane_temp values
    new_cols = ["focal_plane_temp", "focal_plane_temp_std"]
    merge_df = temp_df[["sector", "cam", "ccd"] + new_cols]
    df = df.merge(merge_df, on=["sector", "cam", "ccd"], how="left",
                  suffixes=("_old", ""))
    # Prefer newly merged values; fall back to existing where new is NaN
    for col in new_cols:
        old_col = col + "_old"
        if old_col in df.columns:
            df[col] = df[col].combine_first(df[old_col])
            df = df.drop(columns=[old_col])


    filled = df["focal_plane_temp"].notna().sum()
    print(f"  Filled focal_plane_temp for {filled:,} / {len(df):,} rows "
          f"({filled/len(df)*100:.1f}%)")
    print(f"  Global temp range: {df['focal_plane_temp'].min():.2f} – "
          f"{df['focal_plane_temp'].max():.2f} °C")

    print(f"Saving to {PARQUET} …")
    df.to_parquet(PARQUET, index=False)
    print("Done.")
else:
    print("No data extracted — parquet not modified.")
