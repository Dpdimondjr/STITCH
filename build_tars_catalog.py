"""
Build a catalog of photometrically quiet stars from TARS Table 4.

Strategy:
  1. Try reading only the 5 needed columns directly from Zenodo URL (~1.5 GB
     via HTTP range requests — Arrow IPC column projection). Falls back to
     downloading the full 13 GB file if URL reading fails.
  2. Filter: keep TIC-sector pairs where systematic_score > SYS_THRESH.
  3. Aggregate per TIC: require >= MIN_SECTORS quiet sectors.
  4. Save tars_quiet_tics.csv — TICID list with camera/CCD info, ready to
     feed directly into collect_training_data_v2.py.

Why systematic_score?
  The TARS random-forest classifier assigns each sector a score in [0,1]
  representing how "systematic-like" (non-astrophysical) the variability is.
  Score near 1.0 → star looks like pure instrumental drift → no intrinsic
  variability → ideal STITCH training star (we need the sensor noise, not
  the star's own light curve shape).

Run once; idempotent (skips if tars_quiet_tics.csv already exists).
"""

import os, sys, requests
import numpy as np
import pandas as pd
import pyarrow.feather as feather

# ── Config ────────────────────────────────────────────────────────────────────

ZENODO_URL   = "https://zenodo.org/api/records/19917941/files/tars_table_4.feather/content"
LOCAL_T4     = "tars_table_4.feather"
OUT_CSV      = "tars_quiet_tics_v2.csv"

SYS_THRESH      = 0.95   # lowered from 0.99 to expand training set
MIN_SECTORS     = 2      # star must be quiet in >= this many sectors to be useful
MAX_PER_CAM_CCD = 15000  # cap per camera-CCD combo (16 combos → up to 240K total)

COLS = ["TICID", "systematic_score", "camera", "ccd", "sector", "Tmag"]

# Stars fainter than this won't have TESS-SPOC products (SPOC stops ~Tmag 13-14).
# Without this filter, the quiet-star list is dominated by Tmag~15 stars that
# are "quiet" mainly because they're too faint to detect real variability.
TMAG_MAX = 13.0


if os.path.exists(OUT_CSV):
    print(f"{OUT_CSV} already exists — delete it to rebuild.")
    import sys; sys.exit(0)


def download_with_resume(url: str, dest: str) -> None:
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    if existing:
        print(f"  Resuming from {existing/1e9:.2f} GB...")
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + existing
        downloaded = existing
        with open(dest, "ab" if existing else "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r  {downloaded/1e9:.2f} / {total/1e9:.2f} GB  ({downloaded/total*100:.1f}%)",
                              end="", flush=True)
    print()


# ── 1. Load Table 4 columns ───────────────────────────────────────────────────
# Try URL-based column-projected read first (~1.5 GB); fall back to local file.

def load_table4() -> pd.DataFrame:
    # Try URL first (Arrow IPC column projection via HTTP range requests)
    try:
        import fsspec
        print(f"Attempting column-projected read from URL ({', '.join(COLS)})...")
        print("  (Arrow IPC will issue range requests for only these columns, ~1.5 GB)")
        fs = fsspec.filesystem("http")
        with fs.open(ZENODO_URL, "rb") as f:
            table = feather.read_table(f, columns=COLS)
        df = table.to_pandas()
        print(f"  URL read succeeded: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"  URL read failed ({e}), falling back to local download...")

    # Fall back: download full file with resume support
    if not os.path.exists(LOCAL_T4):
        r = requests.head(ZENODO_URL, allow_redirects=True)
        total_gb = int(r.headers.get("Content-Length", 0)) / 1e9
        print(f"Downloading {total_gb:.1f} GB → {LOCAL_T4} (one-time, may take 30-60 min)")
        download_with_resume(ZENODO_URL, LOCAL_T4)
        print(f"Download complete: {os.path.getsize(LOCAL_T4)/1e9:.2f} GB")
    else:
        print(f"Found local {LOCAL_T4} ({os.path.getsize(LOCAL_T4)/1e9:.1f} GB)")

    print(f"Reading columns {COLS}...")
    return feather.read_table(LOCAL_T4, columns=COLS).to_pandas()


df = load_table4()
print(f"  Loaded {len(df):,} TIC-sector pairs")
print(f"  systematic_score distribution:")
print(f"    min={df['systematic_score'].min():.4f}  "
      f"max={df['systematic_score'].max():.4f}  "
      f"mean={df['systematic_score'].mean():.4f}")
frac_above = (df["systematic_score"] > SYS_THRESH).mean() * 100
print(f"    fraction > {SYS_THRESH}: {frac_above:.2f}%")
n_bright = (df["Tmag"] <= TMAG_MAX).mean() * 100
print(f"    fraction with Tmag ≤ {TMAG_MAX}: {n_bright:.1f}%")


# ── 3. Filter to quiet sectors ────────────────────────────────────────────────
# Apply Tmag cut first: faint stars appear quiet because their signal is noise-
# dominated, not because they're photometrically stable. TESS-SPOC only covers
# Tmag < ~13-14, so this also maximises SPOC yield in the collector.

bright = df[df["Tmag"] <= TMAG_MAX].copy()
print(f"\nAfter Tmag ≤ {TMAG_MAX}: {len(bright):,} TIC-sector pairs "
      f"({bright['TICID'].nunique():,} unique TICs)")

quiet = bright[bright["systematic_score"] > SYS_THRESH].copy()
print(f"After sys_score > {SYS_THRESH}: {len(quiet):,} quiet TIC-sector pairs")
print(f"  Unique TICs: {quiet['TICID'].nunique():,}")


# ── 4. Aggregate per TIC ──────────────────────────────────────────────────────
# Keep only TICs with >= MIN_SECTORS quiet sectors.
# For camera/CCD: use the mode (most common camera-CCD combo for that TIC).

agg = (quiet.groupby("TICID")
       .agg(
           n_quiet_sectors=("sector", "count"),
           mean_sys_score=("systematic_score", "mean"),
           camera=("camera", lambda x: int(x.mode()[0])),
           ccd=("ccd", lambda x: int(x.mode()[0])),
       )
       .reset_index())

agg = agg[agg["n_quiet_sectors"] >= MIN_SECTORS].copy()
agg = agg.sort_values("mean_sys_score", ascending=False)
print(f"\nAfter requiring >= {MIN_SECTORS} quiet sectors: {len(agg):,} TICs")

# Per-cam coverage
print("\n  Camera distribution:")
for cam in sorted(agg["camera"].unique()):
    n = (agg["camera"] == cam).sum()
    print(f"    Cam{cam}: {n:,} TICs")


# ── 5. Cap per cam-CCD to avoid runaway collection ───────────────────────────

agg_capped = (agg
    .sort_values("mean_sys_score", ascending=False)
    .groupby(["camera", "ccd"], group_keys=False)
    .head(MAX_PER_CAM_CCD)
    .reset_index(drop=True))

print(f"\nAfter cap ({MAX_PER_CAM_CCD}/cam-ccd): {len(agg_capped):,} TICs")
print("\n  Per cam-CCD breakdown:")
for (cam, ccd), g in agg_capped.groupby(["camera", "ccd"]):
    print(f"    Cam{cam}/CCD{ccd}: {len(g):4d} TICs  "
          f"(mean_sys={g['mean_sys_score'].mean():.4f})")


# ── 6. Save ───────────────────────────────────────────────────────────────────

# Rename to match collect_training_data.py expectations
agg_capped = agg_capped.rename(columns={"TICID": "tic_id", "camera": "cam"})
agg_capped.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(agg_capped):,} quiet TICs)")
print("\nNext: run collect_training_data.py with STARS_CSV = 'tars_quiet_tics.csv'")
