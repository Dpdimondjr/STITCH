"""
find_training_stars.py  (v2 — MAST-first approach)

Problem with v1: started from TIC sky positions, then hoped stars had SPOC
light curves. Most didn't — QLP is pre-normalized and useless for STITCH.

Fix: start from MAST SPOC observation catalog so every star in the output is
*guaranteed* to have downloadable SPOC PDCSAP light curves (absolute flux,
inter-sector DC offset preserved).

Pipeline:
  1. Query MAST for TESS SPOC timeseries observations across all sectors.
     Run in parallel batches — pure metadata, no FITS downloads.
  2. Extract TIC ID + sector + RA/Dec from observation records.
  3. Group by TIC ID. Keep stars with >= N_MIN_SECTORS SPOC observations.
     (Multi-sector SPOC coverage implies high ecliptic latitude / CVZ membership.)
  4. Query TIC catalog for Tmag; filter to TMAG_MIN–TMAG_MAX.
  5. Run tess-point on all kept TIC IDs → (sector, cam, CCD, col, row).
  6. Per (star, cam, CCD): count sectors and compute position diversity.
  7. Rank by combined score; save training_stars.csv + training_pairs.csv.

Outputs:
  training_stars.csv  — one row per (star, cam/CCD), ranked by training score
  training_pairs.csv  — one row per (star, sector) with col/row/cam/ccd
"""

import warnings
import numpy as np
import pandas as pd
from astroquery.mast import Observations, Catalogs
from tess_stars2px import tess_stars2px_function_entry
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

# Sectors to query. Stars appearing in many of these are CVZ members.
# Covers years 1-8 of TESS. Expand if needed.
SECTOR_MIN    = 1
SECTOR_MAX    = 100

TMAG_MIN      = 8.0    # brighter limit (very bright stars saturate)
TMAG_MAX      = 13.5
N_MIN_SECTORS = 3      # minimum SPOC sectors to qualify
MAX_STARS     = 15000  # cap fed into tess-point

N_BATCH_WORKERS = 8    # parallel MAST sector-batch queries
BATCH_SIZE      = 5    # sectors per batch query


# ── 1. Query MAST for SPOC LC observations ────────────────────────────────────

def query_sector_batch(sectors):
    """Return DataFrame of (tic_id, sector, ra, dec) for one batch of sectors."""
    try:
        obs = Observations.query_criteria(
            obs_collection="TESS",
            provenance_name="SPOC",
            dataproduct_type="timeseries",
            sequence_number=sectors,
        )
    except Exception as e:
        print(f"  batch {sectors[0]}-{sectors[-1]}: FAILED ({e})")
        return pd.DataFrame()

    if len(obs) == 0:
        return pd.DataFrame()

    df = obs.to_pandas()[["target_name", "sequence_number", "s_ra", "s_dec"]]
    df = df.rename(columns={
        "target_name":     "tic_id_raw",
        "sequence_number": "sector",
        "s_ra":            "ra",
        "s_dec":           "dec",
    })

    # target_name is plain TIC number or "TIC XXXXXXXXX"
    df["tic_id"] = (
        df["tic_id_raw"]
          .astype(str)
          .str.replace(r"[^\d]", "", regex=True)  # strip "TIC " prefix if present
          .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["tic_id", "ra", "dec"]).copy()
    df["tic_id"] = df["tic_id"].astype(int)
    df["sector"] = df["sector"].astype(int)

    n = len(df)
    sectors_got = sorted(df["sector"].unique())
    print(f"  sectors {sectors[0]:3d}–{sectors[-1]:3d}: {n:6d} obs "
          f"({len(sectors_got)} sectors with data)")
    return df[["tic_id", "sector", "ra", "dec"]]


sectors = list(range(SECTOR_MIN, SECTOR_MAX + 1))
batches = [sectors[i:i + BATCH_SIZE] for i in range(0, len(sectors), BATCH_SIZE)]

print("=== STITCH: Finding Training Stars (v2 — MAST-first) ===\n")
print(f"Step 1: Querying MAST SPOC observations "
      f"(sectors {SECTOR_MIN}–{SECTOR_MAX}, {N_BATCH_WORKERS} workers)...\n")

all_obs = []
with ThreadPoolExecutor(max_workers=N_BATCH_WORKERS) as pool:
    futures = {pool.submit(query_sector_batch, b): b for b in batches}
    for future in as_completed(futures):
        df = future.result()
        if not df.empty:
            all_obs.append(df)

if not all_obs:
    raise SystemExit("No SPOC observations returned — check MAST connectivity.")

obs_df = pd.concat(all_obs, ignore_index=True)
print(f"\n  Total SPOC observations: {len(obs_df)}")
print(f"  Unique TIC IDs:          {obs_df['tic_id'].nunique()}")
print(f"  Sectors with data:       {obs_df['sector'].nunique()}")


# ── 2. Keep multi-sector stars ────────────────────────────────────────────────

print(f"\nStep 2: Filtering for stars with >= {N_MIN_SECTORS} SPOC sectors...")

sector_counts = obs_df.groupby("tic_id")["sector"].nunique()
multi_sector_tics = sector_counts[sector_counts >= N_MIN_SECTORS].index

obs_df = obs_df[obs_df["tic_id"].isin(multi_sector_tics)].copy()

# Use mean RA/Dec per star (should be identical across sectors)
star_coords = (
    obs_df.groupby("tic_id")[["ra", "dec"]]
          .mean()
          .reset_index()
)

print(f"  Stars with >= {N_MIN_SECTORS} SPOC sectors: {len(star_coords)}")


# ── 3. Query TIC for Tmag ─────────────────────────────────────────────────────

print(f"\nStep 3: Fetching Tmag from TIC catalog (batched)...")

CHUNK = 1000
tmag_rows = []
tic_ids = star_coords["tic_id"].tolist()

for i in range(0, len(tic_ids), CHUNK):
    chunk = tic_ids[i:i + CHUNK]
    try:
        cat = Catalogs.query_criteria(catalog="TIC", ID=chunk)
        if len(cat) == 0:
            continue
        cdf = cat.to_pandas()[["ID", "Tmag"]].copy()
        cdf["ID"] = pd.to_numeric(cdf["ID"], errors="coerce")
        tmag_rows.append(cdf.rename(columns={"ID": "tic_id"}))
        print(f"  [{i+len(chunk)}/{len(tic_ids)}] chunk done")
    except Exception as e:
        print(f"  [{i}] chunk failed: {e}")

if tmag_rows:
    tmag_df = pd.concat(tmag_rows, ignore_index=True).dropna()
    tmag_df["tic_id"] = tmag_df["tic_id"].astype(int)
    star_coords = star_coords.merge(tmag_df, on="tic_id", how="left")
    before = len(star_coords)
    star_coords = star_coords[
        star_coords["Tmag"].between(TMAG_MIN, TMAG_MAX, inclusive="both")
    ]
    print(f"  Tmag filter {TMAG_MIN}–{TMAG_MAX}: {len(star_coords)} / {before} stars kept")
else:
    print("  WARNING: Tmag query failed — skipping magnitude filter")
    star_coords["Tmag"] = np.nan

if len(star_coords) > MAX_STARS:
    star_coords = star_coords.sample(MAX_STARS, random_state=42).reset_index(drop=True)
    print(f"  Sampled down to {MAX_STARS} stars")


# ── 4. Run tess-point ─────────────────────────────────────────────────────────

print(f"\nStep 4: Running tess-point on {len(star_coords)} stars...")

outID, _, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, _ = \
    tess_stars2px_function_entry(
        star_coords["tic_id"].values,
        star_coords["ra"].values,
        star_coords["dec"].values,
    )

pairs = pd.DataFrame({
    "tic_id":    outID.astype(int),
    "sector":    outSec.astype(int),
    "cam":       outCam.astype(int),
    "ccd":       outCcd.astype(int),
    "col":       outColPix.astype(float),
    "row":       outRowPix.astype(float),
    "eclip_lat": outEclipLat.astype(float),
})

# Exclude collateral columns
pairs = pairs[
    (pairs["col"] >= 44) & (pairs["col"] <= 2092) &
    (pairs["row"] >= 0)  & (pairs["row"] <= 2048)
].reset_index(drop=True)

# Only keep tess-point sectors that ALSO exist in the MAST SPOC catalog
# (guarantees the LC actually exists for that sector)
spoc_pairs = set(zip(obs_df["tic_id"], obs_df["sector"]))
pairs["has_spoc"] = pairs.apply(
    lambda r: (r["tic_id"], r["sector"]) in spoc_pairs, axis=1
)
pairs = pairs[pairs["has_spoc"]].drop(columns="has_spoc").reset_index(drop=True)

print(f"  {len(pairs)} (star, sector) pairs confirmed in MAST SPOC catalog")
print(f"  Sectors: {pairs['sector'].min()}–{pairs['sector'].max()}")


# ── 5. Summarise per (star, cam, CCD) ─────────────────────────────────────────

print(f"\nStep 5: Computing position diversity per cam/CCD...")

tmag_lookup = star_coords.set_index("tic_id")["Tmag"].to_dict()

summary_rows = []
for (tic, cam, ccd), grp in pairs.groupby(["tic_id", "cam", "ccd"]):
    n = len(grp)
    if n < N_MIN_SECTORS:
        continue

    col_std = grp["col"].std() if n > 1 else 0.0
    row_std = grp["row"].std() if n > 1 else 0.0
    pos_diversity = float(np.sqrt(col_std**2 + row_std**2))

    summary_rows.append({
        "tic_id":           tic,
        "cam":              cam,
        "ccd":              ccd,
        "n_sectors":        n,
        "sectors":          sorted(grp["sector"].tolist()),
        "col_mean":         grp["col"].mean(),
        "row_mean":         grp["row"].mean(),
        "col_std":          col_std,
        "row_std":          row_std,
        "pos_diversity_px": pos_diversity,
        "col_range_px":     grp["col"].max() - grp["col"].min(),
        "row_range_px":     grp["row"].max() - grp["row"].min(),
        "tmag":             tmag_lookup.get(tic, np.nan),
    })

summary = pd.DataFrame(summary_rows)

if summary.empty:
    raise SystemExit("No stars passed all filters. Check MAST connectivity or widen Tmag range.")

print(f"  {len(summary)} (star, cam/CCD) combos with >= {N_MIN_SECTORS} sectors")
print(f"  Unique stars: {summary['tic_id'].nunique()}")
print(f"  Cam/CCD breakdown:")
for (cam, ccd), g in summary.groupby(["cam", "ccd"]):
    print(f"    Cam{cam}/CCD{ccd}: {len(g):5d} stars | "
          f"median {g['n_sectors'].median():.0f} sec | "
          f"median pos_div {g['pos_diversity_px'].median():.1f} px")


# ── 6. Rank ───────────────────────────────────────────────────────────────────

print(f"\nStep 6: Ranking by n_sectors × position_diversity...")

max_sec = summary["n_sectors"].max()
max_div = summary["pos_diversity_px"].max()
summary["score_sectors"]   = summary["n_sectors"] / max_sec
summary["score_diversity"] = summary["pos_diversity_px"] / (max_div + 1e-6)
summary["training_score"]  = (0.5 * summary["score_sectors"] +
                               0.5 * summary["score_diversity"])

summary = summary.sort_values("training_score", ascending=False).reset_index(drop=True)

print(f"\n  Top 20 training candidates (guaranteed SPOC PDCSAP available):")
print(f"  {'TIC':>12}  Cam CCD  n_sec  pos_div(px)  Tmag  score")
print(f"  {'─'*65}")
for _, r in summary.head(20).iterrows():
    print(f"  {int(r['tic_id']):>12}    {int(r['cam'])}   {int(r['ccd'])}  "
          f"  {int(r['n_sectors']):>4}     {r['pos_diversity_px']:>7.1f}    "
          f"  {r['tmag']:>4.1f}  {r['training_score']:.3f}")


# ── 7. Save ───────────────────────────────────────────────────────────────────

summary_out = summary.copy()
summary_out["sectors"] = summary_out["sectors"].apply(lambda s: ",".join(map(str, s)))
summary_out.to_csv("training_stars.csv", index=False)
print(f"\n  Saved training_stars.csv  ({len(summary_out)} rows)")

tmag_merge = summary[["tic_id", "cam", "ccd", "n_sectors", "training_score", "tmag"]]
pairs_out = pairs.merge(tmag_merge, on=["tic_id", "cam", "ccd"], how="inner")
pairs_out = (pairs_out
             .drop_duplicates(subset=["tic_id", "sector"])
             .sort_values(["training_score", "tic_id", "sector"],
                          ascending=[False, True, True])
             .reset_index(drop=True))

pairs_out.to_csv("training_pairs.csv", index=False)
print(f"  Saved training_pairs.csv  ({len(pairs_out)} rows)")

print(f"\n=== Summary ===")
print(f"  Stars (guaranteed SPOC):   {summary['tic_id'].nunique()}")
print(f"  (star, sector) pairs:      {len(pairs_out)}")
print(f"  Sectors range:             {pairs_out['sector'].min()}–{pairs_out['sector'].max()}")
print(f"  Median sectors/star:       {summary['n_sectors'].median():.0f}")
print(f"  Median pos diversity:      {summary['pos_diversity_px'].median():.1f} px")
print(f"\nAll stars guaranteed to have SPOC PDCSAP. Run collect_training_data.py next.")
