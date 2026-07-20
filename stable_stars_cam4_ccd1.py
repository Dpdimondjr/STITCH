"""
Query TESS SPOC observations for a given sector, filter to Camera 4 / CCD 1,
select the most photometrically stable stars by CDPP1_0, then plot their
pixel positions on the detector using TPF reference coordinates.

MAST queries and product manifests are cached to CSV so re-runs are fast.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SECTOR   = 70
CAMERA   = 4
CCD      = 1
N_SCREEN = 200      # LC files to screen (increase if too few Cam4/CCD1 candidates)
N_STABLE = 25       # keep this many stable stars (lowest CDPP)
TMAG_MIN = 10.0
TMAG_MAX = 13.5
CACHE    = "./tess_cache"
os.makedirs(CACHE, exist_ok=True)

# Cache file paths (keyed by sector + N_SCREEN so changing either re-queries)
OBS_CACHE     = os.path.join(CACHE, f"obs_s{SECTOR:04d}.ecsv")
MANIFEST_CACHE = os.path.join(CACHE, f"manifest_s{SECTOR:04d}_n{N_SCREEN}.csv")
HEADERS_CACHE  = os.path.join(CACHE, f"headers_s{SECTOR:04d}_n{N_SCREEN}.csv")

# ── 1. Query MAST (cached) ────────────────────────────────────────────────────
if os.path.exists(OBS_CACHE):
    print(f"Loading cached obs from {OBS_CACHE}")
    obs = Table.read(OBS_CACHE)
else:
    print(f"Querying MAST: Sector {SECTOR} SPOC timeseries...")
    obs = Observations.query_criteria(
        obs_collection="TESS",
        sequence_number=SECTOR,
        provenance_name="SPOC",
        dataproduct_type="timeseries",
    )
    obs.write(OBS_CACHE, overwrite=True)
    print(f"  Cached to {OBS_CACHE}")

print(f"  {len(obs)} SPOC observations in sector {SECTOR}")

# ── 2. Product list + download (cached) ──────────────────────────────────────
if os.path.exists(MANIFEST_CACHE):
    print(f"Loading cached manifest from {MANIFEST_CACHE}")
    manifest_df = pd.read_csv(MANIFEST_CACHE)
    local_paths  = manifest_df["Local Path"].tolist()
else:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(obs), size=min(N_SCREEN, len(obs)), replace=False)
    sample_obs = obs[idx]

    print(f"Fetching product list for {len(sample_obs)} sampled observations...")
    products = Observations.get_product_list(sample_obs)
    lc_prods = products[products["productSubGroupDescription"] == "LC"]
    print(f"  {len(lc_prods)} LC files — downloading...")

    manifest = Observations.download_products(
        lc_prods, download_dir=CACHE, cache=True
    )
    manifest_df = manifest.to_pandas()
    manifest_df.to_csv(MANIFEST_CACHE, index=False)
    local_paths = manifest_df["Local Path"].tolist()
    print(f"  Manifest cached to {MANIFEST_CACHE}")

# ── 3. Screen headers (cached) ────────────────────────────────────────────────
if os.path.exists(HEADERS_CACHE):
    print(f"Loading cached headers from {HEADERS_CACHE}")
    all_headers = pd.read_csv(HEADERS_CACHE)
else:
    print("Screening FITS headers (HDU 0 + HDU 1)...")
    records = []
    for path in local_paths:
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path, memmap=True) as hdul:
                h0 = hdul[0].header   # CAMERA, CCD, TESSMAG, SECTOR, TICID
                h1 = hdul[1].header   # CDPP1_0, CDPP2_0, CDPP3_0
                records.append({
                    "tic_id":   h0.get("TICID"),
                    "tmag":     h0.get("TESSMAG"),
                    "camera":   h0.get("CAMERA"),
                    "ccd":      h0.get("CCD"),
                    "sector":   h0.get("SECTOR"),
                    "cdpp1_0":  h1.get("CDPP1_0"),
                    "cdpp2_0":  h1.get("CDPP2_0"),
                    "pdcvar":   h1.get("PDCVAR"),
                    "crowdsap": h1.get("CROWDSAP") or h0.get("CROWDSAP"),
                })
        except Exception as e:
            print(f"  Warning: {os.path.basename(path)}: {e}")

    all_headers = pd.DataFrame(records)
    all_headers.to_csv(HEADERS_CACHE, index=False)
    print(f"  Headers cached to {HEADERS_CACHE}")

print(f"  Screened {len(all_headers)} files total")

# ── 4. Filter to target camera/CCD and magnitude range ───────────────────────
df = all_headers.dropna(subset=["cdpp1_0", "tmag", "camera", "ccd"])
df = df[(df["camera"] == CAMERA) & (df["ccd"] == CCD)]
df = df[(df["tmag"] >= TMAG_MIN) & (df["tmag"] <= TMAG_MAX)]
print(f"  Camera {CAMERA}/CCD {CCD}, Tmag {TMAG_MIN}–{TMAG_MAX}: {len(df)} candidates")

if df.empty:
    print(
        "\nNo candidates found. Try:\n"
        "  - Increasing N_SCREEN (currently {N_SCREEN})\n"
        "  - Widening TMAG_MIN/TMAG_MAX\n"
        "  - Checking SECTOR has Cam4/CCD1 coverage"
    )
    raise SystemExit(1)

df = df.sort_values("cdpp1_0").head(N_STABLE)
print(f"\nTop {len(df)} stable stars (lowest CDPP1_0 ppm):")
print(df[["tic_id", "tmag", "cdpp1_0", "pdcvar", "crowdsap"]].to_string(index=False))

# ── 5. TPF pixel positions ────────────────────────────────────────────────────
TPF_CACHE = os.path.join(CACHE, f"positions_s{SECTOR:04d}_cam{CAMERA}_ccd{CCD}.csv")

if os.path.exists(TPF_CACHE):
    print(f"\nLoading cached TPF positions from {TPF_CACHE}")
    pos_df = pd.read_csv(TPF_CACHE)
else:
    print("\nFetching TPF positions...")
    positions = []
    for _, star in df.iterrows():
        tic = int(star["tic_id"])
        try:
            sr = lk.search_targetpixelfile(f"TIC {tic}", sector=SECTOR, author="SPOC")
            if len(sr) == 0:
                print(f"  TIC {tic}: no TPF found")
                continue
            tpf = sr[0].download(download_dir=CACHE)

            # Detector position
            col_centre = tpf.column + tpf.shape[2] / 2.0
            row_centre = tpf.row    + tpf.shape[1] / 2.0

            # POS_CORR: centroid offset from reference pixel (pixels)
            pc1 = tpf.hdu[1].data["POS_CORR1"].astype(float)
            pc2 = tpf.hdu[1].data["POS_CORR2"].astype(float)

            # Δsub-pixel: fractional pixel position of the centroid on the CCD
            # (reference pixel + median centroid offset, then take fractional part)
            delta_sub_col = (col_centre + np.nanmedian(pc1)) % 1
            delta_sub_row = (row_centre + np.nanmedian(pc2)) % 1

            # σjitter: RMS pointing scatter within this sector (pixels)
            jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))

            # Aap: number of pixels in the SPOC pipeline aperture
            aap = int(tpf.pipeline_mask.sum())

            positions.append({
                "tic_id":        tic,
                "tmag":          star["tmag"],
                "cdpp1_0":       star["cdpp1_0"],
                "col":           col_centre,
                "row":           row_centre,
                "delta_sub_col": delta_sub_col,
                "delta_sub_row": delta_sub_row,
                "jitter_rms":    jitter_rms,
                "aap":           aap,
            })
            print(f"  TIC {tic:12d}  Tmag={star['tmag']:.2f}  "
                  f"CDPP={star['cdpp1_0']:.1f} ppm  "
                  f"col={col_centre:.0f}  row={row_centre:.0f}  "
                  f"Δsub=({delta_sub_col:.3f},{delta_sub_row:.3f})  "
                  f"jitter={jitter_rms:.4f}px  Aap={aap}px")
        except Exception as e:
            print(f"  TIC {tic}: failed ({e})")

    pos_df = pd.DataFrame(positions)
    pos_df.to_csv(TPF_CACHE, index=False)
    print(f"  TPF positions cached to {TPF_CACHE}")

print(f"\n{len(pos_df)} stars with detector positions")

if pos_df.empty:
    print("No TPF positions — check TIC IDs and sector.")
    raise SystemExit(1)

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))

sc = ax.scatter(
    pos_df["col"], pos_df["row"],
    c=pos_df["cdpp1_0"], cmap="viridis_r",
    s=pos_df["tmag"].apply(lambda m: max(20, 300 - m * 20)),
    edgecolors="k", linewidths=0.4, zorder=3,
)
for _, r in pos_df.iterrows():
    ax.annotate(
        str(int(r["tic_id"])),
        (r["col"], r["row"]),
        textcoords="offset points", xytext=(5, 3),
        fontsize=5.5, color="0.3",
    )

cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("CDPP 1-hr (ppm)", fontsize=10)

ax.set_xlim(44, 2092)
ax.set_ylim(0, 2048)
ax.set_xlabel("Column (pixels)", fontsize=11)
ax.set_ylabel("Row (pixels)", fontsize=11)
ax.set_title(
    f"Stable Stars — Sector {SECTOR}, Camera {CAMERA}, CCD {CCD}\n"
    f"Tmag {TMAG_MIN}–{TMAG_MAX}, N={len(pos_df)}, colour = CDPP 1-hr (ppm)",
    fontsize=11,
)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2, linewidth=0.4)
plt.tight_layout()

outfile = f"cam{CAMERA}_ccd{CCD}_s{SECTOR:04d}_stable_stars.png"
plt.savefig(outfile, dpi=150)
plt.show()
print(f"Saved {outfile}")
