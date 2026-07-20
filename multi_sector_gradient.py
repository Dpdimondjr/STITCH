"""
For each sector in SECTOR_RANGE, find stable stars on Camera 4 / CCD 1,
compute their flux offsets relative to each star's all-sector global median,
and plot spatial gradient maps per sector.

Sectors are processed in parallel with ThreadPoolExecutor.
All intermediate results are cached so re-runs are fast.
"""

import os
import warnings
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lightkurve as lk
from astroquery.mast import Observations
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SECTOR_RANGE = range(60, 81)   # sectors to search
CAMERA       = 4
CCD          = 1
N_SCREEN     = 300             # LC files to screen per sector
N_STABLE     = 20              # stable stars to keep per sector (lowest CDPP)
TMAG_MIN     = 10.0
TMAG_MAX     = 13.5
N_WORKERS    = 4               # parallel threads (keep ≤6 to respect MAST limits)
CACHE        = "./tess_cache"
os.makedirs(CACHE, exist_ok=True)

# Semaphore to limit concurrent MAST API calls
mast_sem = threading.Semaphore(3)

# ── Per-sector pipeline ───────────────────────────────────────────────────────

def screen_sector(sector):
    """
    Query MAST for a sector, download a sample of LC headers, filter to
    Camera/CCD, return DataFrame of stable star candidates.
    Fully cached — re-running is instant.
    """
    cache_file = os.path.join(CACHE, f"headers_s{sector:04d}_n{N_SCREEN}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        n = len(df[(df["camera"] == CAMERA) & (df["ccd"] == CCD)])
        print(f"  [s{sector}] loaded from cache ({n} on Cam{CAMERA}/CCD{CCD})")
        return df

    print(f"  [s{sector}] querying MAST...")
    with mast_sem:
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS",
                sequence_number=sector,
                provenance_name="SPOC",
                dataproduct_type="timeseries",
            )
        except Exception as e:
            print(f"  [s{sector}] MAST query failed: {e}")
            return pd.DataFrame()

    if len(obs) == 0:
        print(f"  [s{sector}] no SPOC observations found")
        return pd.DataFrame()

    # Random but deterministic sample per sector
    rng = np.random.default_rng(sector)
    idx = rng.choice(len(obs), size=min(N_SCREEN, len(obs)), replace=False)

    with mast_sem:
        try:
            products = Observations.get_product_list(obs[idx])
        except Exception as e:
            print(f"  [s{sector}] product list failed: {e}")
            return pd.DataFrame()

    lc_prods = products[products["productSubGroupDescription"] == "LC"]
    if len(lc_prods) == 0:
        return pd.DataFrame()

    with mast_sem:
        try:
            manifest = Observations.download_products(
                lc_prods, download_dir=CACHE, cache=True
            )
        except Exception as e:
            print(f"  [s{sector}] download failed: {e}")
            return pd.DataFrame()

    records = []
    for row in manifest:
        path = row["Local Path"]
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path, memmap=True) as hdul:
                h0 = hdul[0].header
                h1 = hdul[1].header
                cam, ccd = h0.get("CAMERA"), h0.get("CCD")
                tmag = h0.get("TESSMAG")
                cdpp = h1.get("CDPP1_0")
                if tmag is None or cdpp is None:
                    continue
                records.append({
                    "tic_id":   h0.get("TICID"),
                    "tmag":     tmag,
                    "cdpp1_0":  cdpp,
                    "pdcvar":   h1.get("PDCVAR"),
                    "crowdsap": h0.get("CROWDSAP"),
                    "camera":   cam,
                    "ccd":      ccd,
                    "sector":   h0.get("SECTOR"),
                })
        except Exception:
            pass

    df = pd.DataFrame(records)
    df.to_csv(cache_file, index=False)
    n = len(df[(df["camera"] == CAMERA) & (df["ccd"] == CCD)])
    print(f"  [s{sector}] screened {len(df)} files, {n} on Cam{CAMERA}/CCD{CCD}")
    return df


def get_star_offsets(tic, sector):
    """
    Download all SPOC LCs for a TIC, compute flux offset for the given sector
    relative to the star's global median across all sectors.
    Returns a dict or None.
    """
    try:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC")
        if len(sr) == 0:
            return None
        lc_col = sr.download_all(download_dir=CACHE)
        if lc_col is None or len(lc_col) == 0:
            return None

        # Global median across all sectors/cameras
        all_flux = np.concatenate([lc.flux.value for lc in lc_col])
        global_median = np.nanmedian(all_flux)
        if global_median == 0 or np.isnan(global_median):
            return None

        # Find the LC for the target sector on the right cam/ccd
        for lc in lc_col:
            if (lc.meta.get("SECTOR") == sector and
                    lc.meta.get("CAMERA") == CAMERA and
                    lc.meta.get("CCD") == CCD):
                sector_median = np.nanmedian(lc.flux.value)
                flux_offset   = sector_median / global_median

                # σjitter from POS_CORR
                try:
                    pc1 = lc["pos_corr1"].value.astype(float)
                    pc2 = lc["pos_corr2"].value.astype(float)
                    jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
                except Exception:
                    jitter_rms = np.nan

                return {
                    "flux_offset": flux_offset,
                    "jitter_rms":  jitter_rms,
                    "n_sectors":   len(lc_col),
                }
        return None
    except Exception:
        return None


def get_tpf_position(tic, sector):
    """
    Download TPF and extract detector position + sub-pixel phase + aperture size.
    Returns a dict or None.
    """
    cache_file = os.path.join(CACHE, f"tpf_pos_tic{tic}_s{sector:04d}.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file).iloc[0].to_dict()
    try:
        sr = lk.search_targetpixelfile(f"TIC {tic}", sector=sector, author="SPOC")
        if len(sr) == 0:
            return None
        tpf = sr[0].download(download_dir=CACHE)

        col_centre = tpf.column + tpf.shape[2] / 2.0
        row_centre = tpf.row    + tpf.shape[1] / 2.0

        pc1 = tpf.hdu[1].data["POS_CORR1"].astype(float)
        pc2 = tpf.hdu[1].data["POS_CORR2"].astype(float)
        delta_sub_col = (col_centre + np.nanmedian(pc1)) % 1
        delta_sub_row = (row_centre + np.nanmedian(pc2)) % 1
        aap = int(tpf.pipeline_mask.sum())

        result = {
            "col": col_centre, "row": row_centre,
            "delta_sub_col": delta_sub_col, "delta_sub_row": delta_sub_row,
            "aap": aap,
        }
        pd.DataFrame([result]).to_csv(cache_file, index=False)
        return result
    except Exception:
        return None


def process_sector(sector):
    """
    Full pipeline for one sector:
      1. Screen LC headers → stable star candidates on Cam/CCD
      2. For each candidate: get flux offset + TPF position
    Returns DataFrame of (tic, sector, col, row, flux_offset, ...) or empty DF.
    """
    out_cache = os.path.join(CACHE, f"sector_result_s{sector:04d}_cam{CAMERA}_ccd{CCD}.csv")
    if os.path.exists(out_cache):
        df = pd.read_csv(out_cache)
        print(f"  [s{sector}] result loaded from cache ({len(df)} stars)")
        return df

    # Step 1: screen headers
    headers = screen_sector(sector)
    if headers.empty:
        return pd.DataFrame()

    candidates = (
        headers
        .dropna(subset=["cdpp1_0", "tmag", "camera", "ccd"])
        .query(f"camera == {CAMERA} and ccd == {CCD}")
        .query(f"{TMAG_MIN} <= tmag <= {TMAG_MAX}")
        .sort_values("cdpp1_0")
        .head(N_STABLE)
    )

    if candidates.empty:
        print(f"  [s{sector}] no Cam{CAMERA}/CCD{CCD} candidates after filtering")
        return pd.DataFrame()

    print(f"  [s{sector}] {len(candidates)} candidates — fetching offsets + positions...")

    # Step 2: flux offsets + positions for each candidate
    records = []
    for _, star in candidates.iterrows():
        tic = int(star["tic_id"])

        offsets = get_star_offsets(tic, sector)
        if offsets is None:
            continue

        pos = get_tpf_position(tic, sector)
        if pos is None:
            continue

        records.append({
            "tic_id":        tic,
            "sector":        sector,
            "tmag":          star["tmag"],
            "cdpp1_0":       star["cdpp1_0"],
            "crowdsap":      star["crowdsap"],
            "col":           pos["col"],
            "row":           pos["row"],
            "delta_sub_col": pos["delta_sub_col"],
            "delta_sub_row": pos["delta_sub_row"],
            "aap":           pos["aap"],
            "flux_offset":   offsets["flux_offset"],
            "jitter_rms":    offsets["jitter_rms"],
            "n_sectors":     offsets["n_sectors"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_cache, index=False)
    print(f"  [s{sector}] done — {len(df)} stars with positions + offsets")
    return df


# ── Run all sectors in parallel ───────────────────────────────────────────────
print(f"Processing sectors {list(SECTOR_RANGE)} with {N_WORKERS} workers...\n")

all_results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(process_sector, s): s for s in SECTOR_RANGE}
    for future in as_completed(futures):
        sector = futures[future]
        try:
            df = future.result()
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            print(f"  [s{sector}] unexpected error: {e}")

if not all_results:
    print("No results found across any sector. Try adjusting SECTOR_RANGE or N_SCREEN.")
    raise SystemExit(1)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(os.path.join(CACHE, f"all_sectors_cam{CAMERA}_ccd{CCD}.csv"), index=False)

sectors_found = sorted(combined["sector"].unique())
print(f"\nSectors with Cam{CAMERA}/CCD{CCD} data: {sectors_found}")
print(f"Total (star, sector) pairs: {len(combined)}")
print(f"Unique stars: {combined['tic_id'].nunique()}")
print(f"Flux offset range: {combined['flux_offset'].min():.4f} – {combined['flux_offset'].max():.4f}")

# ── Plot: one panel per sector ────────────────────────────────────────────────
n_sectors = len(sectors_found)
if n_sectors == 0:
    raise SystemExit("Nothing to plot.")

ncols = min(4, n_sectors)
nrows = int(np.ceil(n_sectors / ncols))

spread = max(
    abs(1.0 - combined["flux_offset"].quantile(0.05)),
    abs(combined["flux_offset"].quantile(0.95) - 1.0),
    1e-3,
)
norm = mcolors.TwoSlopeNorm(vmin=1.0 - spread, vcenter=1.0, vmax=1.0 + spread)

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                         constrained_layout=True)
axes = np.array(axes).flatten()

for i, sector in enumerate(sectors_found):
    ax  = axes[i]
    sub = combined[combined["sector"] == sector]

    sc = ax.scatter(
        sub["col"], sub["row"],
        c=sub["flux_offset"], cmap="RdBu_r", norm=norm,
        s=sub["tmag"].apply(lambda m: max(25, 250 - m * 18)),
        edgecolors="k", linewidths=0.3, zorder=3,
    )
    for _, r in sub.iterrows():
        ax.annotate(
            f"{r['flux_offset']:.3f}",
            (r["col"], r["row"]),
            textcoords="offset points", xytext=(4, 3),
            fontsize=5, color="0.2",
        )

    ax.set_xlim(44, 2092)
    ax.set_ylim(0, 2048)
    ax.set_title(f"Sector {int(sector)}  (n={len(sub)})", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linewidth=0.3)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes[:i+1], shrink=0.5, pad=0.02, label="Flux offset (norm.)")
fig.suptitle(
    f"Per-sector flux offsets — Camera {CAMERA}, CCD {CCD}\n"
    f"Sectors {sectors_found[0]}–{sectors_found[-1]}  |  "
    f"blue = below star's global median, red = above",
    fontsize=11,
)

outfile = f"cam{CAMERA}_ccd{CCD}_multi_sector_gradient.png"
plt.savefig(outfile, dpi=150)
plt.show()
print(f"\nSaved {outfile}")

# ── Linear gradient per sector ────────────────────────────────────────────────
print("\nPer-sector linear gradient fit:")
from numpy.linalg import lstsq

grad_rows = []
for sector in sectors_found:
    sub = combined[combined["sector"] == sector].dropna(subset=["col", "row", "flux_offset"])
    if len(sub) < 3:
        print(f"  Sector {int(sector):3d}: skipped (n={len(sub)} < 3)")
        continue
    A = np.column_stack([
        sub["col"].astype(float),
        sub["row"].astype(float),
        np.ones(len(sub)),
    ])
    b = sub["flux_offset"].astype(float).values
    coeffs, _, _, _ = lstsq(A, b, rcond=None)
    grad_rows.append({
        "sector":    int(sector),
        "grad_col":  coeffs[0],
        "grad_row":  coeffs[1],
        "intercept": coeffs[2],
        "n_stars":   len(sub),
    })
    print(f"  Sector {int(sector):3d}: grad_col={coeffs[0]:+.2e}  "
          f"grad_row={coeffs[1]:+.2e}  intercept={coeffs[2]:.4f}  (n={len(sub)})")

grad_df = pd.DataFrame(grad_rows)
grad_csv = f"cam{CAMERA}_ccd{CCD}_gradients.csv"
grad_df.to_csv(grad_csv, index=False)
print(f"\nGradient table saved to {grad_csv}")
