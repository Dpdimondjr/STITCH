"""
For each stable star (with known col/row on Cam4/CCD1), find all sectors
where it was observed on the same camera/CCD, compute per-sector flux offsets,
and visualise the spatial distribution of those offsets across the detector.

This is the raw data needed for a per-sector gradient map.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import lightkurve as lk

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA      = 4
CCD         = 1
REF_SECTOR  = 70          # sector we identified stable stars in
CACHE       = "./tess_cache"
POSITIONS_CSV = os.path.join(
    CACHE, f"positions_s{REF_SECTOR:04d}_cam{CAMERA}_ccd{CCD}.csv"
)
OFFSETS_CACHE = os.path.join(
    CACHE, f"offsets_cam{CAMERA}_ccd{CCD}.csv"
)

# ── 1. Load stable star positions ─────────────────────────────────────────────
pos_df = pd.read_csv(POSITIONS_CSV)
print(f"Loaded {len(pos_df)} stable stars from Cam{CAMERA}/CCD{CCD}/Sector{REF_SECTOR}")
print(pos_df[["tic_id", "tmag", "cdpp1_0", "col", "row"]].to_string(index=False))

# ── 2. For each star, find all sectors + compute flux offsets (cached) ────────
if os.path.exists(OFFSETS_CACHE):
    print(f"\nLoading cached offsets from {OFFSETS_CACHE}")
    offsets_df = pd.read_csv(OFFSETS_CACHE)
else:
    print("\nSearching all sectors for each star and computing flux offsets...")
    records = []

    for _, star in pos_df.iterrows():
        tic = int(star["tic_id"])
        print(f"\n  TIC {tic} (Tmag={star['tmag']:.2f})...")

        # Search all available SPOC light curves for this star
        try:
            sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC")
        except Exception as e:
            print(f"    Search failed: {e}")
            continue

        if len(sr) == 0:
            print("    No SPOC results found")
            continue

        print(f"    Found {len(sr)} sectors in archive")

        # Download all light curves
        try:
            lc_col = sr.download_all(download_dir=CACHE)
        except Exception as e:
            print(f"    Download failed: {e}")
            continue

        # Global median across ALL sectors (any cam/ccd) — this is the reference
        # so that a single sector on Cam4/CCD1 still gives a meaningful offset
        all_flux = np.concatenate([lc.flux.value for lc in lc_col])
        global_median = np.nanmedian(all_flux)
        if global_median == 0 or np.isnan(global_median):
            continue

        # Record every sector, noting which cam/ccd it was on
        same_cam_lcs = []
        for lc in lc_col:
            cam = lc.meta.get("CAMERA")
            ccd = lc.meta.get("CCD")
            if cam == CAMERA and ccd == CCD:
                same_cam_lcs.append(lc)

        print(f"    {len(same_cam_lcs)} sector(s) on Cam{CAMERA}/CCD{CCD}  "
              f"(global median from {len(lc_col)} total sectors)")

        if not same_cam_lcs:
            print(f"    Skipping — no sectors on Cam{CAMERA}/CCD{CCD}")
            continue

        for lc in same_cam_lcs:
            sector_median = np.nanmedian(lc.flux.value)
            flux_offset   = sector_median / global_median

            # σjitter: RMS pointing scatter within this sector from POS_CORR columns
            try:
                pc1 = lc["pos_corr1"].value.astype(float)
                pc2 = lc["pos_corr2"].value.astype(float)
                jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
            except Exception:
                jitter_rms = np.nan

            records.append({
                "tic_id":        tic,
                "tmag":          star["tmag"],
                "cdpp1_0":       star["cdpp1_0"],
                "col":           star["col"],
                "row":           star["row"],
                "delta_sub_col": star.get("delta_sub_col", np.nan),
                "delta_sub_row": star.get("delta_sub_row", np.nan),
                "aap":           star.get("aap", np.nan),
                "sector":        lc.meta.get("SECTOR"),
                "flux_offset":   flux_offset,
                "jitter_rms":    jitter_rms,
                "crowdsap":      lc.meta.get("CROWDSAP"),
                "pdcvar":        lc.meta.get("PDCVAR"),
            })

    offsets_df = pd.DataFrame(records)
    offsets_df.to_csv(OFFSETS_CACHE, index=False)
    print(f"\nOffsets cached to {OFFSETS_CACHE}")

print(f"\nTotal (star, sector) pairs: {len(offsets_df)}")
print(f"Sectors covered: {sorted(offsets_df['sector'].unique())}")
print(f"Stars:           {offsets_df['tic_id'].nunique()}")

# ── 3. Summary table: pivot (rows=sector, cols=tic) ──────────────────────────
pivot = offsets_df.pivot_table(
    index="sector", columns="tic_id", values="flux_offset"
)
print("\nPer-sector flux offsets (rows=sector, cols=TIC):")
print(pivot.round(4).to_string())

# ── 4. Per-sector spatial plots ───────────────────────────────────────────────
sectors = sorted(offsets_df["sector"].unique())
n_sectors = len(sectors)
ncols = min(4, n_sectors)
nrows = int(np.ceil(n_sectors / ncols))

# Shared colour scale centred on 1.0
vmin = offsets_df["flux_offset"].quantile(0.05)
vmax = offsets_df["flux_offset"].quantile(0.95)
spread = max(abs(1.0 - vmin), abs(vmax - 1.0), 1e-4)
vmin = min(vmin, 1.0 - spread)
vmax = max(vmax, 1.0 + spread)
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                         constrained_layout=True)
axes = np.array(axes).flatten()

for i, sector in enumerate(sectors):
    ax = axes[i]
    sub = offsets_df[offsets_df["sector"] == sector]

    sc = ax.scatter(
        sub["col"], sub["row"],
        c=sub["flux_offset"], cmap="RdBu_r", norm=norm,
        s=sub["tmag"].apply(lambda m: max(30, 250 - m * 18)),
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
    ax.set_title(f"Sector {int(sector)}", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linewidth=0.3)

# Hide unused panels
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[:i+1], shrink=0.6, pad=0.02)
cbar.set_label("Flux offset (normalised)", fontsize=10)

fig.suptitle(
    f"Per-sector flux offsets — Camera {CAMERA}, CCD {CCD}\n"
    f"(colour: deviation from star's global median; size ~ brightness)",
    fontsize=11,
)

outfile = f"cam{CAMERA}_ccd{CCD}_sector_offsets.png"
plt.savefig(outfile, dpi=150)
plt.show()
print(f"\nSaved {outfile}")

# ── 5. Optional: per-sector linear gradient fit ───────────────────────────────
print("\nPer-sector linear gradient (col coeff, row coeff, intercept):")
from numpy.linalg import lstsq

grad_records = []
for sector in sectors:
    sub = offsets_df[offsets_df["sector"] == sector].dropna(subset=["col","row","flux_offset"])
    if len(sub) < 3:
        continue
    A = np.column_stack([sub["col"].astype(float), sub["row"].astype(float), np.ones(len(sub))])
    b = sub["flux_offset"].astype(float).values
    coeffs, _, _, _ = lstsq(A, b, rcond=None)
    grad_records.append({
        "sector":    int(sector),
        "grad_col":  coeffs[0],
        "grad_row":  coeffs[1],
        "intercept": coeffs[2],
        "n_stars":   len(sub),
    })
    print(f"  Sector {int(sector):3d}: grad_col={coeffs[0]:+.2e}  "
          f"grad_row={coeffs[1]:+.2e}  intercept={coeffs[2]:.4f}  (n={len(sub)})")

grad_df = pd.DataFrame(grad_records)
grad_csv = f"cam{CAMERA}_ccd{CCD}_gradients.csv"
grad_df.to_csv(grad_csv, index=False)
print(f"\nGradient table saved to {grad_csv}")
