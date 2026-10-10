"""
Plot full TESS lightcurves for known variable stars, before and after STITCH.

Reads per-sector pred_offset values from eval/variable_star_test.csv,
re-downloads the SPOC lightcurves, applies the correction sector-by-sector,
and plots raw vs corrected.

Usage:
  python3 eval/plot_variable_lcs.py
"""

import warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve

warnings.filterwarnings("ignore")

CSV = "eval/variable_star_test.csv"
OUT = "eval/variable_lcs.png"

corrections = pd.read_csv(CSV)
corrections["sector"] = corrections["sector"].astype(int)
print(f"Loaded corrections for {corrections['tic_id'].nunique()} stars, "
      f"{len(corrections)} sector-rows")

stars = (corrections.groupby("tic_id")
         .agg(name=("name","first"), vartype=("vartype","first"),
              period=("period","first"), n_sectors=("sector","count"))
         .reset_index())
print(stars[["tic_id","name","vartype","period","n_sectors"]].to_string(index=False))

BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
C_RAW,C_COR="#ef4444","#4f8ef7"

n = len(stars)
fig, axes = plt.subplots(n, 2, figsize=(16, 3.8*n), facecolor=BG)
if n == 1:
    axes = axes[np.newaxis, :]

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlabel, color=MUT, fontsize=8)
    ax.set_ylabel(ylabel, color=MUT, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, pad=5)
    ax.tick_params(colors=MUT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, lw=0.3, zorder=0)
    ax.set_axisbelow(True)

for row_i, star in stars.iterrows():
    tic   = int(star["tic_id"])
    name  = star["name"]
    vtype = star["vartype"]
    per   = star["period"]
    corr  = corrections[corrections["tic_id"]==tic].set_index("sector")

    ax_raw = axes[row_i, 0]
    ax_cor = axes[row_i, 1]

    print(f"\nDownloading {name} (TIC {tic}) …", flush=True)
    try:
        results = search_lightcurve(f"TIC {tic}", mission="TESS",
                                    author="SPOC", exptime=120)
        lcs = results.download_all(quality_bitmask="default")
    except Exception as e:
        print(f"  Failed: {e}")
        for ax in (ax_raw, ax_cor):
            ax.set_facecolor(SURF)
            ax.text(0.5, 0.5, f"Download failed\n{e}", ha="center", va="center",
                    color=MUT, transform=ax.transAxes, fontsize=8)
        continue

    t_raw, f_raw = [], []
    t_cor, f_cor = [], []

    for lc in lcs:
        try:
            lc_clean = lc.remove_nans().remove_outliers(sigma=5)
            if len(lc_clean) < 50:
                continue
            sector = int(lc.meta.get("SECTOR", -1))
            t = lc_clean.time.btjd  # BTJD
            f = lc_clean.flux.value

            t_raw.extend(t); f_raw.extend(f)

            if sector in corr.index:
                po = corr.loc[sector, "pred_offset"]
                mean_f = np.nanmedian(f)
                f_c = (f / po) / (mean_f / po) * np.nanmedian(f)
                # Simpler: just divide by pred_offset (rescale whole sector)
                f_c = f / po
            else:
                f_c = f  # no correction available, leave raw

            t_cor.extend(t); f_cor.extend(f_c)
        except Exception as e:
            continue

    if not t_raw:
        continue

    t_raw = np.array(t_raw); f_raw = np.array(f_raw)
    t_cor = np.array(t_cor); f_cor = np.array(f_cor)

    # Sort by time
    idx = np.argsort(t_raw)
    t_raw, f_raw = t_raw[idx], f_raw[idx]
    idx2 = np.argsort(t_cor)
    t_cor, f_cor = t_cor[idx2], f_cor[idx2]

    # Normalise to median=1 for display
    med_raw = np.nanmedian(f_raw)
    med_cor = np.nanmedian(f_cor)
    f_raw_n = f_raw / med_raw
    f_cor_n = f_cor / med_cor

    mean_w  = corr["weight"].mean()
    mean_ps = corr["pred_std"].mean()
    nsec    = len(corr)

    # Plot raw
    ax_raw.plot(t_raw, f_raw_n, color=C_RAW, lw=0.3, alpha=0.6, rasterized=True)
    style(ax_raw, "BTJD (days)", "Norm. flux",
          f"{name}  [{vtype}, P~{per}d, {nsec} sectors]  RAW")

    # Plot corrected
    ax_cor.plot(t_cor, f_cor_n, color=C_COR, lw=0.3, alpha=0.6, rasterized=True)
    style(ax_cor, "BTJD (days)", "Norm. flux",
          f"{name}  STITCH corrected  (mean weight={mean_w:.3f})")

    print(f"  Plotted {len(t_raw):,} cadences across {nsec} sectors")

plt.suptitle("Known variable stars: full TESS lightcurve  |  raw (left) vs STITCH corrected (right)",
             color=TEXT, fontsize=11, y=1.005)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {OUT}")
