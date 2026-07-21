"""
Side-by-side comparison: TARS quiet stars vs non-quiet stars
CDPP 1-hour vs photon noise floor.

Quiet stars: from training_data.parquet (TARS sys_score > 0.95)
Non-quiet:   from nonquiet_cdpp_cache.parquet (TARS sys_score ≈ 0, TESS-SPOC confirmed)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BLUE        = "#2a78d6"
BLUE_LIGHT  = "#86b6ef"
RED         = "#e34948"
ORANGE      = "#d97706"

# ── Load data ─────────────────────────────────────────────────────────────

# Quiet: from training parquet
train = pd.read_parquet("training_data.parquet")
train = train.dropna(subset=["cdpp1_0", "tmag", "sector_median"])
train = train[(train["cdpp1_0"] > 0) & (train["cdpp1_0"] < 1e5) & (train["sector_median"] > 0)]

quiet = (train.groupby("tic_id")
               .agg(cdpp=("cdpp1_0", "median"),
                    tmag=("tmag", "median"),
                    flux=("sector_median", "median"))
               .reset_index())
quiet["cdpp_photon"] = 1e6 / np.sqrt(quiet["flux"] * 3600.0)
quiet["excess"]      = quiet["cdpp"] / quiet["cdpp_photon"]
quiet["group"]       = "Quiet (TARS sys > 0.95)"

# Non-quiet: from collected cache
nq = pd.read_parquet("nonquiet_cdpp_cache.parquet")
nq = nq.dropna(subset=["cdpp1_0", "tmag"])
nq = nq[(nq["cdpp1_0"] > 0) & (nq["cdpp1_0"] < 1e5)]

# Estimate photon floor using CDPP directly: cdpp_photon ≈ 1e6/sqrt(F*3600)
# We don't have raw flux for non-quiet; estimate F from Tmag using quiet calibration
# fit log(flux) vs Tmag from quiet sample
from numpy.polynomial import polynomial as P
log_flux  = np.log10(quiet["flux"])
coef = np.polyfit(quiet["tmag"], log_flux, 1)
nq["flux_est"]    = 10 ** np.polyval(coef, nq["tmag"])
nq["cdpp_photon"] = 1e6 / np.sqrt(nq["flux_est"] * 3600.0)
nq["excess"]      = nq["cdpp1_0"] / nq["cdpp_photon"]
nq = nq.rename(columns={"cdpp1_0": "cdpp"})
nq["group"] = "Non-quiet (TARS sys ≈ 0)"

print(f"Quiet stars: {len(quiet):,}  |  Non-quiet stars: {len(nq):,}")
print(f"\nQuiet CDPP excess: median={quiet['excess'].median():.2f}×, mean={quiet['excess'].mean():.2f}×")
print(f"Non-quiet CDPP excess: median={nq['excess'].median():.2f}×, mean={nq['excess'].mean():.2f}×")

print("\nBy Tmag bin:")
for lo, hi in [(7,9),(9,11),(11,13)]:
    q_sub = quiet[(quiet["tmag"]>=lo)&(quiet["tmag"]<hi)]
    n_sub = nq[(nq["tmag"]>=lo)&(nq["tmag"]<hi)]
    if len(q_sub) > 0 and len(n_sub) > 0:
        print(f"  Tmag {lo}-{hi}:  quiet median={q_sub['excess'].median():.2f}×  "
              f"non-quiet median={n_sub['excess'].median():.2f}×  "
              f"(n_quiet={len(q_sub)}, n_nq={len(n_sub)})")

# ── Figure ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=SURFACE,
                          gridspec_kw={"wspace": 0.4})
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.5, zorder=0)

# ── Panel 1: CDPP vs Tmag — quiet ─────────────────────────────────────────
ax = axes[0]
ax.hexbin(quiet["tmag"], quiet["cdpp"], gridsize=50, cmap="Blues", mincnt=1,
          yscale="log", linewidths=0.2, zorder=2)

bins = np.arange(7, 13.25, 0.5)
bin_c, floor_med = [], []
for lo in bins[:-1]:
    hi = lo + 0.5
    sub = quiet[(quiet["tmag"]>=lo)&(quiet["tmag"]<hi)]
    if len(sub) >= 5:
        bin_c.append(lo+0.25); floor_med.append(sub["cdpp_photon"].median())
ax.plot(bin_c, floor_med, color=RED, lw=2, ls="--", zorder=4, label="Photon floor")

pct_2x = (quiet["excess"] < 2.0).mean() * 100
ax.text(0.04, 0.97, f"n = {len(quiet):,}\n{pct_2x:.0f}% within 2× floor",
        transform=ax.transAxes, fontsize=8.5, color=INK_PRIMARY,
        va="top", bbox=dict(fc=SURFACE, ec=GRIDLINE, pad=3, lw=0.8))
ax.set_xlabel("TESS magnitude", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("CDPP 1-hour (ppm)", fontsize=10, color=INK_PRIMARY)
ax.set_title("Quiet stars (sys > 0.95)\nCDPP vs magnitude", fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRIDLINE)
ax.set_xlim(7, 13)

# ── Panel 2: CDPP vs Tmag — non-quiet ─────────────────────────────────────
ax = axes[1]
ax.hexbin(nq["tmag"], nq["cdpp"], gridsize=30, cmap="Oranges", mincnt=1,
          yscale="log", linewidths=0.2, zorder=2)

# Same photon floor (Tmag-based)
ax.plot(bin_c, floor_med, color=RED, lw=2, ls="--", zorder=4, label="Photon floor")

pct_2x_nq = (nq["excess"] < 2.0).mean() * 100
ax.text(0.04, 0.97, f"n = {len(nq):,}\n{pct_2x_nq:.0f}% within 2× floor",
        transform=ax.transAxes, fontsize=8.5, color=INK_PRIMARY,
        va="top", bbox=dict(fc=SURFACE, ec=GRIDLINE, pad=3, lw=0.8))
ax.set_xlabel("TESS magnitude", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("CDPP 1-hour (ppm)", fontsize=10, color=INK_PRIMARY)
ax.set_title("Non-quiet stars (sys ≈ 0)\nCDPP vs magnitude", fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRIDLINE)
ax.set_xlim(quiet["tmag"].min(), quiet["tmag"].max())

# ── Panel 3: Excess ratio histogram — both groups ────────────────────────
ax = axes[2]
bins_ex = np.linspace(0.5, 8, 55)
ax.hist(quiet["excess"].clip(upper=8), bins=bins_ex,
        color=BLUE, alpha=0.7, edgecolor=SURFACE, lw=0.3,
        label=f"Quiet (sys>0.95), n={len(quiet):,}", zorder=3)
ax.hist(nq["excess"].clip(upper=8), bins=bins_ex,
        color=ORANGE, alpha=0.65, edgecolor=SURFACE, lw=0.3,
        label=f"Non-quiet (sys≈0), n={len(nq):,}", zorder=2)
ax.axvline(1.0, color=RED, lw=1.5, ls="--", zorder=4, label="Photon floor")
ax.axvline(quiet["excess"].median(), color=BLUE, lw=2, ls="-", zorder=5,
           label=f"Quiet median ({quiet['excess'].median():.2f}×)")
ax.axvline(nq["excess"].median(), color=ORANGE, lw=2, ls="-", zorder=5,
           label=f"Non-quiet median ({nq['excess'].median():.2f}×)")
ax.set_xlabel("CDPP1_0 / photon noise floor", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Number of stars", fontsize=10, color=INK_PRIMARY)
ax.set_title("Noise excess above photon floor\nQuiet vs non-quiet comparison",
             fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8, framealpha=0.9, edgecolor=GRIDLINE)
ax.set_xlim(0.5, 8)

fig.suptitle(
    "TARS quiet-star validation: Are sys > 0.95 stars actually photon-noise limited?\n"
    "Compared against non-quiet (sys ≈ 0) stars with matched TESS-SPOC FFI coverage",
    fontsize=11, color=INK_PRIMARY, y=1.02, fontweight="bold"
)

out = "stitch_quiet_vs_nonquiet_noise.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
