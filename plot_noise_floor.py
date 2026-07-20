"""
Sanity check: do TARS quiet stars sit near the photon noise floor?

Photon noise for a 1-hour integration:
  CDPP_photon (ppm) = 1e6 / sqrt(F * 3600)
where F = sector_median in e-/s (already in the parquet).

If CDPP1_0 ≈ CDPP_photon, the stars are as quiet as physics allows.
Excess above the floor = astrophysical variability + residual systematics.
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
BLUE_XLIGHT = "#cde2fb"
RED         = "#e34948"

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["cdpp1_0", "tmag", "sector_median"])
df = df[(df["cdpp1_0"] > 0) & (df["cdpp1_0"] < 1e5) & (df["sector_median"] > 0)]

# One record per (star, sector) — already the case; use median across sectors per star
per_star = (df.groupby("tic_id")
              .agg(cdpp=("cdpp1_0", "median"),
                   tmag=("tmag", "median"),
                   flux=("sector_median", "median"))
              .reset_index())

# Theoretical photon noise floor from actual measured flux
# CDPP_photon (ppm) = 1e6 / sqrt(flux_e_per_s * 3600s)
per_star["cdpp_photon"] = 1e6 / np.sqrt(per_star["flux"] * 3600.0)

# Noise excess ratio: how many times above the photon floor?
per_star["excess"] = per_star["cdpp"] / per_star["cdpp_photon"]

print(f"Stars: {len(per_star):,}")
print(f"\nCDPP1_0 / photon_noise_floor:")
print(f"  Median excess: {per_star['excess'].median():.2f}×")
print(f"  Mean excess:   {per_star['excess'].mean():.2f}×")
print(f"  % within 1.5× floor: {(per_star['excess'] < 1.5).mean()*100:.0f}%")
print(f"  % within 2.0× floor: {(per_star['excess'] < 2.0).mean()*100:.0f}%")
print(f"  % within 3.0× floor: {(per_star['excess'] < 3.0).mean()*100:.0f}%")

print(f"\nBy Tmag bin:")
for lo, hi in [(7,9),(9,11),(11,13)]:
    sub = per_star[(per_star["tmag"]>=lo) & (per_star["tmag"]<hi)]
    print(f"  Tmag {lo}–{hi}: n={len(sub):,}  "
          f"median CDPP={sub['cdpp'].median():.0f}ppm  "
          f"floor={sub['cdpp_photon'].median():.0f}ppm  "
          f"excess={sub['excess'].median():.2f}×")

# ── Figure: 2 panels ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE,
                          gridspec_kw={"wspace": 0.38})

for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.5, zorder=0)

# ── Panel 1: CDPP vs Tmag (hexbin density + floor) ───────────────────────────

ax = axes[0]

hb = ax.hexbin(per_star["tmag"], per_star["cdpp"],
               gridsize=60, cmap="Blues", mincnt=1,
               xscale="linear", yscale="log",
               linewidths=0.2, zorder=2)
plt.colorbar(hb, ax=ax, label="Stars per bin", pad=0.01)

# Theoretical floor as a smooth curve over Tmag
tmag_line = np.linspace(7, 13, 200)
# Use median flux-to-tmag mapping from data to anchor the floor
# Fit: log(flux) ~ -0.4*(tmag - tmag_ref)*ln(10) + log(flux_ref)
log_flux  = np.log(per_star["flux"])
log_floor = np.log(per_star["cdpp_photon"])
# Median floor at each tmag bin for the curve
bins = np.arange(7, 13.25, 0.25)
bin_centers, floor_medians = [], []
for lo in bins[:-1]:
    hi = lo + 0.25
    sub = per_star[(per_star["tmag"]>=lo) & (per_star["tmag"]<hi)]
    if len(sub) >= 10:
        bin_centers.append(lo + 0.125)
        floor_medians.append(sub["cdpp_photon"].median())

ax.plot(bin_centers, floor_medians,
        color=RED, lw=2.0, ls="--", zorder=4, label="Photon noise floor")
ax.plot(bin_centers, [f*1.5 for f in floor_medians],
        color=RED, lw=1.0, ls=":", zorder=4, alpha=0.6, label="1.5× floor")

ax.set_xlabel("TESS magnitude (Tmag)", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("CDPP 1-hour (ppm)", fontsize=10, color=INK_PRIMARY)
ax.set_title("TARS quiet stars: measured CDPP vs photon noise floor",
             fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8.5, framealpha=0.85, edgecolor=GRIDLINE)
ax.set_xlim(7, 13)

# ── Panel 2: histogram of excess ratio ───────────────────────────────────────

ax = axes[1]

excess_clip = per_star["excess"].clip(upper=6)
bins_ex = np.linspace(0.5, 6, 60)
ax.hist(excess_clip, bins=bins_ex, color=BLUE, alpha=0.75,
        edgecolor=SURFACE, linewidth=0.3, zorder=2)
ax.axvline(1.0, color=RED,      lw=1.5, ls="--", zorder=3, label="Photon floor (1.0×)")
ax.axvline(per_star["excess"].median(), color=BLUE, lw=2.0, ls="-",
           zorder=4, label=f"Median ({per_star['excess'].median():.2f}×)")
ax.axvspan(0.5, 1.5, color=RED, alpha=0.07, zorder=1, label="Within 1.5× floor")

pct_1p5 = (per_star["excess"] < 1.5).mean()*100
ax.text(0.97, 0.93, f"{pct_1p5:.0f}% of stars\nwithin 1.5× floor",
        transform=ax.transAxes, fontsize=9, color=RED,
        ha="right", va="top",
        bbox=dict(fc=SURFACE, ec=GRIDLINE, pad=4, lw=0.8))

ax.set_xlabel("CDPP1_0 / photon noise floor", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Number of stars", fontsize=10, color=INK_PRIMARY)
ax.set_title("Noise excess above photon floor\n(1.0 = perfectly photon-noise limited)",
             fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8.5, framealpha=0.85, edgecolor=GRIDLINE)

fig.suptitle("TARS quiet star sanity check — are selected stars near the photon noise limit?",
             fontsize=11, color=INK_PRIMARY, y=1.02, fontweight="bold")

out = "stitch_noise_floor.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
