"""
Between-sector DC scatter: TARS quiet vs non-quiet stars.
Both groups use TESS-SPOC FFI (1800s cadence) and ≥3 sectors.

Metric: std(sector_median / mean_sector_median) per star
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BLUE        = "#2a78d6"
BLUE_LIGHT  = "#86b6ef"
ORANGE      = "#e07b2a"
RED         = "#e34948"

# ── Load ──────────────────────────────────────────────────────────────────

quiet = pd.read_parquet("quiet_scatter_cache.parquet")
nq    = pd.read_parquet("nonquiet_scatter_cache.parquet")

# Clip outliers for display
quiet = quiet[quiet["scatter"] < 0.15]
nq    = nq[nq["scatter"] < 0.15]

q_med = quiet["scatter"].median()
n_med = nq["scatter"].median()

print(f"Quiet   n={len(quiet):,}  median scatter={q_med*100:.2f}%  (ALL ≥3 sectors)")
print(f"Non-quiet n={len(nq):,}  median scatter={n_med*100:.2f}%  (sectors 1-10)")
print(f"Ratio: {q_med/n_med:.2f}×")

# Tmag-matched comparison
for lo, hi in [(7,9),(9,11),(11,13)]:
    q_s = quiet[(quiet["tmag"]>=lo)&(quiet["tmag"]<hi)]["scatter"]
    n_s = nq[(nq["tmag"]>=lo)&(nq["tmag"]<hi)]["scatter"]
    if len(q_s)>0 and len(n_s)>0:
        print(f"  Tmag {lo}-{hi}: quiet={q_s.median()*100:.2f}%  nq={n_s.median()*100:.2f}%  "
              f"n=({len(q_s)},{len(n_s)})")

# ── Figure ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE,
                          gridspec_kw={"wspace": 0.38})
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.5, zorder=0)

# ── Panel 1: Overlaid histogram ───────────────────────────────────────────
ax = axes[0]

bins_pct = np.linspace(0, 8, 65)   # in % units

ax.hist(quiet["scatter"]*100, bins=bins_pct,
        color=BLUE, alpha=0.65, edgecolor=SURFACE, lw=0.3, zorder=2,
        label=f"Quiet (TARS sys > 0.95), n={len(quiet):,}", density=True)
ax.hist(nq["scatter"]*100, bins=bins_pct,
        color=ORANGE, alpha=0.65, edgecolor=SURFACE, lw=0.3, zorder=3,
        label=f"Non-quiet (TARS sys ≈ 0), n={len(nq):,}", density=True)

ax.axvline(q_med*100, color=BLUE,   lw=2.0, ls="-",  zorder=4)
ax.axvline(n_med*100, color=ORANGE, lw=2.0, ls="-",  zorder=4)

# Annotation arrows
ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.5
ax.annotate(f"Quiet median\n{q_med*100:.2f}%",
            xy=(q_med*100, ymax*0.6), xytext=(q_med*100+0.9, ymax*0.75),
            fontsize=8.5, color=BLUE, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
ax.annotate(f"Non-quiet median\n{n_med*100:.2f}%",
            xy=(n_med*100, ymax*0.45), xytext=(n_med*100+0.9, ymax*0.55),
            fontsize=8.5, color=ORANGE, ha="left",
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))

ax.set_xlabel("Between-sector scatter (%)", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Density", fontsize=10, color=INK_PRIMARY)
ax.set_title(
    "Between-sector DC scatter: quiet vs non-quiet\n"
    "std(sector_median / mean_sector_median) per star",
    fontsize=10, color=INK_PRIMARY, loc="left"
)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=GRIDLINE)
ax.set_xlim(0, 8)

# ── Panel 2: Scatter vs Tmag (median per bin) ─────────────────────────────
ax = axes[1]

tmag_bins = np.arange(7, 14, 1.0)  # 1-mag bins to reduce noise
centers, q_meds, n_meds, q_lo, q_hi, n_lo, n_hi = [], [], [], [], [], [], []
for lo in tmag_bins[:-1]:
    hi = lo + 1.0
    qb = quiet[(quiet["tmag"]>=lo)&(quiet["tmag"]<hi)]["scatter"] * 100
    nb = nq[(nq["tmag"]>=lo)&(nq["tmag"]<hi)]["scatter"] * 100
    if len(qb) >= 5 and len(nb) >= 3:
        centers.append(lo + 0.5)
        q_meds.append(qb.median()); q_lo.append(qb.quantile(0.25)); q_hi.append(qb.quantile(0.75))
        n_meds.append(nb.median()); n_lo.append(nb.quantile(0.25)); n_hi.append(nb.quantile(0.75))

ax.fill_between(centers, q_lo, q_hi, color=BLUE, alpha=0.15, zorder=1)
ax.fill_between(centers, n_lo, n_hi, color=ORANGE, alpha=0.15, zorder=1)
ax.plot(centers, q_meds, color=BLUE,   lw=2.0, marker="o", ms=5, zorder=3,
        label="Quiet (sys > 0.95)  — median ± IQR")
ax.plot(centers, n_meds, color=ORANGE, lw=2.0, marker="s", ms=5, zorder=3,
        label=f"Non-quiet (sys ≈ 0), n={len(nq):,}")

ax.set_xlabel("TESS magnitude (Tmag)", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Median scatter (%)", fontsize=10, color=INK_PRIMARY)
ax.set_title(
    "Scatter vs brightness — Tmag-matched comparison\n"
    "Similar scatter → DC offsets reflect instrumental PRF, not stellar variability",
    fontsize=10, color=INK_PRIMARY, loc="left"
)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=GRIDLINE)
ax.set_xlim(7, 13)

fig.suptitle(
    "TARS quiet-star validation: between-sector DC scatter comparison\n"
    "Quiet & non-quiet stars have similar scatter at matched Tmag → quiet-star LOO labels are instrument-dominated, not astrophysical",
    fontsize=11, color=INK_PRIMARY, y=1.02, fontweight="bold"
)

out = "stitch_scatter_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
