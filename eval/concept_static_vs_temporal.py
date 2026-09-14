"""
Simple concept figure: static detector pattern vs sector-to-sector variation.
Shows why a flat-field correction is insufficient — the time-variable component
dominates by 4.4x.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PARQUET = "training_data_topup_pdc.parquet"

df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["flux_offset","col","row"])
df = df[(df.flux_offset > 0.85) & (df.flux_offset < 1.15)]
cam4 = df[df.cam == 4].copy()

# Per-star: temporal spread (σ across sectors)
temporal_std = cam4.groupby("tic_id")["flux_offset"].std().dropna()

# Per-star: mean offset (static bias amplitude)
static_mean  = cam4.groupby("tic_id")["flux_offset"].mean()
static_bias  = (static_mean - 1.0).abs().dropna()

# Align
common = temporal_std.index.intersection(static_bias.index)
temporal_std = temporal_std[common]
static_bias  = static_bias[common]

print(f"Median temporal σ:   {temporal_std.median()*100:.3f}%")
print(f"Median static bias:  {static_bias.median()*100:.3f}%")
print(f"Ratio:               {temporal_std.median()/static_bias.median():.1f}×")

# ── Figure ────────────────────────────────────────────────────────────────────
BG   = "#0b0f1a"
BLUE = "#3b8ef0"
RED  = "#f87171"
TEXT = "#dce8f8"
MUTE = "#7a9cc4"

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)

bins = np.linspace(0, 4.5, 60)

n1, _, _ = ax.hist(static_bias  * 100,
                   bins=bins, density=True,
                   color=BLUE, alpha=0.75, label="Static detector pattern\n(time-averaged offset per star)")
n2, _, _ = ax.hist(temporal_std * 100,
                   bins=bins, density=True,
                   color=RED,  alpha=0.75, label="Sector-to-sector variation\n(same star, different sectors)")

med_s = static_bias.median()  * 100
med_t = temporal_std.median() * 100
ymax  = max(n1.max(), n2.max())

ax.axvline(med_s, color=BLUE, lw=2,   ls="--", alpha=0.9)
ax.axvline(med_t, color=RED,  lw=2,   ls="--", alpha=0.9)

ax.text(med_s + 0.06, ymax * 0.92, f"median\n{med_s:.2f}%",
        color=BLUE, fontsize=10, va="top")
ax.text(med_t + 0.06, ymax * 0.72, f"median\n{med_t:.2f}%",
        color=RED,  fontsize=10, va="top")

# Arrow annotating the ratio
ax.annotate("",
    xy=(med_t, ymax * 0.55), xytext=(med_s, ymax * 0.55),
    arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax.text((med_s + med_t) / 2, ymax * 0.58,
        f"{med_t/med_s:.1f}× larger",
        color="white", fontsize=11, ha="center", fontweight="bold")

ax.set_xlabel("Flux offset magnitude (%)", color=MUTE, fontsize=12)
ax.set_ylabel("Density", color=MUTE, fontsize=12)
ax.tick_params(colors=MUTE, labelsize=10)
for sp in ax.spines.values():
    sp.set_edgecolor("#1c2d45")

leg = ax.legend(fontsize=10.5, facecolor="#0d1a30", edgecolor="#1c2d45",
                labelcolor=TEXT, loc="upper right")

ax.set_title(
    "A static flat-field captures the small part. STITCH corrects the large part.",
    color=TEXT, fontsize=13, pad=14
)
ax.text(0.01, 0.97,
        "Cam 4 CVZ stars  ·  each point = one star",
        transform=ax.transAxes, color=MUTE, fontsize=8.5, va="top")

plt.tight_layout()
out = "eval/concept_static_vs_temporal.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
