"""
Noise floor analysis: at what Tmag do flux_offset corrections fall below
the background variation noise floor?

Two noise floor estimates:
  1. intra-sector: bkg_rms / sqrt(N_cadences) / star_flux
     — how much the background fluctuates within one sector (random noise on sector_median)
  2. inter-sector: std(median_sap_bkg across sectors) / star_flux
     — how much the background LEVEL changes between sectors (systematic contamination of flux_offset)

The inter-sector one is what directly contaminates LOO flux_offset labels.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_parquet("training_data_topup_bkg.parquet")
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["sector_median"] > 0]

has_bkg = df["median_sap_bkg"].notna() & df["bkg_rms"].notna()
d = df[has_bkg].copy()
print(f"Records with background data: {len(d):,} ({len(d)/len(df)*100:.1f}%)")
print(f"Unique stars with background data: {d['tic_id'].nunique():,}")

# ── Noise floor 1: intra-sector background scatter ────────────────────────────
# SAP_BKG per cadence has RMS=bkg_rms. The sector_median is computed over
# N_cadences cadences, so background noise on sector_median ≈ bkg_rms/sqrt(N).
# TESS 2-min cadence, 27-day sector: ~19440 cadences. 30-min: ~1296.
# We don't know cadence mode per star, use conservative 1296 (30-min).
N_CADENCES = 1296
d["intra_noise"] = d["bkg_rms"] / np.sqrt(N_CADENCES) / d["sector_median"]

# ── Noise floor 2: inter-sector background variation ─────────────────────────
# For each star, compute std of median_sap_bkg across its sectors.
# This is the sector-to-sector swing in background level → directly biases flux_offset.
star_bkg_std = (d.groupby("tic_id")["median_sap_bkg"]
                  .std()
                  .rename("bkg_inter_std"))
star_flux_mean = (d.groupby("tic_id")["sector_median"]
                    .mean()
                    .rename("star_flux_mean"))
star_stats = pd.concat([star_bkg_std, star_flux_mean], axis=1).dropna()
star_stats["inter_noise"] = star_stats["bkg_inter_std"] / star_stats["star_flux_mean"]

# Also: actual flux_offset std per star (the signal we want to measure)
star_offset_std = (d.groupby("tic_id")["flux_offset"]
                     .std()
                     .rename("offset_std"))
star_tmag = d.groupby("tic_id")["tmag"].median().rename("tmag")
star_stats = star_stats.join(star_offset_std).join(star_tmag)
star_stats["snr"] = star_stats["offset_std"] / star_stats["inter_noise"]

print(f"\nStars with inter-sector background data: {len(star_stats):,}")
print(f"\nMedian inter-sector noise floor by Tmag:")
bins   = [6, 8, 9, 10, 11, 12, 13]
labels = ["6-8","8-9","9-10","10-11","11-12","12-13"]
star_stats["tmag_bin"] = pd.cut(star_stats["tmag"], bins=bins, labels=labels)
g = star_stats.groupby("tmag_bin", observed=True).agg(
    n=("inter_noise","count"),
    noise_floor=("inter_noise","median"),
    signal=("offset_std","median"),
    snr=("snr","median")
)
print(g.to_string(float_format="%.4f"))

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="#0b0f1a")
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

BLUE  = "#4f8ef7"
GREEN = "#34c580"
AMBER = "#f59e0b"
RED   = "#ef4444"

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor("#060a12")
    ax.set_xlabel(xlabel, color="#7a9cc4", fontsize=10)
    ax.set_ylabel(ylabel, color="#7a9cc4", fontsize=10)
    ax.set_title(title, color="#c4d4ee", fontsize=11, pad=7)
    ax.tick_params(colors="#3a5070", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

# ── Panel 1: background fraction vs Tmag ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
d["bkg_frac"] = d["median_sap_bkg"] / d["sector_median"]
tmag_bin = pd.cut(d["tmag"], bins=bins, labels=labels)
bkg_by_tmag = d.groupby(tmag_bin, observed=True)["bkg_frac"].describe()

xs = range(len(bkg_by_tmag))
ax.bar(xs, bkg_by_tmag["50%"]*100, color=BLUE, alpha=0.7, label="median")
ax.errorbar(xs,
            bkg_by_tmag["50%"]*100,
            yerr=[(bkg_by_tmag["50%"]-bkg_by_tmag["25%"])*100,
                  (bkg_by_tmag["75%"]-bkg_by_tmag["50%"])*100],
            fmt="none", color="#c4d4ee", lw=1.5, capsize=4)
ax.axhline(100, color=RED, lw=1, ls="--", alpha=0.5, label="bkg = star flux")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, color="#7a9cc4")
ax.legend(fontsize=8, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
style(ax, "Tmag", "Background / star flux (%)",
      "Background level relative to star")

# ── Panel 2: inter-sector noise floor vs signal ───────────────────────────────
ax = fig.add_subplot(gs[0, 1])
tmag_g = star_stats.groupby("tmag_bin", observed=True)
xs = range(len(labels))

noise_med = tmag_g["inter_noise"].median() * 100
signal_med = tmag_g["offset_std"].median() * 100

ax.plot(xs, signal_med, color=GREEN, lw=2, marker="o", ms=6,
        label="Signal: σ(flux_offset) per star")
ax.plot(xs, noise_med, color=RED, lw=2, marker="s", ms=6,
        label="Noise floor: σ(bkg_median) / star_flux")
ax.fill_between(xs, noise_med, signal_med,
                where=signal_med > noise_med,
                alpha=0.15, color=GREEN, label="signal > noise")
ax.fill_between(xs, noise_med, signal_med,
                where=signal_med < noise_med,
                alpha=0.15, color=RED, label="noise > signal")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, color="#7a9cc4")
ax.legend(fontsize=8, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
style(ax, "Tmag", "Fractional variation (%)",
      "Signal vs background noise floor\n(inter-sector background variation)")

# ── Panel 3: SNR of correction by Tmag ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
snr_med = tmag_g["snr"].median()
snr_p25 = tmag_g["snr"].quantile(0.25)
snr_p75 = tmag_g["snr"].quantile(0.75)
colors = [GREEN if v >= 1 else RED for v in snr_med]
bars = ax.bar(xs, snr_med, color=colors, alpha=0.75)
ax.errorbar(xs, snr_med,
            yerr=[snr_med - snr_p25, snr_p75 - snr_med],
            fmt="none", color="#c4d4ee", lw=1.5, capsize=4)
ax.axhline(1.0, color="white", lw=1.2, ls="--", alpha=0.6, label="SNR = 1 (noise floor)")
ax.axhline(3.0, color=AMBER, lw=1, ls=":", alpha=0.5, label="SNR = 3 (confident)")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8, color="#7a9cc4")
ax.legend(fontsize=8, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
for x, (_, row) in zip(xs, g.iterrows()):
    ax.text(x, 0.05, f"n={int(row['n'])}", ha="center", va="bottom",
            fontsize=7, color="#3a5070", transform=ax.get_xaxis_transform())
style(ax, "Tmag", "SNR  (signal / noise floor)",
      "Correction SNR by brightness\n(green = above noise floor)")

# ── Panel 4: scatter of inter-sector noise vs |flux_offset-1| ─────────────────
ax = fig.add_subplot(gs[1, 1])
samp = star_stats.dropna().sample(min(5000, len(star_stats)), random_state=42)
sc = ax.scatter(samp["inter_noise"]*100,
                samp["offset_std"]*100,
                c=samp["tmag"], cmap="plasma",
                s=4, alpha=0.6, rasterized=True,
                vmin=6, vmax=13)
lim = max(samp["inter_noise"].max(), samp["offset_std"].max()) * 100 * 1.05
ax.plot([0, lim], [0, lim], color="white", lw=0.8, alpha=0.4, ls="--",
        label="signal = noise")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Tmag", color="#c4d4ee", fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor="#7a9cc4", labelsize=8)
ax.legend(fontsize=8, framealpha=0.2, labelcolor="#c4d4ee",
          facecolor="#0f1c30", edgecolor="#1c2d45")
style(ax, "Background noise floor (%)", "flux_offset σ per star (%)",
      "Per-star: signal vs noise  (colour = Tmag)\npoints above line: signal > noise")

fig.suptitle("STITCH noise floor analysis — is the correction above sky background variation?",
             color="#c4d4ee", fontsize=13, y=0.99)

out = "eval/noise_floor.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
print(f"\nSaved → {out}")
