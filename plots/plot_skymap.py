"""
Sky distribution of STITCH training / val / test stars.
Mollweide projection in equatorial coordinates with ecliptic and galactic
plane overlays. Second panel coloured by n_sectors_total.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic, Galactic
import astropy.units as u
from sklearn.model_selection import train_test_split

SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
MUTED   = "#898781"
GRID    = "#e1e0d9"
C_TRAIN = "#2a78d6"
C_VAL   = "#c47900"
C_TEST  = "#e05c3a"


def ra_to_moll(ra_deg):
    """RA → matplotlib Mollweide x, with RA increasing to the left (standard)."""
    x = np.deg2rad(360.0 - np.asarray(ra_deg, dtype=float))
    x[x > np.pi] -= 2 * np.pi
    return x


def reference_plane(frame, n=2000):
    """Return (x, y_rad) for a great circle in equatorial Mollweide coords."""
    lon = np.linspace(0, 360, n)
    if frame == "ecliptic":
        coords = SkyCoord(lon=lon * u.deg, lat=np.zeros(n) * u.deg,
                          frame=GeocentricMeanEcliptic).icrs
    else:  # galactic
        coords = SkyCoord(l=lon * u.deg, b=np.zeros(n) * u.deg,
                          frame=Galactic).icrs
    x = ra_to_moll(coords.ra.deg)
    y = np.deg2rad(coords.dec.deg)
    # Break line at wrap-arounds (|Δx| > π)
    gap = np.abs(np.diff(x)) > np.pi
    x_seg, y_seg = list(x), list(y)
    for i in np.where(gap)[0][::-1]:
        x_seg.insert(i + 1, np.nan)
        y_seg.insert(i + 1, np.nan)
    return np.array(x_seg), np.array(y_seg)


# ── Load data & reconstruct split ─────────────────────────────────────────────

print("Loading training data...")
df = pd.read_parquet("training_data.parquet")

star_info = (df.groupby("tic_id")
               .agg(ra=("ra", "first"), dec=("dec", "first"),
                    n_sectors=("sector", "nunique"),
                    dominant_cam=("cam", lambda x: x.mode()[0]))
               .reset_index())

train_tics, temp_tics = train_test_split(
    star_info["tic_id"], test_size=0.2,
    stratify=star_info["dominant_cam"], random_state=42,
)
temp_cam = star_info[star_info["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics, test_size=0.5, stratify=temp_cam.values, random_state=42,
)

star_info["split"] = "train"
star_info.loc[star_info["tic_id"].isin(val_tics), "split"] = "val"
star_info.loc[star_info["tic_id"].isin(test_tics), "split"] = "test"
print(f"  train {(star_info['split']=='train').sum():,}  "
      f"val {(star_info['split']=='val').sum():,}  "
      f"test {(star_info['split']=='test').sum():,}")

# ── Reference planes ──────────────────────────────────────────────────────────

ecl_x, ecl_y = reference_plane("ecliptic")
gal_x, gal_y = reference_plane("galactic")

# ── Figure: two panels ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 10), facecolor=SURFACE)
fig.subplots_adjust(hspace=0.08, top=0.93, bottom=0.04, left=0.04, right=0.96)

# Panel 1: coloured by split
ax1 = fig.add_subplot(211, projection="mollweide")
ax1.set_facecolor(SURFACE)
ax1.tick_params(labelcolor=MUTED, labelsize=8)
ax1.grid(color=GRID, lw=0.5, alpha=0.7)

split_cfg = [
    ("train", C_TRAIN, 0.18, 1.0),
    ("val",   C_VAL,   0.55, 2.0),
    ("test",  C_TEST,  0.70, 2.5),
]
for split, color, alpha, ms in split_cfg:
    s = star_info[star_info["split"] == split]
    ax1.scatter(ra_to_moll(s["ra"]), np.deg2rad(s["dec"]),
                s=ms, color=color, alpha=alpha, rasterized=True,
                label=f"{split}  ({len(s):,})")

ax1.plot(ecl_x, ecl_y, "-", color="#d62728", lw=1.2, alpha=0.7, label="Ecliptic")
ax1.plot(gal_x, gal_y, "--", color="#2ca02c", lw=1.0, alpha=0.6, label="Galactic plane")

ax1.set_title("Sky distribution of STITCH training stars  (equatorial, Mollweide)",
              color=INK, fontsize=11, fontweight="600", pad=8)
ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9, edgecolor=GRID,
           markerscale=4, ncol=5)

# Panel 2: coloured by n_sectors
ax2 = fig.add_subplot(212, projection="mollweide")
ax2.set_facecolor(SURFACE)
ax2.tick_params(labelcolor=MUTED, labelsize=8)
ax2.grid(color=GRID, lw=0.5, alpha=0.7)

cmap = plt.cm.plasma
norm = mcolors.LogNorm(vmin=2, vmax=star_info["n_sectors"].max())
sc = ax2.scatter(ra_to_moll(star_info["ra"]), np.deg2rad(star_info["dec"]),
                 s=1.0, c=star_info["n_sectors"], cmap=cmap, norm=norm,
                 alpha=0.4, rasterized=True)

ax2.plot(ecl_x, ecl_y, "-", color="#d62728", lw=1.2, alpha=0.7)
ax2.plot(gal_x, gal_y, "--", color="#2ca02c", lw=1.0, alpha=0.6)

cbar = fig.colorbar(sc, ax=ax2, orientation="horizontal", pad=0.04,
                    fraction=0.025, aspect=40)
cbar.set_label("sectors observed (n_sectors_total)", color=MUTED, fontsize=8.5)
cbar.ax.tick_params(labelcolor=MUTED, labelsize=8)

ax2.set_title("Same stars coloured by number of TESS sectors",
              color=INK, fontsize=11, fontweight="600", pad=8)

# RA tick labels (convert back from Mollweide convention)
for ax in (ax1, ax2):
    ax.set_xticklabels(["14h", "16h", "18h", "20h", "22h", "0h",
                         "2h", "4h", "6h", "8h", "10h"], fontsize=7)

out = "stitch_skymap.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"Saved → {out}")
