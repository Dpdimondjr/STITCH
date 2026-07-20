"""
Light curves across the TARS systematic_score spectrum:
sys ≈ 0.0, 0.5, 0.85, >0.95  —  all in one figure.
"""

import io, requests, warnings, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits

warnings.filterwarnings("ignore")

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"

# Palette: one color per sys_score band, dark→light as score increases
COLORS = {
    "~0.00": "#e34948",   # red   — clearly variable
    "~0.50": "#d97706",   # amber — mixed
    "~0.85": "#2a78d6",   # blue  — mostly systematic
    ">0.95": "#16a34a",   # green — quiet / selected
}

MAST = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc"

def tic_url(tic_id, sector):
    t = f"{int(tic_id):016d}"
    g = [t[0:4], t[4:8], t[8:12], t[12:16]]
    fn = f"hlsp_tess-spoc_tess_phot_{t}-s{sector:04d}_tess_v1_lc.fits"
    return f"{MAST}/s{sector:04d}/target/{'/'.join(g)}/{fn}"

def load_lc(tic_id, sector):
    r = requests.get(tic_url(tic_id, sector), timeout=30)
    if r.status_code != 200:
        return None, None
    with fits.open(io.BytesIO(r.content)) as h:
        t    = h[1].data["TIME"].astype(float)
        flux = h[1].data["PDCSAP_FLUX"].astype(float)
        qual = h[1].data["QUALITY"].astype(int)
    mask = (qual == 0) & np.isfinite(flux) & np.isfinite(t)
    t, flux = t[mask], flux[mask]
    if len(flux) < 30:
        return None, None
    t    = t - t[0]
    flux = flux / np.nanmedian(flux)
    return t, flux

# ── Stars: (tic_id, sector, sys_score_label, sys_score_value) ────────────────
# From TARS table 4 sample + training data for the quiet end
STARS = [
    (319310550,  5, "~0.00", 0.000,  10.54),   # clearly variable
    (294093629, 34, "~0.50", 0.470,   9.37),   # ambiguous
    (22713412,  76, "~0.85", 0.890,  10.92),   # mostly systematic
    (38459458,  30, ">0.95", 0.980,  10.96),   # quiet / training set
]

print("Downloading light curves...")
lcs = []
for tic, sec, label, score, tmag in STARS:
    print(f"  TIC {tic}  sys={score:.2f}  S{sec:02d}...", flush=True)
    t, flux = load_lc(tic, sec)
    lcs.append((t, flux, tic, sec, label, score, tmag))
    if t is not None:
        print(f"    → {len(t)} pts, rms={np.std(flux)*100:.2f}%")
    else:
        print(f"    → no data")

# ── Figure: 4-panel vertical strip ───────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor=SURFACE)
gs  = gridspec.GridSpec(4, 1, hspace=0.55, left=0.08, right=0.97,
                         top=0.91, bottom=0.06)

for i, (t, flux, tic, sec, label, score, tmag) in enumerate(lcs):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    ax.grid(color=GRIDLINE, linewidth=0.4, zorder=0)
    ax.ticklabel_format(useOffset=False, axis="y")

    color = COLORS[label]

    if t is not None:
        # Break line at gaps > 1 day
        dt   = np.diff(t)
        gaps = np.where(dt > 1.0)[0] + 1
        for seg in np.split(np.arange(len(t)), gaps):
            if len(seg) > 1:
                ax.plot(t[seg], flux[seg], lw=0.7, color=color,
                        alpha=0.9, zorder=2, rasterized=True)
        rms = np.std(flux) * 100
        ax.text(0.985, 0.88, f"rms = {rms:.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK_MUTED, fontfamily="monospace")
    else:
        ax.text(0.5, 0.5, "download failed", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED)

    # Score badge on the left
    ax.text(-0.065, 0.5, f"sys\n{label}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=color,
            fontfamily="monospace",
            bbox=dict(fc=SURFACE, ec=color, lw=1.2, pad=3.5,
                      boxstyle="round,pad=0.35"))

    ax.set_title(
        f"TIC {tic}   Tmag = {tmag:.2f}   sector {sec}   "
        f"systematic_score = {score:.3f}",
        fontsize=9, color=INK_PRIMARY, loc="left", pad=4
    )
    if i == 3:
        ax.set_xlabel("Time from sector start (days)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("Norm. flux", fontsize=8.5, color=INK_MUTED)

fig.suptitle(
    "TARS systematic_score spectrum — TESS PDCSAP light curves\n"
    "From clearly astrophysical variability (red) to instrument-dominated / quiet (green)",
    fontsize=11.5, color=INK_PRIMARY, fontweight="bold", y=0.975
)

# Horizontal colour key at top
for label, color in COLORS.items():
    pass  # drawn via axis badges; no separate legend needed

out = "stitch_sys_score_spectrum.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
