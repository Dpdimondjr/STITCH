"""
Compare light curves: TARS quiet (sys_score > 0.95) vs variable (sys_score ≈ 0).
Downloads one sector per star via direct MAST URL (fast).
"""

import io, requests, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

warnings.filterwarnings("ignore")

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BLUE        = "#2563eb"
ORANGE      = "#ea580c"
MAST        = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc"

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
    if len(flux) < 50:
        return None, None
    t    = t - t[0]                     # time from sector start (days)
    flux = flux / np.nanmedian(flux)    # normalise to 1.0
    return t, flux

# ── Stars to plot ─────────────────────────────────────────────────────────────
# (tic_id, sector_to_use, label)
QUIET = [
    (38459458,  30, "TIC 38459458  Tmag 10.96"),
    (30037565,  36, "TIC 30037565  Tmag 9.60"),
    (279157652, 36, "TIC 279157652  Tmag 10.30"),
    (302976612,  6, "TIC 302976612  Tmag 9.77"),
]

VARIABLE = [
    (89524,  20, "TIC 89524  Tmag 10.24  amp=26×"),      # 8 sectors, high amplitude
    (172112435, 20, None),   # actually this is the same as 89524 alias? let me use:
    (296758284, 15, "TIC 296758284  Tmag 11.09  amp=27×"),
    (247528226,  6, "TIC 247528226  Tmag 11.87  amp=26×"),
]

# Fix: TIC 89524 is short (n_secs listed — let me use full TIC ids properly)
VARIABLE = [
    (172112435, 20, "TIC 172112435  Tmag 10.24  amp=26×"),
    (296758284, 15, "TIC 296758284  Tmag 11.09  amp=27×"),
    (247528226,  6, "TIC 247528226  Tmag 11.87  amp=26×"),
    (130924518, 10, "TIC 130924518  Tmag 10.98  amp=25×"),
]

# ── Download all ──────────────────────────────────────────────────────────────
print("Downloading quiet stars...")
quiet_lcs  = [load_lc(tic, sec) for tic, sec, _ in QUIET]
print("Downloading variable stars...")
var_lcs    = [load_lc(tic, sec) for tic, sec, _ in VARIABLE]

# ── Figure: 2 columns × 4 rows ───────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(14, 12), facecolor=SURFACE,
                          gridspec_kw={"hspace": 0.72, "wspace": 0.18})
fig.patch.set_facecolor(SURFACE)

# Column headers as figure text above the top row
for ci, (lbl, color) in enumerate([
    ("TARS quiet  (sys_score > 0.95)", BLUE),
    ("TARS variable  (sys_score ≈ 0)", ORANGE),
]):
    x = 0.27 + ci * 0.50
    fig.text(x, 0.965, lbl, ha="center", va="bottom",
             fontsize=11, color=color, fontweight="bold")

def plot_lc(ax, t, flux, label, color, sec):
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(color=GRIDLINE, linewidth=0.4, zorder=0)
    if t is not None:
        # Break line at time gaps > 1 day (momentum windows, data outages)
        dt = np.diff(t)
        gap = np.where(dt > 1.0)[0] + 1
        segments = np.split(np.arange(len(t)), gap)
        for seg in segments:
            if len(seg) > 1:
                ax.plot(t[seg], flux[seg], lw=0.65, color=color,
                        alpha=0.85, zorder=2, rasterized=True)
        rms = float(np.std(flux))
        ax.text(0.97, 0.93, f"rms = {rms*100:.2f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK_MUTED, fontfamily="monospace")
        # Force y-axis to show absolute values (no +1 offset)
        ax.ticklabel_format(useOffset=False, axis='y')
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED, fontsize=9)
    ax.set_xlabel("Time from sector start (days)", fontsize=8, color=INK_MUTED)
    ax.set_ylabel("Norm. flux", fontsize=8, color=INK_MUTED)
    name = label or ""
    ax.set_title(f"{name}  ·  S{sec:02d}", fontsize=8.5,
                 color=INK_PRIMARY, loc="left", pad=4)

for i in range(4):
    t_q, f_q = quiet_lcs[i]
    tic_q, sec_q, lbl_q = QUIET[i]
    plot_lc(axes[i, 0], t_q, f_q, lbl_q, BLUE, sec_q)

    t_v, f_v = var_lcs[i]
    tic_v, sec_v, lbl_v = VARIABLE[i]
    plot_lc(axes[i, 1], t_v, f_v, lbl_v, ORANGE, sec_v)

fig.suptitle(
    "TESS PDCSAP light curves: TARS quiet vs variable stars\n"
    "All 1800s FFI (TESS-SPOC HLSP), normalised to median = 1.0",
    fontsize=12, color=INK_PRIMARY, y=1.00, fontweight="bold"
)

out = "stitch_sys_score_lc_examples.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"Saved → {out}")
