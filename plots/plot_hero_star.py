"""
Hero star: TIC 298091708 — highest-improvement star in held-out test set.
Shows sector-median before/after STITCH, plus downloads one representative
sector LC to make the plot tangible.
"""

import io, json, requests, warnings
import numpy as np, pandas as pd
import torch, zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
C_RAW       = "#9e9e9e"
C_STITCH    = "#2166ac"
C_ENDPOINT  = "#d6604d"
MAST        = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tess-spoc"

HERO_TIC = 298091708

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading model...")
ckpt = torch.load("stitch_nsf.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(**{k: cfg[k] for k in
    ["features","context","transforms","hidden_features","bins"]})
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means = ckpt["means"]; stds = ckpt["stds"]
y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS = ckpt["cam_cols"]; CCD_COLS = ckpt["ccd_cols"]

# ── 2. Get hero star's training rows ──────────────────────────────────────────
print("Loading data for hero star...")
df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Verify it's in test set
star_cam = (df.groupby("tic_id")["cam"].agg(lambda x: x.mode()[0])
              .reset_index().rename(columns={"cam":"dominant_cam"}))
_, temp = train_test_split(star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=42)
temp_cam = star_cam[star_cam["tic_id"].isin(temp)]["dominant_cam"]
_, test_t = train_test_split(temp, test_size=0.5,
    stratify=temp_cam.values, random_state=42)
test_tics = set(test_t)
assert HERO_TIC in test_tics, f"TIC {HERO_TIC} is NOT in the test set!"
print(f"  ✓ TIC {HERO_TIC} confirmed held-out (not seen during training)")

hero_df = df[df["tic_id"] == HERO_TIC].sort_values("sector").copy()
print(f"  Sectors: {sorted(hero_df['sector'].astype(int).tolist())}")
print(f"  Tmag: {hero_df['tmag'].mean():.2f}  Cam{int(hero_df['cam'].mode()[0])}")

# ── 3. Predict offsets ────────────────────────────────────────────────────────
cont   = (hero_df[CONTINUOUS] - means) / stds
cam_oh = pd.get_dummies(hero_df["cam"].astype(int), prefix="cam").reindex(columns=CAM_COLS, fill_value=0)
ccd_oh = pd.get_dummies(hero_df["ccd"].astype(int), prefix="ccd").reindex(columns=CCD_COLS, fill_value=0)
C = pd.concat([cont.reset_index(drop=True),
               cam_oh.reset_index(drop=True),
               ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

with torch.no_grad():
    samples = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
    pred_z     = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_raw = pred_z * y_std + y_mean
pred_std = pred_z_std * y_std
K = 5.0
weight = 1.0 / (1.0 + K * pred_std)
pred_offset = 1.0 * (1 - weight) + pred_raw * weight

hero_df = hero_df.copy()
hero_df["pred_offset"] = pred_offset
hero_df["pred_offset_std"] = pred_std

raw   = hero_df["sector_median"].values
secs  = hero_df["sector"].values.astype(int)
pred  = hero_df["pred_offset"].values
pstd  = hero_df["pred_offset_std"].values

global_ref  = raw.mean()
raw_norm    = raw / global_ref
stitch_norm = (raw / pred) / global_ref

scatter_before = raw_norm.std()
scatter_stitch = stitch_norm.std()
improv = (scatter_before - scatter_stitch) / scatter_before * 100

print(f"  Scatter: {scatter_before*100:.2f}% → {scatter_stitch*100:.2f}%  ({improv:.1f}% improvement)")

# ── 4. Download two sectors for LC insets ─────────────────────────────────────
def tic_url(tic_id, sector):
    t = f"{int(tic_id):016d}"
    g = [t[0:4], t[4:8], t[8:12], t[12:16]]
    fn = f"hlsp_tess-spoc_tess_phot_{t}-s{sector:04d}_tess_v1_lc.fits"
    return f"{MAST}/s{sector:04d}/target/{'/'.join(g)}/{fn}"

def load_lc(tic_id, sector, pred_off, global_ref_e):
    r = requests.get(tic_url(tic_id, sector), timeout=30)
    if r.status_code != 200:
        return None, None, None
    with fits.open(io.BytesIO(r.content)) as h:
        t    = h[1].data["TIME"].astype(float)
        flux = h[1].data["PDCSAP_FLUX"].astype(float)
        qual = h[1].data["QUALITY"].astype(int)
    mask = (qual == 0) & np.isfinite(flux) & np.isfinite(t)
    t, flux = t[mask], flux[mask]
    if len(flux) < 30:
        return None, None, None
    t = t - t[0]
    raw_n   = flux / global_ref_e
    stitch_n = (flux / pred_off) / global_ref_e
    return t, raw_n, stitch_n

# Pick two sectors far apart for contrast
sec_early = int(secs[0])
sec_late  = int(secs[-1])
print(f"  Downloading LC insets: S{sec_early:02d} and S{sec_late:02d}...")

off_early = pred_offset[0]
off_late  = pred_offset[-1]

# Download raw PDCSAP flux (in e-/s); use global_ref in same units
# The sector_median in training data is in e-/s, so global_ref is too
global_ref_epers = float(global_ref)

t_e, raw_e, stitch_e = load_lc(HERO_TIC, sec_early, off_early, global_ref_epers)
t_l, raw_l, stitch_l = load_lc(HERO_TIC, sec_late,  off_late,  global_ref_epers)
print(f"  S{sec_early:02d}: {'ok' if t_e is not None else 'failed'}  "
      f"S{sec_late:02d}: {'ok' if t_l is not None else 'failed'}")

# ── 5. Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
fig.patch.set_facecolor(SURFACE)

# Layout: top half = LC insets (2 panels), bottom = sector median before/after
if t_e is not None and t_l is not None:
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.6, 1.6, 2.0],
                           hspace=0.55, wspace=0.35,
                           left=0.08, right=0.97, top=0.90, bottom=0.06)
    has_insets = True
else:
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5,
                           left=0.08, right=0.97, top=0.90, bottom=0.06)
    has_insets = False

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    ax.grid(color=GRIDLINE, linewidth=0.4, zorder=0)
    ax.ticklabel_format(useOffset=False, axis="y")

if has_insets:
    for col_idx, (t, raw_n, stitch_n, sec, off) in enumerate([
            (t_e, raw_e, stitch_e, sec_early, off_early),
            (t_l, raw_l, stitch_l, sec_late,  off_late),
    ]):
        if t is None:
            continue
        # Raw
        ax_r = fig.add_subplot(gs[0, col_idx])
        style_ax(ax_r)
        ax_r.plot(t, raw_n, lw=0.5, color=C_RAW, alpha=0.8, rasterized=True)
        ax_r.axhline(raw_norm[col_idx if col_idx == 0 else -1],
                     color=C_RAW, lw=1.5, ls="--", alpha=0.6)
        ax_r.set_title(f"Sector {sec}  —  Raw flux", fontsize=9, color=INK_PRIMARY, loc="left")
        ax_r.set_ylabel("Norm. flux", fontsize=8, color=INK_MUTED)

        # STITCH-corrected
        ax_s = fig.add_subplot(gs[1, col_idx])
        style_ax(ax_s)
        ax_s.plot(t, stitch_n, lw=0.5, color=C_STITCH, alpha=0.8, rasterized=True)
        ax_s.axhline(stitch_norm[col_idx if col_idx == 0 else -1],
                     color=C_STITCH, lw=1.5, ls="--", alpha=0.6)
        ax_s.axhline(1.0, color=INK_MUTED, lw=0.8, ls=":", alpha=0.5)
        ax_s.set_title(f"Sector {sec}  —  After STITCH  (÷ {off:.4f})", fontsize=9, color=C_STITCH, loc="left")
        ax_s.set_ylabel("Norm. flux", fontsize=8, color=INK_MUTED)
        ax_s.set_xlabel("Days from sector start", fontsize=8, color=INK_MUTED)

    ax_main = fig.add_subplot(gs[2, :])
else:
    ax_main = fig.add_subplot(gs[0])

# ── Main panel: sector medians ────────────────────────────────────────────────
style_ax(ax_main)
ax_main.axhline(1.0, color=INK_MUTED, lw=0.8, ls="--", alpha=0.5, zorder=1)

# Error bars: predicted offset std (uncertainty)
ax_main.errorbar(secs - 0.15, raw_norm, yerr=None,
                 fmt="o-", color=C_RAW, lw=1.8, ms=7,
                 label=f"Raw  (σ = {scatter_before*100:.2f}%)", zorder=3)
ax_main.errorbar(secs + 0.15, stitch_norm,
                 yerr=pstd / global_ref,
                 fmt="o-", color=C_STITCH, lw=1.8, ms=7, capsize=4, capthick=1.2,
                 label=f"STITCH  (σ = {scatter_stitch*100:.2f}%  ·  {improv:.0f}% improvement)", zorder=4)

# Annotate each sector with its predicted offset
for i, (sec, poff) in enumerate(zip(secs, pred_offset)):
    ax_main.text(sec, stitch_norm[i] + (scatter_before*0.25),
                 f"×{1/poff:.4f}", ha="center", va="bottom",
                 fontsize=6.5, color=C_STITCH, alpha=0.7)

ax_main.set_xlabel("TESS Sector", fontsize=10, color=INK_PRIMARY)
ax_main.set_ylabel("Sector median flux (norm.)", fontsize=10, color=INK_PRIMARY)
ax_main.set_title(
    f"TIC {HERO_TIC}  ·  Cam{int(hero_df['cam'].mode()[0])}  ·  {len(secs)} sectors  ·  Tmag {hero_df['tmag'].mean():.1f}",
    fontsize=10, color=INK_PRIMARY, loc="left"
)
ax_main.legend(fontsize=9.5, framealpha=0.95, edgecolor=GRIDLINE, loc="upper right")

fig.suptitle(
    f"STITCH Hero Star — TIC {HERO_TIC}  (held-out, never seen during training)\n"
    f"Between-sector scatter: {scatter_before*100:.2f}% → {scatter_stitch*100:.2f}%  "
    f"({improv:.0f}% reduction)",
    fontsize=12, color=INK_PRIMARY, fontweight="bold", y=0.975
)

out = f"stitch_hero_tic{HERO_TIC}.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"\nSaved → {out}")
