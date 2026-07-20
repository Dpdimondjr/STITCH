"""
Run STITCH inference on a single star and plot before/after stitching.

Usage: python3 infer_star.py <TIC_ID>
"""

import sys
import warnings
import numpy as np
import pandas as pd
import torch
import zuko
import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")

TIC_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 264221449
CACHE_DIR = "./tess_cache"
MAX_SECTORS = 30  # no cap for inference — get everything

# ── 1. Load model ─────────────────────────────────────────────────────────────

ckpt = torch.load("stitch_nsf.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features        = cfg["features"],
    context         = cfg["context"],
    transforms      = cfg["transforms"],
    hidden_features = cfg["hidden_features"],
    bins            = cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()

means      = ckpt["means"]
stds       = ckpt["stds"]
y_mean     = ckpt["y_mean"]
y_std      = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]
CCD_COLS   = ckpt["ccd_cols"]

# ── 2. Download TESS-SPOC light curves ────────────────────────────────────────

print(f"Searching TESS-SPOC for TIC {TIC_ID}...")
sr = lk.search_lightcurve(f"TIC {TIC_ID}", mission="TESS", author="TESS-SPOC")
print(f"  Found {len(sr)} results")
if len(sr) == 0:
    sys.exit("No TESS-SPOC data found.")

try:
    sr = sr[sr.exptime.value >= 100]  # keep 10-min FFI cadence
except Exception:
    pass
print(f"  After cadence filter: {len(sr)} sectors")

print("  Downloading...")
lc_list = []
for i in range(len(sr)):
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTO
        with ThreadPoolExecutor(max_workers=1) as ex:
            lc = ex.submit(sr[i].download, download_dir=CACHE_DIR).result(timeout=120)
        if lc is not None:
            lc_list.append(lc)
            print(f"    sector {lc.meta.get('SECTOR')} ✓")
    except Exception as e:
        print(f"    sector {i} failed: {e}")

print(f"  Downloaded {len(lc_list)} sectors")
if not lc_list:
    sys.exit("No sectors downloaded.")

# ── 3. Pixel positions via tess_stars2px ──────────────────────────────────────

ra  = float(lc_list[0].meta["RA_OBJ"])
dec = float(lc_list[0].meta["DEC_OBJ"])
print(f"  RA={ra:.4f}  Dec={dec:.4f}")

_, _, _, out_sec, out_cam, out_ccd, out_col, out_row, _ = \
    tess_stars2px_function_entry(TIC_ID, ra, dec)
pos_lookup = {int(s): (int(c1), int(c2), float(cl), float(rw))
              for s, c1, c2, cl, rw in zip(out_sec, out_cam, out_ccd, out_col, out_row)}

# ── 4. Build per-sector features and predict ──────────────────────────────────

def _flt(v):
    try: return float(v)
    except: return None

sectors_meta = []
lcs_valid = []

for lc in lc_list:
    sec = lc.meta.get("SECTOR")
    if sec is None or int(sec) not in pos_lookup:
        continue
    cam_tp, ccd_tp, col, row = pos_lookup[int(sec)]

    def _med_pc(col_name):
        try:
            v = lc[col_name].value.astype(float)
            return float(np.nanmedian(v)), float(np.sqrt(np.nanvar(v)))
        except:
            return np.nan, np.nan

    pc1_med, pc1_rms = _med_pc("pos_corr1")
    pc2_med, pc2_rms = _med_pc("pos_corr2")
    jitter_rms = float(np.sqrt(pc1_rms**2 + pc2_rms**2)) if np.isfinite(pc1_rms) else np.nan

    try:
        sec_med = float(np.nanmedian(lc.flux.value))
    except Exception:
        sec_med = np.nan

    row_d = {
        "sector":             float(sec),
        "cam":                cam_tp,
        "ccd":                ccd_tp,
        "col":                col,
        "row":                row,
        "delta_sub_col":      (col + pc1_med) % 1.0 if np.isfinite(col) and np.isfinite(pc1_med) else np.nan,
        "delta_sub_row":      (row + pc2_med) % 1.0 if np.isfinite(row) and np.isfinite(pc2_med) else np.nan,
        "tmag":               _flt(lc.meta.get("TESSMAG")),
        "crowdsap":           _flt(lc.meta.get("CROWDSAP")),
        "cdpp1_0":            _flt(lc.meta.get("CDPP1_0")),
        "pdcvar":             _flt(lc.meta.get("PDCVAR")),
        "jitter_rms":         jitter_rms,
        "log_sector_median":  np.log1p(max(sec_med, 0)) if np.isfinite(sec_med) else np.nan,
    }
    sectors_meta.append(row_d)
    lcs_valid.append(lc)

meta_df = pd.DataFrame(sectors_meta)
# n_sectors_total known only after loop
meta_df["n_sectors_total"] = float(len(sectors_meta))

# Impute NaNs with training medians
for col in CONTINUOUS:
    if col in meta_df.columns and meta_df[col].isna().any():
        meta_df[col] = meta_df[col].fillna(float(means[col]))

# Build context tensor
cont   = (meta_df[CONTINUOUS] - means) / stds
cam_oh = pd.get_dummies(meta_df["cam"].astype(int), prefix="cam").reindex(
             columns=CAM_COLS, fill_value=0)
ccd_oh = pd.get_dummies(meta_df["ccd"].astype(int), prefix="ccd").reindex(
             columns=CCD_COLS, fill_value=0)
C = pd.concat([cont.reset_index(drop=True),
               cam_oh.reset_index(drop=True),
               ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

with torch.no_grad():
    samples = flow(torch.tensor(C)).sample((500,)).squeeze(-1)  # (500, N)
    pred_z     = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_offset_raw = pred_z * y_std + y_mean
pred_offset_std = pred_z_std * y_std          # posterior std in flux_offset units

# Shrink uncertain predictions toward 1.0 (no correction).
# Weight = 1 when std→0 (confident), → 0 when std is large (uncertain).
K = 5.0
weight      = 1.0 / (1.0 + K * pred_offset_std)
pred_offset = 1.0 * (1 - weight) + pred_offset_raw * weight

print(f"\nPredicted offsets per sector (K={K:.0f} shrinkage):")
for row_d, raw, std, w, off in zip(sectors_meta, pred_offset_raw, pred_offset_std, weight, pred_offset):
    print(f"  Sector {int(row_d['sector']):3d}  Cam{row_d['cam']}/CCD{row_d['ccd']}  "
          f"raw={raw:.4f}  std={std:.4f}  w={w:.2f}  → {off:.4f}")

# ── 5. Plot full multi-sector light curve ─────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
colors_cam = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}

# Compute global reference = mean of all sector medians (raw).
# This is the shared denominator for both raw and corrected — never renormalise
# the corrected output by its own mean, or we hide any systematic shift.
sector_meds = []
sector_data = []  # (time, flux, med, offset, cam)
for lc, meta, offset in zip(lcs_valid, sectors_meta, pred_offset):
    flux = lc.flux.value.astype(float)
    time = lc.time.value.astype(float)
    mask = np.isfinite(flux) & np.isfinite(time)
    flux, time = flux[mask], time[mask]
    if len(flux) == 0:
        continue
    med = np.nanmedian(flux)
    sector_meds.append(med)
    sector_data.append((time, flux, med, offset, meta["cam"]))

global_ref = np.mean(sector_meds)

for time, flux, med, offset, cam in sector_data:
    col = colors_cam.get(cam, "#888888")
    raw_norm = flux / global_ref
    cor_norm = (flux / offset) / global_ref
    axes[0].plot(time, raw_norm, ".", ms=0.8, color=col, alpha=0.35, rasterized=True)
    axes[1].plot(time, cor_norm, ".", ms=0.8, color=col, alpha=0.35, rasterized=True)

# Sector median markers — triangles at sector midpoint
raw_sec_meds, cor_sec_meds = [], []
for time, flux, med, offset, cam in sector_data:
    t_mid = np.median(time)
    col = colors_cam.get(cam, "#888888")
    raw_level = med / global_ref
    cor_level = (med / offset) / global_ref   # absolute, not renormalised
    axes[0].plot(t_mid, raw_level, "^", color=col, ms=6, zorder=5)
    axes[1].plot(t_mid, cor_level, "^", color=col, ms=6, zorder=5)
    raw_sec_meds.append(raw_level)
    cor_sec_meds.append(cor_level)

for ax in axes:
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

raw_sec_meds = np.array(raw_sec_meds)
cor_sec_meds = np.array(cor_sec_meds)

# Scatter (relative): std around each method's own mean
raw_scatter = raw_sec_meds.std()
cor_scatter = cor_sec_meds.std()
scatter_improv = (raw_scatter - cor_scatter) / raw_scatter * 100 if raw_scatter > 0 else 0

# RMS from global (absolute): how close do corrected sectors sit to 1.0?
raw_rms = np.sqrt(np.mean((raw_sec_meds - 1.0) ** 2))
cor_rms = np.sqrt(np.mean((cor_sec_meds - 1.0) ** 2))
rms_improv = (raw_rms - cor_rms) / raw_rms * 100 if raw_rms > 0 else 0

# Formatting
for ax, title in zip(axes, ["Before STITCH", "After STITCH"]):
    ax.set_ylabel("Normalised flux", fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylim(0.90, 1.10)

axes[1].set_xlabel("BTJD (days)", fontsize=10)

# Camera legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=c, lw=3, label=f"Cam {k}") for k, c in colors_cam.items()]
axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=4)

fig.suptitle(
    f"TIC {TIC_ID}  ·  {len(lcs_valid)} sectors\n"
    f"Scatter (relative):  {raw_scatter:.4f} → {cor_scatter:.4f}  ({scatter_improv:+.1f}%)\n"
    f"RMS from global:     {raw_rms:.4f} → {cor_rms:.4f}  ({rms_improv:+.1f}%)",
    fontsize=11
)

out = f"stitch_tic{TIC_ID}.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
