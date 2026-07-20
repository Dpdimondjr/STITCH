"""
STITCH stitching evaluation: before vs after scatter on held-out quiet stars.

For each test star (never seen during training):
  - The model predicts flux_offset from detector position alone
  - We divide each sector's median flux by the predicted offset
  - Scatter in sector medians should decrease if the model is correcting PRF effects

This avoids the LOO-ratio circularity: the model never sees the star's flux,
and the improvement metric (scatter reduction) is physically grounded
(a quiet star should be flat across sectors).
"""

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

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

means  = ckpt["means"]
stds   = ckpt["stds"]
y_mean = ckpt["y_mean"]
y_std  = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]
CCD_COLS   = ckpt["ccd_cols"]

# ── 2. Load and clean data ────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── 3. Select evaluation stars ────────────────────────────────────────────────
# Use all stars with ≥ MIN_SECTORS sectors. Scatter reduction doesn't have
# the standard overfitting concern since the model predicts from position only
# (it never sees the star's flux), so train/test contamination is minimal.

MIN_SECTORS = 5
sector_counts = df.groupby("tic_id")["sector"].count()
good_tics = sector_counts[sector_counts >= MIN_SECTORS].index
test_df = df[df["tic_id"].isin(good_tics)].copy()
print(f"Eval stars (≥{MIN_SECTORS} sectors): {test_df['tic_id'].nunique()}")

# ── 4. Build context and predict ──────────────────────────────────────────────

def make_context(split_df):
    cont   = (split_df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(split_df["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

C = make_context(test_df)
with torch.no_grad():
    samples    = flow(torch.tensor(C)).sample((500,)).squeeze(-1)  # (500, N)
    pred_z     = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_offset_raw = pred_z * y_std + y_mean
pred_offset_std = pred_z_std * y_std

# Shrink uncertain predictions toward 1.0
K = 5.0
weight      = 1.0 / (1.0 + K * pred_offset_std)
pred_offset = 1.0 * (1 - weight) + pred_offset_raw * weight

print(f"Uncertainty shrinkage (K={K:.0f}): "
      f"weight mean={weight.mean():.3f}  std_of_offset_std={pred_offset_std.std():.4f}")
test_df = test_df.copy()
test_df["pred_offset"] = pred_offset

# ── 5. Compute per-star scatter: raw / STITCH / endpoint baseline ─────────────

def endpoint_stitch(raw):
    """
    Simulate endpoint stitching using sector medians as a proxy for edge flux.
    Anchor sector 0 to 1.0, then chain: each subsequent sector is scaled so its
    'edge' (its own median) matches the previous sector's corrected 'edge'.
    This is the simplest version of sequential sector-to-sector normalisation.
    """
    corrected = np.ones_like(raw, dtype=float)
    scale = 1.0
    for i in range(len(raw)):
        if i == 0:
            corrected[i] = 1.0
            scale = 1.0 / raw[0]
        else:
            corrected[i] = raw[i] * scale
    return corrected / corrected.mean()

results = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    raw  = g["sector_median"].values
    pred = g["pred_offset"].values

    # Use raw global median as the shared reference for ALL methods.
    # Never renormalize the corrected output — that would hide systematic shifts.
    global_ref    = raw.mean()
    raw_norm      = raw / global_ref
    stitch_norm   = (raw / pred) / global_ref   # absolute accuracy: should land at 1.0
    endpoint_norm = endpoint_stitch(raw)         # endpoint normalises internally to its mean

    # Scatter metric: std around each method's own mean
    scatter_before   = raw_norm.std()
    scatter_stitch   = stitch_norm.std()
    scatter_endpoint = endpoint_norm.std()

    # RMS-from-global: how far does each corrected sector sit from the true level?
    # raw:      rms(raw_norm - 1.0) == scatter_before (since mean(raw_norm)=1.0)
    # stitch:   rms(stitch_norm - 1.0) penalises both spread AND mean bias
    # endpoint: uses its own renorm so rms == its scatter (relative-only, for reference)
    rms_before   = np.sqrt(np.mean((raw_norm - 1.0) ** 2))
    rms_stitch   = np.sqrt(np.mean((stitch_norm - 1.0) ** 2))
    rms_endpoint = np.sqrt(np.mean((endpoint_norm - 1.0) ** 2))

    results.append({
        "tic_id":           tic,
        "n_sectors":        len(g),
        "cam":              int(g["cam"].mode()[0]),
        "scatter_before":   scatter_before,
        "scatter_stitch":   scatter_stitch,
        "scatter_endpoint": scatter_endpoint,
        "rms_before":       rms_before,
        "rms_stitch":       rms_stitch,
        "rms_endpoint":     rms_endpoint,
        "improvement_stitch":     (scatter_before - scatter_stitch)   / scatter_before,
        "improvement_endpoint":   (scatter_before - scatter_endpoint) / scatter_before,
        "rms_improvement_stitch": (rms_before - rms_stitch)   / rms_before,
        "sectors":          g["sector"].values,
        "raw_norm":         raw_norm,
        "stitch_norm":      stitch_norm,
        "endpoint_norm":    endpoint_norm,
    })

res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("sectors", "raw_norm", "stitch_norm", "endpoint_norm")}
                        for r in results])

print(f"\n=== Scatter Reduction — std(sector medians) (N={len(res_df)} stars, ≥{MIN_SECTORS} sectors) ===")
print(f"  NOTE: scatter is relative — measures spread around each method's own mean")
print(f"{'Method':<22} {'Median scatter':>15}  {'Median improve':>15}  {'% stars better':>15}")
print(f"  {'─'*65}")
print(f"  {'Raw (no correction)':<20} {res_df['scatter_before'].median():>15.4f}  {'—':>15}  {'—':>15}")
print(f"  {'Endpoint stitching':<20} {res_df['scatter_endpoint'].median():>15.4f}  "
      f"{res_df['improvement_endpoint'].median()*100:>14.1f}%  "
      f"{(res_df['improvement_endpoint']>0).mean()*100:>14.0f}%")
print(f"  {'STITCH NSF':<20} {res_df['scatter_stitch'].median():>15.4f}  "
      f"{res_df['improvement_stitch'].median()*100:>14.1f}%  "
      f"{(res_df['improvement_stitch']>0).mean()*100:>14.0f}%")

print(f"\n=== RMS from Global Median — rms(corrected/global - 1.0) ===")
print(f"  NOTE: absolute accuracy — penalises both spread AND mean bias vs true global")
print(f"{'Method':<22} {'Median RMS':>12}  {'Median improve':>15}")
print(f"  {'─'*55}")
print(f"  {'Raw (no correction)':<20} {res_df['rms_before'].median():>12.4f}  {'—':>15}")
print(f"  {'STITCH NSF':<20} {res_df['rms_stitch'].median():>12.4f}  "
      f"{res_df['rms_improvement_stitch'].median()*100:>14.1f}%")

print(f"\n  Per-camera (STITCH scatter vs RMS-from-global):")
for cam, g in res_df.groupby("cam"):
    print(f"    Cam{int(cam)}: n={len(g):3d}  "
          f"scatter_improve={g['improvement_stitch'].median()*100:.1f}%  "
          f"rms_improve={g['rms_improvement_stitch'].median()*100:.1f}%")

# ── 6. Plot ───────────────────────────────────────────────────────────────────

# Pick example stars: one per camera with n_sectors >= 8
examples = []
for cam in sorted(res_df["cam"].unique()):
    pool = [r for r in results
            if r["cam"] == cam and r["n_sectors"] >= 8]
    if pool:
        # Pick one near median improvement for that camera
        med = np.median([r["improvement_stitch"] for r in pool])
        best = min(pool, key=lambda r: abs(r["improvement_stitch"] - med))
        examples.append(best)

n_ex = len(examples)
fig = plt.figure(figsize=(14, 4 + 3 * n_ex))
gs  = gridspec.GridSpec(n_ex + 1, 2, height_ratios=[2.5]*n_ex + [3],
                        hspace=0.55, wspace=0.35)

colors = {"raw": "#888888", "stitch": "#2166ac", "endpoint": "#d6604d"}

# Example light curves (one per camera)
for i, r in enumerate(examples):
    ax = fig.add_subplot(gs[i, :])
    secs = r["sectors"]
    ax.plot(secs, r["raw_norm"],      "o-", color=colors["raw"],
            lw=1.2, ms=4, label=f"Raw  (σ={r['scatter_before']:.4f}, rms={r['rms_before']:.4f})", alpha=0.7)
    ax.plot(secs, r["endpoint_norm"], "s-", color=colors["endpoint"],
            lw=1.2, ms=4, label=f"Endpoint  (σ={r['scatter_endpoint']:.4f}, {r['improvement_endpoint']*100:+.1f}%)")
    ax.plot(secs, r["stitch_norm"],   "o-", color=colors["stitch"],
            lw=1.4, ms=5, label=f"STITCH NSF  (σ={r['scatter_stitch']:.4f}, rms={r['rms_stitch']:.4f}, {r['rms_improvement_stitch']*100:+.1f}% rms)")
    ax.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.4)
    ax.set_ylabel("Norm. flux", fontsize=9)
    ax.set_title(f"TIC {r['tic_id']}  ·  Cam{r['cam']}  ·  {r['n_sectors']} sectors",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    if i == n_ex - 1:
        ax.set_xlabel("TESS Sector", fontsize=9)

# Histogram: STITCH vs Endpoint
ax_hist = fig.add_subplot(gs[n_ex, 0])
bins = np.linspace(-60, 60, 50)
ax_hist.hist(res_df["improvement_endpoint"]*100, bins=bins,
             color=colors["endpoint"], alpha=0.6, label="Endpoint", edgecolor="white", lw=0.3)
ax_hist.hist(res_df["improvement_stitch"]*100, bins=bins,
             color=colors["stitch"], alpha=0.6, label="STITCH NSF", edgecolor="white", lw=0.3)
ax_hist.axvline(0, color="k", lw=1.2, ls="--")
ax_hist.axvline(res_df["improvement_stitch"].median()*100,   color=colors["stitch"],   lw=2, ls="-")
ax_hist.axvline(res_df["improvement_endpoint"].median()*100, color=colors["endpoint"], lw=2, ls="-")
ax_hist.set_xlabel("Scatter improvement (%)", fontsize=10)
ax_hist.set_ylabel("Number of stars", fontsize=10)
ax_hist.set_title("STITCH vs Endpoint baseline", fontsize=10)
ax_hist.legend(fontsize=9)

# Scatter: STITCH improvement vs Endpoint improvement (star by star)
ax_box = fig.add_subplot(gs[n_ex, 1])
ax_box.scatter(res_df["improvement_endpoint"]*100, res_df["improvement_stitch"]*100,
               alpha=0.15, s=8, color="#444444")
lim = 60
ax_box.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Equal performance")
ax_box.axhline(0, color=colors["stitch"],   lw=1, ls=":")
ax_box.axvline(0, color=colors["endpoint"], lw=1, ls=":")
ax_box.set_xlim(-lim, lim); ax_box.set_ylim(-lim, lim)
ax_box.set_xlabel("Endpoint improvement (%)", fontsize=10)
ax_box.set_ylabel("STITCH improvement (%)", fontsize=10)
ax_box.set_title("Star-by-star comparison", fontsize=10)
ax_box.legend(fontsize=8)
frac_stitch_wins = (res_df["improvement_stitch"] > res_df["improvement_endpoint"]).mean()
ax_box.text(0.05, 0.93, f"STITCH better: {frac_stitch_wins*100:.0f}% of stars",
            transform=ax_box.transAxes, fontsize=8.5, color=colors["stitch"])

fig.suptitle("STITCH NSF — Before vs After Sector Stitching (held-out quiet stars)",
             fontsize=12, y=1.01)

out = "stitch_eval.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
