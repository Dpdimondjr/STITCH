"""
Held-out evaluation: reconstruct the exact 80/10/10 TIC split used in training
(same random_state=42, same stratify-by-cam logic), then evaluate ONLY on the
10% test TICs that the model never saw during training.
"""

import numpy as np
import pandas as pd
import torch, zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
C_STITCH    = "#2166ac"
C_ENDPOINT  = "#d6604d"
C_RAW       = "#9e9e9e"

# ── 1. Load model ──────────────────────────────────────────────────────────────
print("Loading model...")
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
print(f"  Context dim: {cfg['context']}  (continuous: {len(CONTINUOUS)} + 4 cam OHE + 4 ccd OHE)")
print(f"  Continuous features: {CONTINUOUS}")

# ── 2. Load & clean data ───────────────────────────────────────────────────────
print("\nLoading training data...")
df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

print(f"  Records: {len(df):,}  |  TICs: {df['tic_id'].nunique():,}")

# ── 3. Reproduce exact train/val/test split ────────────────────────────────────
print("\nReproducing train/val/test split (random_state=42)...")
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))

train_tics, temp_tics = train_test_split(
    star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=42,
)
temp_cam = star_cam[star_cam["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics, test_size=0.5,
    stratify=temp_cam.values, random_state=42,
)

train_tics = set(train_tics)
val_tics   = set(val_tics)
test_tics  = set(test_tics)

print(f"  Train: {len(train_tics):,} stars")
print(f"  Val:   {len(val_tics):,} stars")
print(f"  Test:  {len(test_tics):,} stars  ← evaluating these only")

test_df = df[df["tic_id"].isin(test_tics)].copy()
print(f"  Test records: {len(test_df):,}")

# ── 4. Filter to ≥5 sectors ────────────────────────────────────────────────────
MIN_SECTORS = 5
sector_counts = test_df.groupby("tic_id")["sector"].count()
good_tics = sector_counts[sector_counts >= MIN_SECTORS].index
test_df = test_df[test_df["tic_id"].isin(good_tics)].copy()
print(f"  Eval stars (≥{MIN_SECTORS} sectors): {test_df['tic_id'].nunique():,}")

# ── 5. Build context and predict ───────────────────────────────────────────────
print("\nRunning inference...")
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
    samples    = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
    pred_z     = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_offset_raw = pred_z * y_std + y_mean
pred_offset_std = pred_z_std * y_std
K = 5.0
weight      = 1.0 / (1.0 + K * pred_offset_std)
pred_offset = 1.0 * (1 - weight) + pred_offset_raw * weight
test_df = test_df.copy()
test_df["pred_offset"] = pred_offset

# ── 6. Per-star scatter ────────────────────────────────────────────────────────

def endpoint_stitch(raw):
    corrected = np.ones_like(raw, dtype=float)
    scale = 1.0 / raw[0]
    for i in range(len(raw)):
        corrected[i] = raw[i] * scale
    return corrected / corrected.mean()

results = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    raw  = g["sector_median"].values
    pred = g["pred_offset"].values

    global_ref    = raw.mean()
    raw_norm      = raw / global_ref
    stitch_norm   = (raw / pred) / global_ref
    endpoint_norm = endpoint_stitch(raw)

    scatter_before   = raw_norm.std()
    scatter_stitch   = stitch_norm.std()
    scatter_endpoint = endpoint_norm.std()

    rms_before   = np.sqrt(np.mean((raw_norm - 1.0) ** 2))
    rms_stitch   = np.sqrt(np.mean((stitch_norm - 1.0) ** 2))
    rms_endpoint = np.sqrt(np.mean((endpoint_norm - 1.0) ** 2))

    results.append({
        "tic_id":                  tic,
        "n_sectors":               len(g),
        "cam":                     int(g["cam"].mode()[0]),
        "tmag":                    float(g["tmag"].mean()) if "tmag" in g.columns else np.nan,
        "scatter_before":          scatter_before,
        "scatter_stitch":          scatter_stitch,
        "scatter_endpoint":        scatter_endpoint,
        "rms_before":              rms_before,
        "rms_stitch":              rms_stitch,
        "rms_endpoint":            rms_endpoint,
        "improvement_stitch":      (scatter_before - scatter_stitch)   / scatter_before,
        "improvement_endpoint":    (scatter_before - scatter_endpoint) / scatter_before,
        "rms_improvement_stitch":  (rms_before - rms_stitch) / rms_before,
        "sectors":                 g["sector"].values,
        "raw_norm":                raw_norm,
        "stitch_norm":             stitch_norm,
        "endpoint_norm":           endpoint_norm,
    })

res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("sectors", "raw_norm", "stitch_norm", "endpoint_norm")}
                        for r in results])

# ── 7. Print results ───────────────────────────────────────────────────────────
n = len(res_df)
print(f"\n{'='*65}")
print(f"  HELD-OUT TEST SET (N={n:,} stars, ≥{MIN_SECTORS} sectors)")
print(f"  Model NEVER saw these {len(test_tics):,} TICs during training")
print(f"{'='*65}")
print(f"\n  Scatter (std of normalised sector medians):")
print(f"    Raw (no correction):  {res_df['scatter_before'].median()*100:.2f}%  median")
print(f"    Endpoint stitching:   {res_df['scatter_endpoint'].median()*100:.2f}%  "
      f"({res_df['improvement_endpoint'].median()*100:+.1f}%,  "
      f"{(res_df['improvement_endpoint']>0).mean()*100:.0f}% of stars improve)")
print(f"    STITCH NSF:           {res_df['scatter_stitch'].median()*100:.2f}%  "
      f"({res_df['improvement_stitch'].median()*100:+.1f}%,  "
      f"{(res_df['improvement_stitch']>0).mean()*100:.0f}% of stars improve)")

print(f"\n  RMS from global (absolute accuracy):")
print(f"    Raw:                  {res_df['rms_before'].median()*100:.2f}%")
print(f"    STITCH NSF:           {res_df['rms_stitch'].median()*100:.2f}%  "
      f"({res_df['rms_improvement_stitch'].median()*100:+.1f}%)")

print(f"\n  Per-camera breakdown:")
for cam, g in res_df.groupby("cam"):
    print(f"    Cam{int(cam)}: n={len(g):4d}  "
          f"scatter_raw={g['scatter_before'].median()*100:.2f}%  "
          f"→ stitch={g['scatter_stitch'].median()*100:.2f}%  "
          f"({g['improvement_stitch'].median()*100:+.1f}%)  "
          f"rms_improv={g['rms_improvement_stitch'].median()*100:+.1f}%")

# Find hero star candidates (for the hero plot script)
hero_pool = [r for r in results if r["n_sectors"] >= 8]
hero_pool.sort(key=lambda r: -r["scatter_before"])
print(f"\n  Top hero star candidates (highest raw scatter, ≥8 sectors):")
for r in hero_pool[:5]:
    print(f"    TIC {r['tic_id']}  Cam{r['cam']}  "
          f"n_sec={r['n_sectors']}  "
          f"tmag={r['tmag']:.1f}  "
          f"raw_scatter={r['scatter_before']*100:.2f}%  "
          f"→ {r['scatter_stitch']*100:.2f}%  "
          f"({r['improvement_stitch']*100:+.1f}%)")
# Save hero info for next script
import json
hero = hero_pool[0]
with open("/tmp/hero_star.json", "w") as f:
    json.dump({
        "tic_id":   int(hero["tic_id"]),
        "cam":      int(hero["cam"]),
        "n_sectors": int(hero["n_sectors"]),
        "tmag":     float(hero["tmag"]),
        "sectors":  [int(s) for s in hero["sectors"]],
        "raw_norm":     [float(v) for v in hero["raw_norm"]],
        "stitch_norm":  [float(v) for v in hero["stitch_norm"]],
        "endpoint_norm":[float(v) for v in hero["endpoint_norm"]],
        "scatter_before": float(hero["scatter_before"]),
        "scatter_stitch": float(hero["scatter_stitch"]),
        "improvement":  float(hero["improvement_stitch"]),
        "rms_before":   float(hero["rms_before"]),
        "rms_stitch":   float(hero["rms_stitch"]),
    }, f)
print(f"\n  Hero star saved → /tmp/hero_star.json")

# Also save full results for artifact
res_df.to_parquet("/tmp/held_out_results.parquet")
print(f"  Full results saved → /tmp/held_out_results.parquet")

# ── 8. Plot ────────────────────────────────────────────────────────────────────
print("\nGenerating plot...")

fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
fig.patch.set_facecolor(SURFACE)
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38,
                        left=0.08, right=0.97, top=0.88, bottom=0.07)

def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, linewidth=0.4, zorder=0)

# ── Panel A: scatter improvement histogram ────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
style_ax(ax)
bins = np.linspace(-60, 80, 60)
ax.hist(res_df["improvement_endpoint"]*100, bins=bins,
        color=C_ENDPOINT, alpha=0.65, edgecolor=SURFACE, lw=0.3,
        label=f"Endpoint (median {res_df['improvement_endpoint'].median()*100:+.1f}%)", zorder=2)
ax.hist(res_df["improvement_stitch"]*100, bins=bins,
        color=C_STITCH, alpha=0.65, edgecolor=SURFACE, lw=0.3,
        label=f"STITCH NSF (median {res_df['improvement_stitch'].median()*100:+.1f}%)", zorder=3)
ax.axvline(0, color=INK_PRIMARY, lw=1.0, ls="--", alpha=0.4)
ax.axvline(res_df["improvement_stitch"].median()*100,   color=C_STITCH,   lw=2, zorder=4)
ax.axvline(res_df["improvement_endpoint"].median()*100, color=C_ENDPOINT, lw=2, zorder=4)
ax.set_xlabel("Scatter improvement (%)", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Number of stars", fontsize=10, color=INK_PRIMARY)
ax.set_title("Scatter reduction distribution", fontsize=10, color=INK_PRIMARY, loc="left")
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=GRIDLINE)

# ── Panel B: star-by-star scatter plot ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
style_ax(ax)
ax.scatter(res_df["improvement_endpoint"]*100, res_df["improvement_stitch"]*100,
           alpha=0.12, s=5, color=INK_MUTED, rasterized=True)
lim = 70
ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, lw=1, ls="--", label="Equal performance")
ax.axhline(0, color=C_STITCH,   lw=0.8, ls=":")
ax.axvline(0, color=C_ENDPOINT, lw=0.8, ls=":")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Endpoint improvement (%)", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("STITCH improvement (%)", fontsize=10, color=INK_PRIMARY)
ax.set_title("Star-by-star comparison", fontsize=10, color=INK_PRIMARY, loc="left")
frac = (res_df["improvement_stitch"] > res_df["improvement_endpoint"]).mean()
ax.text(0.04, 0.94, f"STITCH outperforms\nendpoint in {frac*100:.0f}% of stars",
        transform=ax.transAxes, fontsize=8.5, color=C_STITCH)

# ── Panel C: per-camera RMS improvement ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
style_ax(ax)
cam_labels = [f"Cam {int(c)}" for c in sorted(res_df["cam"].unique())]
cam_vals   = [res_df[res_df["cam"]==c]["improvement_stitch"].median()*100
              for c in sorted(res_df["cam"].unique())]
cam_ns     = [len(res_df[res_df["cam"]==c]) for c in sorted(res_df["cam"].unique())]
bars = ax.bar(cam_labels, cam_vals, color=C_STITCH, alpha=0.8,
              edgecolor=SURFACE, linewidth=0.5, zorder=2)
for bar, n in zip(bars, cam_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"n={n:,}", ha="center", va="bottom", fontsize=8, color=INK_MUTED)
ax.set_ylabel("Median scatter improvement (%)", fontsize=10, color=INK_PRIMARY)
ax.set_title("By TESS camera", fontsize=10, color=INK_PRIMARY, loc="left")
ax.set_ylim(0, max(cam_vals)*1.25)

# ── Panel D: scatter improvement vs n_sectors ─────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
style_ax(ax)
sec_bins = [3, 5, 8, 12, 20, 100]
sec_centers, sec_meds, sec_lo, sec_hi = [], [], [], []
for lo, hi in zip(sec_bins[:-1], sec_bins[1:]):
    sub = res_df[(res_df["n_sectors"]>=lo) & (res_df["n_sectors"]<hi)]["improvement_stitch"]*100
    if len(sub) > 5:
        sec_centers.append(f"{lo}–{hi-1}")
        sec_meds.append(sub.median())
        sec_lo.append(sub.quantile(0.25))
        sec_hi.append(sub.quantile(0.75))

x = np.arange(len(sec_centers))
ax.bar(x, sec_meds, color=C_STITCH, alpha=0.8, edgecolor=SURFACE, linewidth=0.5, zorder=2)
ax.errorbar(x, sec_meds,
            yerr=[np.array(sec_meds)-np.array(sec_lo),
                  np.array(sec_hi)-np.array(sec_meds)],
            fmt="none", color=INK_MUTED, capsize=4, lw=1.5, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(sec_centers, fontsize=9)
ax.set_xlabel("Number of sectors per star", fontsize=10, color=INK_PRIMARY)
ax.set_ylabel("Median scatter improvement (%)", fontsize=10, color=INK_PRIMARY)
ax.set_title("Improvement vs sector count", fontsize=10, color=INK_PRIMARY, loc="left")

# ── Headline ──────────────────────────────────────────────────────────────────
stitch_med = res_df["improvement_stitch"].median()*100
stitch_pct = (res_df["improvement_stitch"]>0).mean()*100
endpoint_med = res_df["improvement_endpoint"].median()*100
endpoint_pct = (res_df["improvement_endpoint"]>0).mean()*100

fig.suptitle(
    f"STITCH NSF — Held-out test set ({n:,} stars, never seen during training)\n"
    f"Scatter reduction:  STITCH {stitch_med:.1f}% median ({stitch_pct:.0f}% of stars improve)  ·  "
    f"Endpoint {endpoint_med:.1f}% ({endpoint_pct:.0f}% improve)",
    fontsize=11, color=INK_PRIMARY, fontweight="bold", y=0.96
)

out = "stitch_eval_held_out.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
plt.close()
print(f"Saved → {out}")
