"""
Generate per-TIC images showing what endpoint stitching looks like vs STITCH.
Endpoint stitching = naive normalization by dividing all sectors by the global mean
(equivalent to raw_norm). It leaves between-sector DC offsets completely uncorrected.
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
import os

OUT_DIR = "endpoint_baseline_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color scheme ─────────────────────────────────────────────────────────────
SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
C_RAW       = "#898781"
C_ENDPOINT  = "#d03b3b"
C_STITCH    = "#2a78d6"

# ── 1. Load model ─────────────────────────────────────────────────────────────
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
print(f"  Context dim: {cfg['context']}  continuous: {CONTINUOUS}")

# ── 2. Load data + test split ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))
train_tics, temp_tics = train_test_split(
    star_cam["tic_id"], test_size=0.2, stratify=star_cam["dominant_cam"], random_state=42)
temp_cam = star_cam[star_cam["tic_id"].isin(temp_tics)]["dominant_cam"]
val_tics, test_tics = train_test_split(
    temp_tics, test_size=0.5, stratify=temp_cam.values, random_state=42)
test_tics = set(test_tics)

# ── 3. Run inference on test set ──────────────────────────────────────────────
print("Running model inference on test set...")
test_df = df[df["tic_id"].isin(test_tics)].copy()

def make_context(df):
    cont = (df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(df["cam"].astype(int), prefix="cam").reindex(columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(df["ccd"].astype(int), prefix="ccd").reindex(columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True), cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

C = make_context(test_df)
with torch.no_grad():
    samples   = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
    pred_z    = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_offset_raw = pred_z * y_std + y_mean
pred_offset_std = pred_z_std * y_std
weight = 1.0 / (1.0 + 5.0 * pred_offset_std)
pred_offset = 1.0 * (1 - weight) + pred_offset_raw * weight

test_df = test_df.copy()
test_df["pred_offset"] = pred_offset

# ── 4. Endpoint stitching (same as raw_norm) ──────────────────────────────────
def endpoint_stitch(raw):
    corrected = np.ones_like(raw, dtype=float)
    scale = 1.0 / raw[0]
    for i in range(len(raw)):
        corrected[i] = raw[i] * scale
    return corrected / corrected.mean()

# ── 5. Build per-star results ─────────────────────────────────────────────────
results = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    if len(g) < 5:
        continue
    raw  = g["sector_median"].values
    pred = g["pred_offset"].values
    global_ref  = raw.mean()
    raw_norm    = raw / global_ref
    stitch_norm = (raw / pred) / global_ref
    ep_norm     = endpoint_stitch(raw)
    imp_stitch  = (raw_norm.std() - stitch_norm.std()) / raw_norm.std()
    results.append({
        "tic_id": tic, "n_sectors": len(g), "cam": int(g["cam"].mode()[0]),
        "tmag": float(g["tmag"].mean()) if "tmag" in g.columns else np.nan,
        "scatter_before": raw_norm.std() * 100,
        "scatter_stitch": stitch_norm.std() * 100,
        "improvement": imp_stitch * 100,
        "sectors": g["sector"].values,
        "raw_norm": raw_norm,
        "stitch_norm": stitch_norm,
        "ep_norm": ep_norm,
    })

# ── 6. Select example TICs ───────────────────────────────────────────────────
# Pick stars with high raw scatter, many sectors, good STITCH improvement
# Variety of cameras
res_df = pd.DataFrame([{k: v for k, v in r.items() if not hasattr(v, '__len__')} for r in results])
res_full = {r['tic_id']: r for r in results}

picks = []
for cam in [1, 2, 3, 4]:
    sub = res_df[(res_df['cam'] == cam) & (res_df['scatter_before'] > 2.5) &
                 (res_df['improvement'] > 50) & (res_df['n_sectors'] >= 7)]
    if len(sub) == 0:
        sub = res_df[(res_df['cam'] == cam) & (res_df['scatter_before'] > 1.5) & (res_df['improvement'] > 40)]
    if len(sub) > 0:
        picks.append(sub.nlargest(1, 'scatter_before').iloc[0])

print(f"\nSelected {len(picks)} TICs:")
for p in picks:
    print(f"  TIC {int(p.tic_id)}: cam{int(p.cam)} scatter {p.scatter_before:.2f}% -> {p.scatter_stitch:.2f}% ({p.improvement:.0f}% improvement)")

# ── 7. Generate images ────────────────────────────────────────────────────────
def plot_tic(r):
    tic = r['tic_id']
    sectors = r['sectors']
    raw_n   = r['raw_norm']
    ep_n    = r['ep_norm']
    st_n    = r['stitch_norm']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=SURFACE)
    fig.suptitle(
        f"TIC {tic}  ·  Cam {r['cam']}  ·  Tmag {r['tmag']:.1f}  ·  {r['n_sectors']} sectors  ·  "
        f"Between-sector scatter: {r['scatter_before']:.2f}%",
        fontsize=12, color=INK_PRIMARY, x=0.5, y=1.01
    )

    # Shared y range
    all_vals = np.concatenate([raw_n, ep_n, st_n])
    ymin, ymax = all_vals.min() - 0.005, all_vals.max() + 0.005
    pad = (ymax - ymin) * 0.12
    ymin -= pad; ymax += pad

    panels = [
        (raw_n, C_RAW,      "Raw (uncorrected)",
         f"scatter = {r['scatter_before']:.3f}%"),
        (ep_n,  C_ENDPOINT, "Endpoint stitching",
         f"scatter = {raw_n.std()*100:.3f}%  (no change)"),
        (st_n,  C_STITCH,   "STITCH corrected",
         f"scatter = {r['scatter_stitch']:.3f}%  ({r['improvement']:.0f}% reduction)"),
    ]

    for ax, (vals, col, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.set_ylim(ymin, ymax)

        # Grid
        for spine in ax.spines.values():
            spine.set_edgecolor(GRIDLINE)
        ax.axhline(1.0, color=GRIDLINE, lw=1.2, ls='--', zorder=0)
        ax.yaxis.set_tick_params(labelcolor=INK_MUTED, labelsize=9)
        ax.xaxis.set_tick_params(labelcolor=INK_MUTED, labelsize=9)
        ax.tick_params(colors=INK_MUTED)

        # Line + markers
        ax.plot(sectors, vals, color=col, lw=1.8, alpha=0.7, zorder=2)
        ax.scatter(sectors, vals, color=col, s=60, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)

        # Sector labels on x axis
        ax.set_xticks(sectors)
        ax.set_xticklabels([f'S{s}' for s in sectors], fontsize=8, rotation=45, ha='right')

        ax.set_title(title, color=INK_PRIMARY, fontsize=11, fontweight='600', pad=6)
        ax.set_ylabel("Normalised sector median", color=INK_MUTED, fontsize=9)
        ax.text(0.98, 0.04, sub, transform=ax.transAxes,
                fontsize=9, color=col, ha='right', va='bottom', fontweight='600')
        ax.grid(axis='y', color=GRIDLINE, lw=0.6, zorder=0)

    # Vertical line between panel 2 and 3 to emphasize the comparison
    fig.text(0.66, 0.5, "→", fontsize=28, ha='center', va='center', color=INK_MUTED, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT_DIR, f"endpoint_vs_stitch_TIC{tic}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"  Saved: {out}")

for p in picks:
    plot_tic(res_full[int(p.tic_id)])

# ── 8. Summary grid: 4 stars in one figure ───────────────────────────────────
if len(picks) >= 3:
    fig, axes = plt.subplots(len(picks), 3, figsize=(14, len(picks)*3.2), facecolor=SURFACE)
    if len(picks) == 1:
        axes = [axes]
    fig.suptitle("Endpoint stitching vs STITCH — held-out test stars",
                 fontsize=13, color=INK_PRIMARY, fontweight='700', y=1.01)

    for row_i, p in enumerate(picks):
        r = res_full[int(p.tic_id)]
        sectors = r['sectors']
        all_vals = np.concatenate([r['raw_norm'], r['ep_norm'], r['stitch_norm']])
        ymin, ymax = all_vals.min() - 0.005, all_vals.max() + 0.005
        pad = (ymax - ymin) * 0.15
        ymin -= pad; ymax += pad

        panels = [
            (r['raw_norm'], C_RAW, "Raw",
             f"σ = {r['scatter_before']:.2f}%"),
            (r['ep_norm'], C_ENDPOINT, "Endpoint stitching",
             f"σ = {r['raw_norm'].std()*100:.2f}%  (unchanged)"),
            (r['stitch_norm'], C_STITCH, "STITCH",
             f"σ = {r['scatter_stitch']:.2f}%  ({r['improvement']:.0f}% better)"),
        ]
        row_label = f"TIC {r['tic_id']} · Cam{r['cam']} · Tmag {r['tmag']:.1f}"

        for ci, (ax, (vals, col, title, sub)) in enumerate(zip(axes[row_i], panels)):
            ax.set_facecolor(SURFACE)
            ax.set_ylim(ymin, ymax)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRIDLINE)
            ax.axhline(1.0, color=GRIDLINE, lw=1, ls='--', zorder=0)
            ax.yaxis.set_tick_params(labelcolor=INK_MUTED, labelsize=8)
            ax.xaxis.set_tick_params(labelcolor=INK_MUTED, labelsize=8)
            ax.plot(sectors, vals, color=col, lw=1.5, alpha=0.7, zorder=2)
            ax.scatter(sectors, vals, color=col, s=45, zorder=3, edgecolors=SURFACE, linewidths=1.2)
            ax.set_xticks(sectors[::max(1, len(sectors)//6)])
            ax.set_xticklabels([f'S{s}' for s in sectors[::max(1, len(sectors)//6)]], fontsize=7)
            ax.grid(axis='y', color=GRIDLINE, lw=0.5, zorder=0)
            if row_i == 0:
                ax.set_title(title, color=INK_PRIMARY, fontsize=10, fontweight='600')
            if ci == 0:
                ax.set_ylabel(row_label, color=INK_MUTED, fontsize=8, labelpad=4)
            ax.text(0.98, 0.05, sub, transform=ax.transAxes,
                    fontsize=8, color=col, ha='right', va='bottom', fontweight='600')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(OUT_DIR, "endpoint_vs_stitch_summary.png")
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    plt.close()
    print(f"\nSummary grid saved: {out}")

print("\nDone. Images in:", OUT_DIR)
