"""
Scatter reduction eval for the per-cam/CCD model ensemble.
Same methodology as evaluate_stitching.py but loads stitch_nsf_per_ccd.pt.
"""

import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. Load all 16 models ─────────────────────────────────────────────────────

ckpt_all = torch.load("stitch_nsf_per_ccd.pt", map_location="cpu", weights_only=False)
print(f"Loaded {len(ckpt_all)} cam/CCD models")

flows = {}
for key, c in ckpt_all.items():
    cfg = c["flow_config"]
    f = zuko.flows.NSF(**cfg)
    f.load_state_dict(c["model_state"])
    f.eval()
    flows[key] = f

# ── 2. Load and clean data ────────────────────────────────────────────────────

df = pd.read_parquet("training_data.parquet")
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))

MIN_SECTORS = 5
sector_counts = df.groupby("tic_id")["sector"].count()
good_tics = sector_counts[sector_counts >= MIN_SECTORS].index
test_df = df[df["tic_id"].isin(good_tics)].copy()
print(f"Eval stars (≥{MIN_SECTORS} sectors): {test_df['tic_id'].nunique()}")

# ── 3. Predict per cam/CCD group ──────────────────────────────────────────────

test_df = test_df.copy()
test_df["pred_offset"] = np.nan

K = 5.0
for key, flow in flows.items():
    c = ckpt_all[key]
    CONTINUOUS = c["continuous_cols"]
    means, stds = c["means"], c["stds"]
    y_mean, y_std = c["y_mean"], c["y_std"]
    cam, ccd = c["cam"], c["ccd"]

    mask = (test_df["cam"] == cam) & (test_df["ccd"] == ccd)
    sub = test_df[mask].copy()
    if len(sub) == 0:
        continue

    for col in CONTINUOUS:
        if sub[col].isna().any():
            sub[col] = sub[col].fillna(float(means[col]))

    cont = ((sub[CONTINUOUS] - means) / stds).values.astype(np.float32)
    C = torch.tensor(cont)

    with torch.no_grad():
        samples    = flow(C).sample((500,)).squeeze(-1)
        pred_z     = samples.mean(0).numpy()
        pred_z_std = samples.std(0).numpy()

    pred_raw = pred_z * y_std + y_mean
    pred_std = pred_z_std * y_std
    weight   = 1.0 / (1.0 + K * pred_std)
    pred     = 1.0 * (1 - weight) + pred_raw * weight

    test_df.loc[mask, "pred_offset"] = pred
    print(f"  {key}: {mask.sum()} rows predicted")

# Drop rows where no model covered (shouldn't happen)
test_df = test_df.dropna(subset=["pred_offset"])

# ── 4. Per-star scatter reduction ─────────────────────────────────────────────

def endpoint_stitch(raw):
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

    raw_norm      = raw / raw.mean()
    stitch_norm   = (raw / pred); stitch_norm /= stitch_norm.mean()
    endpoint_norm = endpoint_stitch(raw)

    scatter_before   = raw_norm.std()
    scatter_stitch   = stitch_norm.std()
    scatter_endpoint = endpoint_norm.std()

    results.append({
        "tic_id":               tic,
        "n_sectors":            len(g),
        "cam":                  int(g["cam"].mode()[0]),
        "scatter_before":       scatter_before,
        "scatter_stitch":       scatter_stitch,
        "scatter_endpoint":     scatter_endpoint,
        "improvement_stitch":   (scatter_before - scatter_stitch)   / scatter_before,
        "improvement_endpoint": (scatter_before - scatter_endpoint) / scatter_before,
        "sectors":              g["sector"].values,
        "raw_norm":             raw_norm,
        "stitch_norm":          stitch_norm,
        "endpoint_norm":        endpoint_norm,
    })

res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("sectors","raw_norm","stitch_norm","endpoint_norm")}
                        for r in results])

print(f"\n=== Scatter Reduction (N={len(res_df)} stars, ≥{MIN_SECTORS} sectors) ===")
print(f"{'Method':<22} {'Median scatter':>15}  {'Median improve':>15}  {'% stars better':>15}")
print(f"  {'─'*65}")
print(f"  {'Raw (no correction)':<20} {res_df['scatter_before'].median():>15.4f}  {'—':>15}  {'—':>15}")
print(f"  {'Endpoint stitching':<20} {res_df['scatter_endpoint'].median():>15.4f}  "
      f"{res_df['improvement_endpoint'].median()*100:>14.1f}%  "
      f"{(res_df['improvement_endpoint']>0).mean()*100:>14.0f}%")
print(f"  {'STITCH per-CCD':<20} {res_df['scatter_stitch'].median():>15.4f}  "
      f"{res_df['improvement_stitch'].median()*100:>14.1f}%  "
      f"{(res_df['improvement_stitch']>0).mean()*100:>14.0f}%")

print(f"\n  Per-camera (STITCH per-CCD):")
for cam, g in res_df.groupby("cam"):
    print(f"    Cam{int(cam)}: n={len(g):4d}  "
          f"STITCH={g['improvement_stitch'].median()*100:.1f}%  "
          f"Endpoint={g['improvement_endpoint'].median()*100:.1f}%")

# ── 5. Plot ───────────────────────────────────────────────────────────────────

examples = []
for cam in sorted(res_df["cam"].unique()):
    pool = [r for r in results if r["cam"] == cam and r["n_sectors"] >= 8]
    if pool:
        med = np.median([r["improvement_stitch"] for r in pool])
        examples.append(min(pool, key=lambda r: abs(r["improvement_stitch"] - med)))

n_ex = len(examples)
fig = plt.figure(figsize=(14, 4 + 3 * n_ex))
gs  = gridspec.GridSpec(n_ex + 1, 2, height_ratios=[2.5]*n_ex + [3],
                        hspace=0.55, wspace=0.35)
colors = {"raw": "#888888", "stitch": "#2166ac", "endpoint": "#d6604d"}

for i, r in enumerate(examples):
    ax = fig.add_subplot(gs[i, :])
    secs = r["sectors"]
    ax.plot(secs, r["raw_norm"],      "o-", color=colors["raw"],      lw=1.2, ms=4,
            label=f"Raw  (σ={r['scatter_before']:.4f})", alpha=0.7)
    ax.plot(secs, r["endpoint_norm"], "s-", color=colors["endpoint"], lw=1.2, ms=4,
            label=f"Endpoint  (σ={r['scatter_endpoint']:.4f}, {r['improvement_endpoint']*100:+.1f}%)")
    ax.plot(secs, r["stitch_norm"],   "o-", color=colors["stitch"],   lw=1.4, ms=5,
            label=f"STITCH per-CCD  (σ={r['scatter_stitch']:.4f}, {r['improvement_stitch']*100:+.1f}%)")
    ax.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.4)
    ax.set_ylabel("Norm. flux", fontsize=9)
    ax.set_title(f"TIC {r['tic_id']}  ·  Cam{r['cam']}  ·  {r['n_sectors']} sectors",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="upper right")
    if i == n_ex - 1:
        ax.set_xlabel("TESS Sector", fontsize=9)

ax_hist = fig.add_subplot(gs[n_ex, 0])
bins = np.linspace(-60, 60, 50)
ax_hist.hist(res_df["improvement_endpoint"]*100, bins=bins,
             color=colors["endpoint"], alpha=0.6, label="Endpoint", edgecolor="white", lw=0.3)
ax_hist.hist(res_df["improvement_stitch"]*100, bins=bins,
             color=colors["stitch"], alpha=0.6, label="STITCH per-CCD", edgecolor="white", lw=0.3)
ax_hist.axvline(0, color="k", lw=1.2, ls="--")
ax_hist.axvline(res_df["improvement_stitch"].median()*100,   color=colors["stitch"],   lw=2)
ax_hist.axvline(res_df["improvement_endpoint"].median()*100, color=colors["endpoint"], lw=2)
ax_hist.set_xlabel("Scatter improvement (%)", fontsize=10)
ax_hist.set_ylabel("Number of stars", fontsize=10)
ax_hist.set_title("STITCH per-CCD vs Endpoint baseline", fontsize=10)
ax_hist.legend(fontsize=9)

ax_sc = fig.add_subplot(gs[n_ex, 1])
ax_sc.scatter(res_df["improvement_endpoint"]*100, res_df["improvement_stitch"]*100,
              alpha=0.15, s=8, color="#444444")
lim = 60
ax_sc.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Equal performance")
ax_sc.axhline(0, color=colors["stitch"],   lw=1, ls=":")
ax_sc.axvline(0, color=colors["endpoint"], lw=1, ls=":")
ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
ax_sc.set_xlabel("Endpoint improvement (%)", fontsize=10)
ax_sc.set_ylabel("STITCH per-CCD improvement (%)", fontsize=10)
ax_sc.set_title("Star-by-star comparison", fontsize=10)
ax_sc.legend(fontsize=8)
frac_wins = (res_df["improvement_stitch"] > res_df["improvement_endpoint"]).mean()
ax_sc.text(0.05, 0.93, f"STITCH better: {frac_wins*100:.0f}% of stars",
           transform=ax_sc.transAxes, fontsize=8.5, color=colors["stitch"])

fig.suptitle("STITCH per-CCD — Before vs After Sector Stitching (held-out quiet stars)",
             fontsize=12, y=1.01)
plt.savefig("stitch_eval_per_ccd.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → stitch_eval_per_ccd.png")
