"""
Injection-recovery test — does STITCH eat long-period astrophysical signals?

Takes quiet TARS test stars, injects a known sinusoidal signal at the
sector-median level, runs STITCH correction, and measures what fraction
of the injected amplitude survives.

A well-behaved corrector should:
  - Preserve signals with periods >> sector length (the signal is star-specific,
    not predictable from spatial neighbours → high pred_std → weight ≈ 0 → no correction)
  - Remove instrumental offsets (coherent across many stars on same CCD/sector)

Grid: amplitude × period. For each combination, recovery fraction =
  std(corrected_norm) / std(raw_norm_with_signal), where raw_norm_with_signal
  has the synthetic signal added. Perfect recovery = 1.0; STITCH eating signal < 1.0.

Usage:
  python3 eval/injection_recovery.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
  python3 eval/injection_recovery.py stitch_nsf_knn.pt training_data_topup_pdc.parquet --n-stars=200

Output:
  eval/injection_recovery.png  — heatmap of recovery fraction over amplitude × period grid
  eval/injection_recovery.csv  — full results table
"""

import sys
import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

MODEL    = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET  = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")
N_STARS  = int(next((s.split("=")[1] for s in sys.argv[1:] if s.startswith("--n-stars=")), 300))
SEED     = 42
print(f"Model: {MODEL}  |  Parquet: {PARQUET}  |  n_stars={N_STARS}")

# Injection grid
AMPLITUDES = [0.005, 0.010, 0.020, 0.050]   # fractional (0.5%, 1%, 2%, 5%)
PERIODS    = [27, 54, 81, 135, 200, 365]     # days (~1, 2, 3, 5, 7, 13 sectors)
SECTOR_DAYS = 27.4                            # approximate TESS sector duration

# ── Load model ─────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
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

# ── Load & clean ───────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[df["sector_median"] > 0]
df = df[df["n_sectors_total"] >= 4]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Test split ─────────────────────────────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))
tr_tics, tmp = train_test_split(
    star_cam["tic_id"], test_size=0.2,
    stratify=star_cam["dominant_cam"], random_state=SEED)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te_tics = train_test_split(
    tmp, test_size=0.5, stratify=tmp_cam.values, random_state=SEED)

# Keep test stars with >= 5 sectors for a meaningful signal measurement
test_df = df[df["tic_id"].isin(te_tics)].copy()
sc = test_df.groupby("tic_id")["sector"].count()
quiet_tics = sc[sc >= 5].index
test_df = test_df[test_df["tic_id"].isin(quiet_tics)].copy()

# Sample N_STARS quiet stars (low raw scatter, to avoid confounding)
raw_scatter = test_df.groupby("tic_id")["sector_median"].apply(
    lambda x: (x / x.mean()).std()
)
quiet_ranked = raw_scatter.sort_values().index[:N_STARS * 2]
rng = np.random.default_rng(SEED)
selected_tics = rng.choice(quiet_ranked, size=min(N_STARS, len(quiet_ranked)), replace=False)
test_df = test_df[test_df["tic_id"].isin(selected_tics)].copy()
print(f"Stars for injection test: {test_df['tic_id'].nunique():,}  records: {len(test_df):,}")

# ── Predict STITCH offsets (once, context doesn't change with injection) ────────
def make_context(d):
    cont   = (d[CONTINUOUS].reindex(columns=CONTINUOUS, fill_value=0) - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
flow = flow.to(device)
test_df = test_df.reset_index(drop=True)
C = torch.tensor(make_context(test_df)).to(device)

print("Running inference …", flush=True)
with torch.no_grad():
    samples  = flow(C).sample((300,)).squeeze(-1)
    mu_z     = samples.mean(0).cpu().numpy()
    sig_z    = samples.std(0).cpu().numpy()

pred_raw = mu_z * y_std + y_mean
pred_std = sig_z * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
pred     = weight * pred_raw + (1 - weight) * 1.0

test_df["pred_offset"] = pred
test_df["weight"]      = weight

# ── Injection-recovery loop ────────────────────────────────────────────────────
results = []
total = len(AMPLITUDES) * len(PERIODS)
done  = 0

for amp in AMPLITUDES:
    for period_days in PERIODS:
        recoveries = []
        for tic, g in test_df.groupby("tic_id"):
            g = g.sort_values("sector")
            # Sector mid-point in days (sector number × sector duration)
            t_mid   = g["sector"].values * SECTOR_DAYS
            raw     = g["sector_median"].values
            po      = g["pred_offset"].values

            # Inject sinusoid: random phase per star (unknown to STITCH)
            phase = rng.uniform(0, 2 * np.pi)
            signal = amp * np.sin(2 * np.pi * t_mid / period_days + phase)
            injected = raw * (1 + signal)

            # STITCH corrects using pred_offset (unchanged — context hasn't changed)
            stitch_norm = (injected / po) / (injected / po).mean()
            raw_norm    = injected / injected.mean()

            sc_before = raw_norm.std()
            sc_after  = stitch_norm.std()

            # Signal amplitude in the raw-normalised domain
            signal_norm    = (1 + signal) / (1 + signal).mean()
            signal_amp_raw = signal_norm.std()

            if signal_amp_raw < 1e-8:
                continue

            # Recovery: how much of the injected signal's scatter survives?
            # Perfect recovery = sc_after retains all signal variance on top of baseline.
            # We compare std of STITCH-corrected to std of raw (both with signal).
            recovery = sc_after / sc_before if sc_before > 0 else np.nan
            recoveries.append(recovery)

        med_rec = float(np.median(recoveries)) if recoveries else np.nan
        done += 1
        print(f"  [{done}/{total}] amp={amp:.1%}  period={period_days}d  "
              f"recovery={med_rec:.3f}  (n={len(recoveries)})", flush=True)
        results.append(dict(amplitude=amp, period_days=period_days,
                            median_recovery=med_rec, n_stars=len(recoveries)))

res = pd.DataFrame(results)
res.to_csv("eval/injection_recovery.csv", index=False)
print(f"\nSaved → eval/injection_recovery.csv")
print(res.pivot(index="amplitude", columns="period_days", values="median_recovery").to_string())

# ── Figure — heatmap ───────────────────────────────────────────────────────────
BG, SURF, TEXT, MUT = "#0b0f1a", "#0f1826", "#c4d4ee", "#7a9cc4"

pivot = res.pivot(index="amplitude", columns="period_days", values="median_recovery")
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(SURF)

im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.05,
               origin="lower")
ax.set_xticks(range(len(PERIODS)))
ax.set_xticklabels([f"{p}d" for p in PERIODS], color=TEXT, fontsize=10)
ax.set_yticks(range(len(AMPLITUDES)))
ax.set_yticklabels([f"{a:.1%}" for a in AMPLITUDES], color=TEXT, fontsize=10)
ax.set_xlabel("Injected period", color=MUT, fontsize=11)
ax.set_ylabel("Injected amplitude", color=MUT, fontsize=11)
ax.set_title("STITCH injection-recovery: fraction of scatter retained\n"
             "(1.0 = perfect recovery, < 1.0 = signal suppressed by correction)",
             color=TEXT, fontsize=11, pad=10)
for i in range(len(AMPLITUDES)):
    for j in range(len(PERIODS)):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="black" if 0.8 < v < 1.0 else "white", fontsize=9, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Median scatter retained", color=TEXT, fontsize=9)
cbar.ax.yaxis.set_tick_params(labelcolor=MUT, labelsize=8)
for sp in ax.spines.values(): sp.set_edgecolor("#1c2d45")

plt.tight_layout()
out = "eval/injection_recovery.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
