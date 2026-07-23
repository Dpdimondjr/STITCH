"""
Export per-star test-set results to JSON for the eval dashboard.
Outputs: eval/dashboard_data.json
"""

import json, sys, numpy as np, pandas as pd, torch, zuko
from sklearn.model_selection import train_test_split

MODEL_PT = next((s for s in sys.argv[1:] if s.endswith(".pt")), "stitch_nsf_v3.pt")
PARQUET  = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup.parquet")
OUT      = "eval/dashboard_data.json"
print(f"Model: {MODEL_PT}  Parquet: {PARQUET}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()
means      = ckpt["means"];  stds  = ckpt["stds"]
y_mean     = ckpt["y_mean"]; y_std = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]; CCD_COLS = ckpt["ccd_cols"]

# ── Load + clean data ─────────────────────────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
if "log_sector_median" not in df.columns:
    df["log_sector_median"] = np.log1p(df["sector_median"].clip(lower=0))
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Reproduce train/test split ────────────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0]).reset_index()
              .rename(columns={"cam": "dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                           stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)
test_df = df[df["tic_id"].isin(set(te))].copy()
print(f"Test records: {len(test_df):,}  TICs: {test_df['tic_id'].nunique():,}")

# ── Inference ─────────────────────────────────────────────────────────────────
print("Running inference...")
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

C = make_context(test_df)
with torch.no_grad():
    samples    = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
    pred_z     = samples.mean(0).numpy()
    pred_z_std = samples.std(0).numpy()

pred_raw = pred_z * y_std + y_mean
pred_std = pred_z_std * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
test_df  = test_df.copy()
test_df["pred_offset"] = 1.0 * (1 - weight) + pred_raw * weight

# ── Per-star results ──────────────────────────────────────────────────────────
print("Building per-star results...")
stars = []
for tic, g in test_df.groupby("tic_id"):
    g = g.sort_values("sector")
    raw   = g["sector_median"].values
    pred  = g["pred_offset"].values
    secs  = g["sector"].astype(int).tolist()
    ref   = raw.mean()
    raw_n = raw / ref
    cor_n = (raw / pred) / (raw / pred).mean()

    sc_before = float(raw_n.std() * 100)
    sc_after  = float(cor_n.std() * 100)
    improv    = float((sc_before - sc_after) / sc_before * 100) if sc_before > 0 else 0.0
    harmed    = bool(sc_after > sc_before)

    stars.append({
        "tic_id":     int(tic),
        "tmag":       round(float(g["tmag"].iloc[0]), 2) if "tmag" in g else None,
        "n":          len(g),
        "cam":        int(g["cam"].mode()[0]),
        "ccd":        int(g["ccd"].mode()[0]),
        "sc_before":  round(sc_before, 3),
        "sc_after":   round(sc_after, 3),
        "improv":     round(improv, 1),
        "harmed":     harmed,
        "sectors":    secs,
        "raw_meds":   [round(float(v), 5) for v in raw_n],
        "cor_meds":   [round(float(v), 5) for v in cor_n],
    })

stars.sort(key=lambda s: s["improv"], reverse=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
res = pd.DataFrame(stars)
well = res[res["n"] >= 5]
summary = {
    "model":         MODEL_PT,
    "total_stars":   len(stars),
    "total_records": len(test_df),
    "harm_rate_all": round(res["harmed"].mean() * 100, 1),
    "harm_rate_n5":  round(well["harmed"].mean() * 100, 1),
    "median_improv": round(well["improv"].median(), 1),
    "n_harmed":      int(res["harmed"].sum()),
    "n_harmed_n5":   int(well["harmed"].sum()),
    "n_cvz":         int((res["n"] >= 30).sum()),
}

out = {"summary": summary, "stars": stars}
import os; os.makedirs("eval", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"Saved {len(stars):,} stars → {OUT}  ({os.path.getsize(OUT)//1024} KB)")
print("\nSummary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
