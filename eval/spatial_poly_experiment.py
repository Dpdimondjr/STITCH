"""
Spatial 2D polynomial experiment.

For each (sector, CCD) pair, fit a degree-2 polynomial through the measured
LOO flux_offsets of all TRAINING stars at their (col, row) positions. Predict
test stars by evaluating that polynomial at their positions.

Compare to v3 flow on per-star scatter reduction.

Three methods evaluated:
  flow     — v3 NSF model (current best)
  spatial  — 2D polynomial per (sector, CCD)
  blend    — weighted average of both
"""

import numpy as np, pandas as pd, torch, zuko
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

MODEL_PT = "stitch_nsf_v3.pt"
PARQUET  = "training_data_topup.parquet"
POLY_DEG = 2
MIN_TRAIN_STARS = 8   # min training stars per (sector,ccd) to fit poly; else fallback

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(features=cfg["features"], context=cfg["context"],
                      transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
                      bins=cfg["bins"])
flow.load_state_dict(ckpt["model_state"]); flow.eval()
means = ckpt["means"]; stds = ckpt["stds"]
y_mean = ckpt["y_mean"]; y_std = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS   = ckpt["cam_cols"]; CCD_COLS = ckpt["ccd_cols"]

# ── Load data + split ──────────────────────────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0]).reset_index()
              .rename(columns={"cam":"dominant_cam"}))
tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                           stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
_, te   = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)
train_tics = set(tr); test_tics = set(te)

train_df = df[df["tic_id"].isin(train_tics)].copy()
test_df  = df[df["tic_id"].isin(test_tics)].copy()
print(f"Train: {len(train_df):,} records ({train_df['tic_id'].nunique():,} stars)")
print(f"Test:  {len(test_df):,} records ({test_df['tic_id'].nunique():,} stars)")

# ── Flow predictions on test set ───────────────────────────────────────────────
print("Running flow inference...")
def make_context(d):
    cont   = (d[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(d["cam"].astype(int), prefix="cam").reindex(
                 columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(d["ccd"].astype(int), prefix="ccd").reindex(
                 columns=CCD_COLS, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

C = make_context(test_df)
with torch.no_grad():
    samp = flow(torch.tensor(C)).sample((300,)).squeeze(-1)
    pred_z     = samp.mean(0).numpy()
    pred_z_std = samp.std(0).numpy()

pred_raw = pred_z * y_std + y_mean
pred_std = pred_z_std * y_std
weight   = 1.0 / (1.0 + 5.0 * pred_std)
test_df  = test_df.copy()
test_df["pred_flow"] = 1.0 * (1 - weight) + pred_raw * weight

# ── Spatial 2D polynomial per (sector, CCD) ───────────────────────────────────
print("Fitting spatial polynomials...")

poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)

# normalise col/row to [0,1] for numerical stability
COL_MAX, ROW_MAX = 2048.0, 2048.0

def fit_poly(tr_sub):
    """Fit 2D polynomial to training stars in a (sector, ccd) cell."""
    X = np.column_stack([tr_sub["col"].values / COL_MAX,
                         tr_sub["row"].values / ROW_MAX])
    y = tr_sub["flux_offset"].values
    Xp = poly.fit_transform(X)
    reg = Ridge(alpha=1e-3)
    reg.fit(Xp, y)
    return reg

def eval_poly(reg, col, row):
    X = np.column_stack([np.array(col) / COL_MAX,
                         np.array(row) / ROW_MAX])
    Xp = poly.transform(X)
    return reg.predict(Xp)

# Build poly models per (sector, ccd) from training data
poly_models = {}
for (sec, ccd), g in train_df.groupby(["sector","ccd"]):
    if len(g) >= MIN_TRAIN_STARS:
        try:
            poly_models[(sec, ccd)] = fit_poly(g)
        except Exception:
            pass

print(f"Fitted polynomials for {len(poly_models):,} (sector, CCD) pairs "
      f"(min {MIN_TRAIN_STARS} training stars)")

# Predict for test stars
poly_preds = []
fallback_count = 0
for _, row in test_df.iterrows():
    key = (row["sector"], row["ccd"])
    if key in poly_models:
        pred = eval_poly(poly_models[key], [row["col"]], [row["row"]])[0]
        # sanity clip
        pred = float(np.clip(pred, 0.85, 1.15))
    else:
        pred = float(row["pred_flow"])   # fall back to flow
        fallback_count += 1
    poly_preds.append(pred)

test_df["pred_spatial"] = poly_preds
print(f"Fallback to flow (no poly model): {fallback_count:,} records "
      f"({fallback_count/len(test_df)*100:.1f}%)")

# Blend: equal weight for now
test_df["pred_blend"] = 0.5 * test_df["pred_flow"] + 0.5 * test_df["pred_spatial"]

# ── Per-star scatter comparison ────────────────────────────────────────────────
print("Computing per-star scatter...")
rows = []
for tic, g in test_df.groupby("tic_id"):
    if len(g) < 3:
        continue
    raw  = g["sector_median"].values
    ref  = raw.mean()
    raw_n = raw / ref

    def scatter(preds):
        cor = raw / preds
        return (cor / cor.mean()).std() * 100

    sc_raw     = raw_n.std() * 100
    sc_flow    = scatter(g["pred_flow"].values)
    sc_spatial = scatter(g["pred_spatial"].values)
    sc_blend   = scatter(g["pred_blend"].values)

    rows.append({
        "tic_id":      tic,
        "n":           len(g),
        "cam":         int(g["cam"].mode()[0]),
        "ccd":         int(g["ccd"].mode()[0]),
        "sc_raw":      sc_raw,
        "sc_flow":     sc_flow,
        "sc_spatial":  sc_spatial,
        "sc_blend":    sc_blend,
        "improv_flow":    (sc_raw - sc_flow)    / sc_raw * 100,
        "improv_spatial": (sc_raw - sc_spatial) / sc_raw * 100,
        "improv_blend":   (sc_raw - sc_blend)   / sc_raw * 100,
        "harmed_flow":    sc_flow    > sc_raw,
        "harmed_spatial": sc_spatial > sc_raw,
        "harmed_blend":   sc_blend   > sc_raw,
    })

res = pd.DataFrame(rows)
well = res[res["n"] >= 5]

# ── Print results ──────────────────────────────────────────────────────────────
print("\n" + "="*62)
print(f"{'Spatial Polynomial vs Flow  —  test set':^62}")
_n_stars = res["tic_id"].nunique()
print(f"({_n_stars:,} stars, well-obs n≥5: {len(well):,})".center(62))
print("="*62)
print(f"{'Metric':<38} {'Flow':>7} {'Spatial':>8} {'Blend':>7}")
print("-"*62)

metrics = [
    ("Median improv % (all)",
     res["improv_flow"].median(), res["improv_spatial"].median(), res["improv_blend"].median()),
    ("Median improv % (n≥5)",
     well["improv_flow"].median(), well["improv_spatial"].median(), well["improv_blend"].median()),
    ("Mean improv % (n≥5)",
     well["improv_flow"].mean(), well["improv_spatial"].mean(), well["improv_blend"].mean()),
    ("Harm rate % (all)",
     res["harmed_flow"].mean()*100, res["harmed_spatial"].mean()*100, res["harmed_blend"].mean()*100),
    ("Harm rate % (n≥5)",
     well["harmed_flow"].mean()*100, well["harmed_spatial"].mean()*100, well["harmed_blend"].mean()*100),
]
for label, f, s, b in metrics:
    print(f"  {label:<36} {f:>7.1f} {s:>8.1f} {b:>7.1f}")

print("\n  By camera (n≥5, median improv %):")
print(f"  {'Cam':<6} {'n':>5}  {'Flow':>7} {'Spatial':>8} {'Blend':>7}  {'Harm-flow':>10} {'Harm-spat':>10}")
for cam in [1,2,3,4]:
    sub = well[well["cam"]==cam]
    if len(sub) == 0: continue
    print(f"  Cam{cam:<3} {len(sub):>5}  "
          f"{sub['improv_flow'].median():>7.1f} "
          f"{sub['improv_spatial'].median():>8.1f} "
          f"{sub['improv_blend'].median():>7.1f}  "
          f"{sub['harmed_flow'].mean()*100:>9.1f}% "
          f"{sub['harmed_spatial'].mean()*100:>9.1f}%")

print("\n  By n_sectors bin (median improv %):")
bins = [(3,5),(6,9),(10,14),(15,25),(26,50)]
print(f"  {'n_sectors':<12} {'n':>5}  {'Flow':>7} {'Spatial':>8} {'Blend':>7}")
for lo, hi in bins:
    sub = res[(res["n"]>=lo) & (res["n"]<=hi)]
    if len(sub) == 0: continue
    print(f"  {lo}-{hi:<9} {len(sub):>5}  "
          f"{sub['improv_flow'].median():>7.1f} "
          f"{sub['improv_spatial'].median():>8.1f} "
          f"{sub['improv_blend'].median():>7.1f}")

# ── How many (sector,CCD) cells have enough stars? ────────────────────────────
cell_counts = train_df.groupby(["sector","ccd"]).size()
print(f"\n  (sector,CCD) cells with ≥{MIN_TRAIN_STARS} training stars: "
      f"{(cell_counts>=MIN_TRAIN_STARS).sum()} / {len(cell_counts)}")
print(f"  Median training stars per cell: {cell_counts.median():.0f}")
print(f"  Min: {cell_counts.min()}  Max: {cell_counts.max()}")
