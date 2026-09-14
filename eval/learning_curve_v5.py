"""
Learning curve: v5 model efficiency vs training set size.

Pre-computes all v5 features once on the full dataset, then trains with
10/25/50/75/100% of training stars (by star count) and measures within-star
CV efficiency on a fixed test set. Answers: does more data still help?

Usage:
    python3 eval/learning_curve_v5.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, TensorDataset

PARQUET    = "training_data_topup_pdc.parquet"
TARGET     = "flux_offset_loo"
MIN_SECTS  = 4
TMAG_MAX   = 13.0
FRACTIONS  = [0.10, 0.25, 0.50, 0.75, 1.00]
N_REPEATS  = 2
N_SAMPLES  = 100   # samples for CV inference (fewer than full eval for speed)

TRANSFORMS = 8
HIDDEN     = [256, 256]
BINS       = 16
MAX_EPOCHS = 150
PATIENCE   = 15
BATCH_SIZE = 512
LR         = 3e-4

CONTINUOUS_BASE = ["col", "row", "delta_sub_col", "delta_sub_row",
                   "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar",
                   "jitter_rms", "pdc_noi", "pr_wght2",
                   "median_sap_bkg", "p90_sap_bkg", "bkg_rms",
                   "scatter_flag_frac", "flfrcsap", "gaiarp",
                   "perstar_gaia_offset"]
CONTINUOUS_V5   = CONTINUOUS_BASE + ["sector_ccd_mean_loo", "spatial_knn_mean_loo",
                                      "spatial_poly_pred", "loo_prev",
                                      "sector_gap", "n_sectors_total"]

device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device: {device}")

# ── Load & clean ───────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", TARGET])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]
df = df[(df[TARGET] > 0.85) & (df[TARGET] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTS]
df = df[df["tmag"] <= TMAG_MAX]
df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")
df = df.reset_index(drop=True)
print(f"  {len(df):,} records, {df['tic_id'].nunique():,} stars")

for col in CONTINUOUS_BASE:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Pre-compute v5 features ───────────────────────────────────────────────────
print("Computing sector_ccd_mean_loo …")
grp = df.groupby(["sector", "cam", "ccd"])[TARGET]
df["sector_ccd_mean_loo"] = (
    (grp.transform("sum") - df[TARGET]) / (grp.transform("count") - 1).clip(lower=1)
)

print("Computing spatial KNN (k=20) …")
_knn = np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2: continue
    _xy = _g[["col", "row"]].values.astype(np.float32)
    _v  = _g[TARGET].values
    _k  = min(20, _n - 1)
    _, _nn = cKDTree(_xy).query(_xy, k=_k + 1)
    _knn[_g.index.values] = _v[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = np.where(np.isnan(_knn), df["sector_ccd_mean_loo"], _knn)

print("Computing 2D polynomial spatial field …")
def _poly2d_design(c, r, degree=3):
    terms = []
    for d in range(degree + 1):
        for i in range(d + 1):
            terms.append(c ** (d - i) * r ** i)
    return np.column_stack(terms)

_poly = np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _idx = _g.index.values
    if len(_g) < 15:
        _poly[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32); continue
    _cc = _g["col"].values.astype(np.float64) - _g["col"].mean()
    _rr = _g["row"].values.astype(np.float64) - _g["row"].mean()
    _yy = _g[TARGET].values.astype(np.float64)
    _X  = _poly2d_design(_cc, _rr, degree=3)
    try:
        _cf, _, _, _ = np.linalg.lstsq(_X, _yy, rcond=None)
        _res = _yy - _X @ _cf
        _msk = np.abs(_res) <= 3 * (_res.std() or 1e-9)
        if _msk.sum() >= 15:
            _cf, _, _, _ = np.linalg.lstsq(_X[_msk], _yy[_msk], rcond=None)
        _poly[_idx] = np.clip(_X @ _cf, 0.85, 1.15).astype(np.float32)
    except Exception:
        _poly[_idx] = _g["sector_ccd_mean_loo"].values.astype(np.float32)
df["spatial_poly_pred"] = _poly

print("Computing temporal feature …")
_lp = np.full(len(df), np.nan, dtype=np.float64)
_sg = np.full(len(df), np.nan, dtype=np.float64)
for _tic, _g in df.groupby("tic_id"):
    _gs = _g.sort_values("sector"); _idx = _gs.index.values
    if len(_idx) < 2: continue
    _lp[_idx[1:]] = _gs[TARGET].values[:-1]
    _sg[_idx[1:]] = np.diff(_gs["sector"].values)
df["loo_prev"]   = np.where(np.isnan(_lp), df["sector_ccd_mean_loo"].values, _lp)
df["sector_gap"] = np.where(np.isnan(_sg), 99.0, np.clip(_sg, 1, 99))

# ── Fixed train/val/test split (same as v5 training) ─────────────────────────
sc = (df.groupby("tic_id")["cam"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index().rename(columns={"cam": "dom"}))
tr_tics, tmp = train_test_split(sc["tic_id"], test_size=0.2,
                                 stratify=sc["dom"], random_state=42)
tmp_cam = sc[sc["tic_id"].isin(tmp)]["dom"]
val_tics, te_tics = train_test_split(tmp, test_size=0.5,
                                      stratify=tmp_cam.values, random_state=42)

test_df  = df[df["tic_id"].isin(te_tics)].copy().reset_index(drop=True)
train_pool = df[df["tic_id"].isin(tr_tics)].copy()
all_train_tics = tr_tics.values

cam_cols = sorted([f"cam_{c}" for c in df["cam"].astype(int).unique()])
ccd_cols = sorted([f"ccd_{c}" for c in df["ccd"].astype(int).unique()])
CONTINUOUS = [c for c in CONTINUOUS_V5 if c in df.columns]

print(f"  Train pool: {len(all_train_tics):,} stars | Test: {test_df['tic_id'].nunique():,} stars")
print(f"  Context dim: {len(CONTINUOUS) + len(cam_cols) + len(ccd_cols)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_context(sdf, means, stds):
    cont   = (sdf[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(sdf["cam"].astype(int), prefix="cam").reindex(
                 columns=cam_cols, fill_value=0)
    ccd_oh = pd.get_dummies(sdf["ccd"].astype(int), prefix="ccd").reindex(
                 columns=ccd_cols, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)


def cv_efficiency(sdf, pred):
    sdf = sdf.copy()
    sdf["_raw"]    = sdf["sector_median"]
    sdf["_oracle"] = sdf["sector_median"] / sdf[TARGET]
    sdf["_model"]  = sdf["sector_median"] / pred

    def cv_med(col):
        return sdf.groupby("tic_id")[col].apply(
            lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan).dropna()

    cv_r = cv_med("_raw"); cv_o = cv_med("_oracle"); cv_m = cv_med("_model")
    common = cv_r.index.intersection(cv_o.index).intersection(cv_m.index)
    eff = ((cv_r[common] - cv_m[common]) /
           (cv_r[common] - cv_o[common]).clip(lower=1e-6)).median() * 100
    cv_val = cv_m[common].median() * 100
    return float(eff), float(cv_val)


def train_and_eval(train_df):
    means  = train_df[CONTINUOUS].mean()
    stds   = train_df[CONTINUOUS].std().replace(0, 1)
    y_mean = float(train_df[TARGET].mean())
    y_std  = float(train_df[TARGET].std())

    val_tic_sample = np.random.choice(
        train_df["tic_id"].unique(),
        size=max(1, train_df["tic_id"].nunique() // 10),
        replace=False)
    val_df = train_df[train_df["tic_id"].isin(val_tic_sample)]
    tr_df  = train_df[~train_df["tic_id"].isin(val_tic_sample)]

    C_tr = make_context(tr_df, means, stds)
    y_tr = ((tr_df[TARGET].values - y_mean) / y_std).astype(np.float32)
    C_va = torch.tensor(make_context(val_df, means, stds)).to(device)
    y_va = torch.tensor(((val_df[TARGET].values - y_mean) / y_std).astype(np.float32)).unsqueeze(-1).to(device)

    context_dim = C_tr.shape[1]
    flow = zuko.flows.NSF(features=1, context=context_dim,
                          transforms=TRANSFORMS, hidden_features=HIDDEN, bins=BINS)
    flow.to(device)
    opt   = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loader = DataLoader(
        TensorDataset(torch.tensor(C_tr), torch.tensor(y_tr).unsqueeze(-1)),
        batch_size=BATCH_SIZE, shuffle=True)

    best_val, best_state, pat = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        flow.train()
        for cb, yb in loader:
            cb, yb = cb.to(device), yb.to(device)
            nll = -flow(cb).log_prob(yb).mean()
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        flow.eval()
        with torch.no_grad():
            val_nll = -flow(C_va).log_prob(y_va).mean().item()
        sched.step(val_nll)
        if val_nll < best_val:
            best_val = val_nll; best_state = {k: v.clone() for k, v in flow.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    flow.load_state_dict(best_state); flow.eval()

    # Eval on fixed test set
    C_te = torch.tensor(make_context(test_df, means, stds)).to(device)
    with torch.no_grad():
        pred = flow(C_te).sample((N_SAMPLES,)).squeeze(-1).mean(0).cpu().numpy()
    pred = pred * y_std + y_mean
    eff, cv = cv_efficiency(test_df, pred)
    return eff, cv, best_val, epoch


# ── Learning curve ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  {'Frac':>6}  {'Stars':>7}  {'Eff (%)':>9}  {'CV (%)':>8}  {'Rep'}")
print(f"{'─'*60}")

results = {}
for frac in FRACTIONS:
    n_stars = max(50, int(len(all_train_tics) * frac))
    effs, cvs = [], []
    for rep in range(N_REPEATS):
        chosen = np.random.choice(all_train_tics, size=n_stars, replace=False)
        tr_df  = train_pool[train_pool["tic_id"].isin(chosen)]
        eff, cv, nll, ep = train_and_eval(tr_df)
        effs.append(eff); cvs.append(cv)
        print(f"  {frac:>6.0%}  {n_stars:>7,}  {eff:>9.1f}  {cv:>8.3f}  rep{rep+1} (ep{ep}, nll={nll:.4f})", flush=True)
    results[frac] = {"n_stars": n_stars, "eff": effs, "cv": cvs}

print(f"\n{'─'*60}")
print(f"  Summary (median across {N_REPEATS} repeats):")
print(f"  {'Frac':>6}  {'Stars':>7}  {'Eff (%)':>9}  {'CV (%)':>8}")
for frac in FRACTIONS:
    r = results[frac]
    print(f"  {frac:>6.0%}  {r['n_stars']:>7,}  {np.median(r['eff']):>9.1f}  {np.median(r['cv']):>8.3f}")
print(f"{'─'*60}")
