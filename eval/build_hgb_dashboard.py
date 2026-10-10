"""
Build the interactive STITCH lightcurve dashboard with HGB as the corrector.

Uses Method A split (80/10/10, stratified by cam, random_state=42).
HGB uses sector_mean_loo from non-test stars only (no leakage).
Spatial KNN features use train set as reference (no test-star signal).

Usage (from STITCH root):
  python3 eval/build_hgb_dashboard.py [parquet] [output.html]

Defaults:
  parquet = training_data_topup_pdc.parquet
  output  = stitch_hgb_dashboard.html
"""

import sys, os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree

pq_args  = [a for a in sys.argv[1:] if a.endswith(".parquet")]
htm_args = [a for a in sys.argv[1:] if a.endswith(".html")]
PARQUET  = pq_args[0]  if pq_args  else "training_data_v3.parquet"
OUT      = htm_args[0] if htm_args else "stitch_hgb_dashboard.html"
TARS_CSV = "tars_quiet_tics_v2.csv"

import time; T0 = time.time()
def log(msg): print(msg, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

log(f"Parquet: {PARQUET}")
log(f"Output : {OUT}\n")

# ── Load & clean ──────────────────────────────────────────────────────────────
log("Loading data …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","flux_offset_loo"])
df = df[(df["flux_offset"]     > 0.85) & (df["flux_offset"]     < 1.15)]
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
df = df[df["n_sectors_total"] >= 4]
df = df[df["tmag"] <= 13.0]
df = df.reset_index(drop=True)
log(f"  {len(df):,} records  ·  {df['tic_id'].nunique():,} stars")

# ── Method A split ────────────────────────────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: int(x.mode()[0]))
              .reset_index()
              .rename(columns={"cam":"dominant_cam"}))
tr_tics, tmp = train_test_split(
    star_cam["tic_id"], test_size=0.2, stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
vl_tics, te_tics = train_test_split(
    tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)
log(f"  Train {len(tr_tics):,}  Val {len(vl_tics):,}  Test {len(te_tics):,}")

train_df = df[df["tic_id"].isin(tr_tics)].copy().reset_index(drop=True)
val_df   = df[df["tic_id"].isin(vl_tics)].copy().reset_index(drop=True)
test_df  = df[df["tic_id"].isin(te_tics)].copy().reset_index(drop=True)
ref_full = df[~df["tic_id"].isin(te_tics)].copy().reset_index(drop=True)  # non-test only

# ── Sector aggregates (reference stars only — no test leakage) ─────────────
log("Sector aggregates from reference stars …")
sec_stats = (ref_full.groupby(["sector","cam","ccd"])["flux_offset_loo"]
    .agg(sector_mean_loo="mean", sector_std_loo="std",
         sector_frac_extreme=lambda x: (np.abs(x-1)>0.02).mean())
    .reset_index())
for df_ in [train_df, val_df, test_df]:
    df_.drop(columns=["sector_mean_loo","sector_std_loo","sector_frac_extreme"], errors="ignore", inplace=True)
    t = df_.merge(sec_stats, on=["sector","cam","ccd"], how="left")
    df_["sector_mean_loo"]     = t["sector_mean_loo"].fillna(1.0).values
    df_["sector_std_loo"]      = t["sector_std_loo"].fillna(0.0).values
    df_["sector_frac_extreme"] = t["sector_frac_extreme"].fillna(0.0).values

# ── KNN-100 baseline (all data, in-sector) ────────────────────────────────────
log("KNN-100 baseline …")
knn100 = np.full(len(df), 1.0, dtype=np.float64)
for (s,c,q), g in df.groupby(["sector","cam","ccd"]):
    n = len(g)
    if n < 2: knn100[g.index.values] = g["flux_offset_loo"].mean(); continue
    xy = g[["col","row"]].values.astype(np.float32)
    v  = g["flux_offset_loo"].values
    k  = min(100, n-1)
    _, nns = cKDTree(xy).query(xy, k=k+1)
    knn100[g.index.values] = v[nns[:,1:]].mean(axis=1)
for df_ in [train_df, val_df, test_df]:
    df_["pred_knn"] = knn100[df_.index.values]
log(f"  Done  {T()}")

# ── Spatial KNN features (train set as reference for all splits) ──────────────
log("Spatial KNN features …")
KNN_KS = [5, 10, 20, 50]
SPATIAL = ([f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]
           + ["ref_count"])

def add_spatial(ref_df, eval_df):
    cols = {c: np.full(len(eval_df), np.nan, dtype=np.float32) for c in SPATIAL}
    for (s,c,q), gr in ref_df.groupby(["sector","cam","ccd"]):
        ge = eval_df[(eval_df["sector"]==s)&(eval_df["cam"]==c)&(eval_df["ccd"]==q)]
        if not len(ge): continue
        nr = len(gr)
        xyr = gr[["col","row"]].values.astype(np.float32)
        xye = ge[["col","row"]].values.astype(np.float32)
        vr  = gr["flux_offset_loo"].values.astype(np.float32)
        tree = cKDTree(xyr)
        cols["ref_count"][ge.index.values] = nr
        for k in KNN_KS:
            kk = min(k, nr)
            if kk < 1: continue
            d, nn = tree.query(xye, k=kk)
            if kk == 1: nn = nn[:,None]; d = d[:,None]
            nbv = vr[nn]
            cols[f"knn{k}_mean"]  [ge.index.values] = nbv.mean(axis=1)
            cols[f"knn{k}_median"][ge.index.values] = np.median(nbv, axis=1)
            cols[f"knn{k}_mad"]   [ge.index.values] = np.abs(nbv - np.median(nbv,axis=1,keepdims=True)).mean(axis=1)
            cols[f"knn{k}_radius"][ge.index.values] = d.max(axis=1)
    for c_name, arr in cols.items():
        eval_df[c_name] = arr
    # Fill missing spatial with sector_mean_loo
    for c_name in [f"knn{k}_{s}" for k in KNN_KS for s in ["mean","median","mad","radius"]]:
        eval_df[c_name] = eval_df[c_name].fillna(eval_df["sector_mean_loo"])
    eval_df["ref_count"] = eval_df["ref_count"].fillna(0.0)

add_spatial(train_df, train_df)
add_spatial(train_df, val_df)
add_spatial(train_df, test_df)
log(f"  Done  {T()}")

# ── Feature list ──────────────────────────────────────────────────────────────
BASE    = ["col","row","delta_sub_col","delta_sub_row","sector","tmag",
           "crowdsap","cdpp1_0","pdcvar","jitter_rms"]
THERMAL = ["focal_plane_temp","focal_plane_temp_std"]
PDC_F   = ["pdc_corp","pdc_noi","pdc_totp","pdc_var","pdc_wght","pdc_wght2","pr_wght2","teff"]
MISC    = ["n_sectors_total","gaiarp","ra","dec"]
SEC_AGG = ["sector_mean_loo","sector_std_loo","sector_frac_extreme"]
TAB     = [c for c in BASE+THERMAL+PDC_F+MISC+SEC_AGG if c in train_df.columns]
FEATS   = TAB + SPATIAL
log(f"Features: {len(FEATS)}D ({len(TAB)} tab + {len(SPATIAL)} spatial)")

def get_X(df_):
    return df_[FEATS].copy().apply(
        lambda c: c.fillna(c.median()) if c.isna().any() else c
    ).values

def cv_per_star(df_, pred_col):
    """Median within-star CV% for corrected flux = sector_median / pred."""
    def _cv(g):
        r = g["sector_median"].values / g[pred_col].values
        return float(r.std() / r.mean()) * 100 if len(r) >= 2 else np.nan
    return df_.groupby("tic_id").apply(_cv)

# ── HGB: train once, sweep beta on val set ────────────────────────────────────
log("Training HGB …")
hgb = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42, n_iter_no_change=20)
hgb.fit(get_X(train_df), train_df["flux_offset_loo"].values)
log(f"  Trained  {T()}")

log("Beta sweep on val set …")
val_raw_pred = hgb.predict(get_X(val_df))
best_beta, best_val_cv = 0.8, 1e9
for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.0]:
    blend = np.clip((val_raw_pred**beta) * (val_df["pred_knn"].values**(1-beta)), 0.85, 1.15)
    val_df["_tmp"] = blend
    vcv = cv_per_star(val_df, "_tmp").median()
    log(f"  β={beta:.1f}  val_CV={vcv:.4f}%")
    if vcv < best_val_cv:
        best_val_cv = vcv
        best_beta   = beta
log(f"  → Best β={best_beta}  val_CV={best_val_cv:.4f}%")

# ── Test set predictions ──────────────────────────────────────────────────────
test_raw_pred = hgb.predict(get_X(test_df))
test_df["pred_hgb"] = np.clip(
    (test_raw_pred**best_beta) * (test_df["pred_knn"].values**(1-best_beta)), 0.85, 1.15)

test_df["_ones"] = 1.0
cv_raw = cv_per_star(test_df, "_ones").median()
cv_hgb = cv_per_star(test_df, "pred_hgb").median()
log(f"  HGB test_CV={cv_hgb:.4f}%  raw={cv_raw:.4f}%  {T()}")

# ── Build per-star data ───────────────────────────────────────────────────────
log("Building per-star data …")
tars_map = {}
if os.path.exists(TARS_CSV):
    tars = pd.read_csv(TARS_CSV)
    tars_map = {int(r.tic_id): float(r.mean_sys_score) for r in tars.itertuples()}

stars_out  = []
n_harmed   = n_harmed_n5 = n_cvz = 0
improvs    = []

for tic, g in test_df.sort_values(["tic_id","sector"]).groupby("tic_id"):
    tic = int(tic)
    raw_meds = g["sector_median"].tolist()
    cor_meds = (g["sector_median"] / g["pred_hgb"]).tolist()
    sectors  = [int(s) for s in g["sector"].tolist()]
    n        = len(sectors)
    raw_a    = np.array(raw_meds)
    cor_a    = np.array(cor_meds)
    sc_bef   = float(raw_a.std()/raw_a.mean()*100) if n >= 2 else 0.0
    sc_aft   = float(cor_a.std()/cor_a.mean()*100) if n >= 2 else 0.0
    improv   = float((sc_bef-sc_aft)/sc_bef*100)   if sc_bef > 0 else 0.0
    harmed   = sc_aft > sc_bef
    if harmed:          n_harmed   += 1
    if harmed and n>=5: n_harmed_n5 += 1
    if n >= 30:         n_cvz       += 1
    improvs.append(improv)
    entry = {
        "tic_id":    tic,
        "tmag":      round(float(g["tmag"].mean()), 2),
        "n":         n,
        "cam":       int(g["cam"].mode()[0]),
        "ccd":       int(g["ccd"].mode()[0]),
        "sc_before": round(sc_bef, 3),
        "sc_after":  round(sc_aft,  3),
        "improv":    round(improv,  1),
        "harmed":    harmed,
        "sectors":   sectors,
        "raw_meds":  [round(v,5) for v in raw_meds],
        "cor_meds":  [round(v,5) for v in cor_meds],
    }
    ss = tars_map.get(tic)
    if ss is not None: entry["sys_score"] = round(ss, 4)
    stars_out.append(entry)

imp_arr = [s["improv"] for s in stars_out if abs(s["improv"]) < 1000]
summary = {
    "model":        f"HGB β={best_beta} sector_mean_loo",
    "total_stars":  len(stars_out),
    "total_records": len(test_df),
    "harm_rate_all": round(n_harmed/len(stars_out)*100, 1),
    "harm_rate_n5":  round(n_harmed_n5/len(stars_out)*100, 1),
    "median_improv": round(float(np.median(imp_arr)), 1),
    "n_harmed":      n_harmed,
    "n_harmed_n5":   n_harmed_n5,
    "n_cvz":         n_cvz,
    "cv_raw_pct":    round(float(cv_raw), 4),
    "cv_hgb_pct":    round(float(cv_hgb), 4),
    "beta":          best_beta,
}
log(f"  {len(stars_out):,} stars  median_improv={summary['median_improv']}%  "
    f"harm_n5={summary['harm_rate_n5']}%  n_cvz={n_cvz}")

# ── Dataset-explorer metadata ─────────────────────────────────────────────────
split_of = {int(t):0 for t in tr_tics}
split_of.update({int(t):1 for t in vl_tics})
split_of.update({int(t):2 for t in te_tics})
md_split=[];md_cam=[];md_ccd=[];md_n=[];md_tmag=[];md_sc=[];md_sys=[]
for tic, g in df.groupby("tic_id"):
    sp = split_of.get(int(tic))
    if sp is None: continue
    g   = g.sort_values("sector")
    sm  = g["sector_median"].values
    sc  = float(sm.std()/sm.mean()*100) if len(sm) >= 2 else 0.0
    tmag= float(g["tmag"].iloc[0]) if "tmag" in g.columns else None
    md_split.append(sp); md_cam.append(int(g["cam"].mode()[0]))
    md_ccd.append(int(g["ccd"].mode()[0])); md_n.append(int(len(g)))
    md_tmag.append(round(tmag,2) if tmag else None)
    md_sc.append(round(sc,3))
    ss = tars_map.get(int(tic))
    md_sys.append(round(ss,4) if ss else None)

scs = md_sc
meta = {
    "tmag_min": round(min(v for v in md_tmag if v), 1),
    "tmag_max": round(max(v for v in md_tmag if v), 1),
    "n_min":    int(min(md_n)), "n_max": int(max(md_n)),
    "sc_max":   round(float(np.percentile(scs,99)), 1),
    "improv_min": round(float(np.percentile(imp_arr,1)), 0),
}

# ── Build HTML from build_lightcurve_dashboard.py template ────────────────────
log("Building HTML …")
lc_data = {"summary":summary,"stars":stars_out,"meta":meta}
lc_payload = json.dumps(lc_data, separators=(",",":"))
md_payload = json.dumps(
    {"split":md_split,"cam":md_cam,"ccd":md_ccd,
     "n":md_n,"tmag":md_tmag,"sc":md_sc,"sys":md_sys},
    separators=(",",":"))
log(f"  LC {len(lc_payload)//1024}KB  MD {len(md_payload)//1024}KB")

_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_lightcurve_dashboard.py")
with open(_script) as f:
    _src = f.read()
m = re.search(r'TEMPLATE\s*=\s*r"""(.*?)"""', _src, re.DOTALL)
if not m:
    raise RuntimeError("Could not extract TEMPLATE from build_lightcurve_dashboard.py")
html = m.group(1)

SUBS = {
    "%%TOTAL%%":     f"{len(stars_out):,}",
    "%%MEDIMPROV%%": f"{summary['median_improv']}%",
    "%%HARMRATE%%":  f"{summary['harm_rate_n5']}%",
    "%%NCVZ%%":      str(n_cvz),
    "%%MODEL%%":     summary["model"],
    "%%N_MAX%%":     str(meta["n_max"]),
    "%%TMAG_MIN%%":  str(meta["tmag_min"]),
    "%%TMAG_MAX%%":  str(meta["tmag_max"]),
    "%%SC_MAX%%":    str(meta["sc_max"]),
    "%%IMP_MIN%%":   str(int(meta["improv_min"])),
    "%%LC_JSON%%":   lc_payload,
    "%%MD_JSON%%":   md_payload,
}
for k, v in SUBS.items():
    html = html.replace(k, v)

with open(OUT, "w") as f:
    f.write(html)
log(f"\nWritten → {OUT}  ({os.path.getsize(OUT)/1024/1024:.1f} MB)  {T()}")
log("DONE")
