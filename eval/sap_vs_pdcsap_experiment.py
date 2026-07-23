"""
SAP vs PDCSAP learnability experiment.

For a sample of well-observed stars, extract both SAP and PDCSAP sector medians,
compute LOO offsets for each, and measure how much variance is explained by
our context features. Higher R² = more learnable by STITCH.

Metrics:
  - LOO scatter distribution (raw signal size)
  - Linear R² from context features
  - Residual std after linear fit (irreducible noise floor)
  - NSF NLL on a held-out subset (model quality comparison)
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, zuko
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, sys

CACHE_DIR  = "./tess_cache"
PARQUET    = "training_data_topup.parquet"
N_SAMPLE   = 200    # stars to download
MIN_SECS   = 12     # minimum sectors per star
N_WORKERS  = 4
SEED       = 42

# ── Load existing parquet to pick sample ─────────────────────────────────────
print("Loading parquet...")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]

ns = df.groupby("tic_id")["sector"].nunique()
good_tics = ns[ns >= MIN_SECS].index.tolist()

# Stratify by cam
rng = np.random.default_rng(SEED)
cam_map = df.groupby("tic_id")["cam"].agg(lambda x: x.mode()[0])
sample = []
for cam in [1, 2, 3, 4]:
    pool = [t for t in good_tics if cam_map.get(t) == cam]
    k = N_SAMPLE // 4
    sample += rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
print(f"Sample: {len(sample)} stars  (min {MIN_SECS} sectors, stratified by cam)")

# ── Download and extract SAP + PDCSAP ────────────────────────────────────────
def safe_float(v):
    try: return float(v)
    except: return np.nan

def process_star(tic_id):
    try:
        tic_id = int(tic_id)
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
        try: sr = sr[sr.exptime.value >= 100]
        except: pass
        if len(sr) == 0:
            return None

        lc_list = []
        for i in range(len(sr)):
            try:
                lc = sr[i].download(download_dir=CACHE_DIR)
                if lc is not None:
                    lc_list.append(lc)
            except:
                pass
        if len(lc_list) < MIN_SECS:
            return None

        ra  = float(lc_list[0].meta["RA_OBJ"])
        dec = float(lc_list[0].meta["DEC_OBJ"])
        _, _, _, out_sec, out_cam, out_ccd, out_col, out_row, _ = \
            tess_stars2px_function_entry(tic_id, ra, dec)
        pos = {int(s): (int(c1), int(c2), float(cl), float(rw))
               for s, c1, c2, cl, rw in zip(out_sec, out_cam, out_ccd, out_col, out_row)}

        rows = []
        for lc in lc_list:
            sec = lc.meta.get("SECTOR")
            if sec is None or int(sec) not in pos:
                continue
            cam_tp, ccd_tp, col, row = pos[int(sec)]
            try:
                pc1 = lc["pos_corr1"].value.astype(float)
                pc2 = lc["pos_corr2"].value.astype(float)
                jitter  = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
                pc1_med = float(np.nanmedian(pc1))
                pc2_med = float(np.nanmedian(pc2))
            except:
                pc1_med = pc2_med = jitter = np.nan

            pdcsap = lc.flux.value.astype(float)
            try:
                sap = lc["SAP_FLUX"].value.astype(float)
            except:
                sap = None

            pdcsap_med = float(np.nanmedian(pdcsap))
            sap_med    = float(np.nanmedian(sap)) if sap is not None else np.nan

            rows.append({
                "tic_id": tic_id, "sector": int(sec),
                "cam": cam_tp, "ccd": ccd_tp, "col": col, "row": row,
                "delta_sub_col": (col + pc1_med) % 1.0,
                "delta_sub_row": (row + pc2_med) % 1.0,
                "tmag":       safe_float(lc.meta.get("TESSMAG")),
                "crowdsap":   safe_float(lc.meta.get("CROWDSAP")),
                "cdpp1_0":    safe_float(lc.meta.get("CDPP1_0")),
                "pdcvar":     safe_float(lc.meta.get("PDCVAR")),
                "jitter_rms": jitter,
                "pdcsap_med": pdcsap_med,
                "sap_med":    sap_med,
            })

        if len(rows) < MIN_SECS:
            return None
        return rows
    except Exception as e:
        print(f"  ERR {tic_id}: {e}", flush=True)
        return None

print(f"\nDownloading {len(sample)} stars ({N_WORKERS} workers)...")
all_rows = []
done = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(process_star, t): t for t in sample}
    for f in as_completed(futs):
        done += 1
        r = f.result()
        if r:
            all_rows.extend(r)
        if done % 20 == 0:
            print(f"  {done}/{len(sample)} done  ({len(all_rows)} sector records so far)")

raw = pd.DataFrame(all_rows)
raw = raw.dropna(subset=["pdcsap_med"])
raw = raw[raw["sap_med"].notna()]
print(f"\nCollected {len(raw):,} sector records from {raw['tic_id'].nunique()} stars")

# ── Compute LOO offsets for both ──────────────────────────────────────────────
def compute_loo(df_in, med_col):
    rows = []
    for tic, g in df_in.groupby("tic_id"):
        meds = g[med_col].values
        secs = g["sector"].values
        if len(meds) < 3:
            continue
        for i in range(len(meds)):
            others = np.delete(meds, i)
            ref    = others.mean()
            if ref <= 0:
                continue
            rows.append({
                "tic_id":  tic,
                "sector":  secs[i],
                "loo_off": meds[i] / ref,
            })
    return pd.DataFrame(rows)

print("Computing LOO offsets...")
loo_pdc = compute_loo(raw, "pdcsap_med").rename(columns={"loo_off": "loo_pdc"})
loo_sap = compute_loo(raw, "sap_med").rename(columns={"loo_off": "loo_sap"})
merged  = raw.merge(loo_pdc, on=["tic_id","sector"]).merge(loo_sap, on=["tic_id","sector"])

# Clip to sane range
merged = merged[(merged["loo_pdc"] > 0.85) & (merged["loo_pdc"] < 1.15)]
merged = merged[(merged["loo_sap"] > 0.75) & (merged["loo_sap"] < 1.25)]
print(f"After LOO clip: {len(merged):,} records")

# ── Feature matrix ────────────────────────────────────────────────────────────
FEATS = ["col","row","delta_sub_col","delta_sub_row","sector","tmag",
         "crowdsap","cdpp1_0","pdcvar","jitter_rms"]
for f in FEATS:
    if merged[f].isna().any():
        merged[f] = merged[f].fillna(merged[f].median())

cam_oh = pd.get_dummies(merged["cam"].astype(int), prefix="cam")
ccd_oh = pd.get_dummies(merged["ccd"].astype(int), prefix="ccd")
X_raw  = pd.concat([merged[FEATS], cam_oh, ccd_oh], axis=1).values.astype(np.float32)
sc     = StandardScaler(); X = sc.fit_transform(X_raw)

y_pdc  = merged["loo_pdc"].values.astype(np.float32)
y_sap  = merged["loo_sap"].values.astype(np.float32)

# ── Linear R² (5-fold CV) ─────────────────────────────────────────────────────
print("\nFitting linear models (5-fold CV)...")
ridge = Ridge(alpha=1.0)
r2_pdc = cross_val_score(ridge, X, y_pdc, cv=5, scoring="r2").mean()
r2_sap = cross_val_score(ridge, X, y_sap, cv=5, scoring="r2").mean()

# Fit for residuals
ridge.fit(X, y_pdc); resid_pdc = y_pdc - ridge.predict(X)
ridge.fit(X, y_sap); resid_sap = y_sap - ridge.predict(X)

# ── Per-star scatter ──────────────────────────────────────────────────────────
def star_scatter(df_in, col):
    rows = []
    for tic, g in df_in.groupby("tic_id"):
        v = g[col].values
        rows.append(v.std() * 100)
    return np.array(rows)

sc_pdc = star_scatter(merged, "loo_pdc")
sc_sap = star_scatter(merged, "loo_sap")

# ── NSF NLL comparison ────────────────────────────────────────────────────────
print("Training two small NSF models for NLL comparison...")
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def train_small_flow(X_tr, y_tr, X_va, y_va, label):
    flow = zuko.flows.NSF(features=1, context=X_tr.shape[1],
                          transforms=4, hidden_features=[64,64], bins=8)
    opt  = torch.optim.Adam(flow.parameters(), lr=3e-4)
    ds   = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(-1)),
                      batch_size=256, shuffle=True)
    Xv   = torch.tensor(X_va); yv = torch.tensor(y_va).unsqueeze(-1)
    best_nll = float("inf"); best_ep = 0
    for ep in range(1, 81):
        flow.train()
        for xb, yb in ds:
            loss = -flow(xb).log_prob(yb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        flow.eval()
        with torch.no_grad():
            val_nll = -flow(Xv).log_prob(yv).mean().item()
        if val_nll < best_nll:
            best_nll = val_nll; best_ep = ep
        if ep - best_ep >= 15:
            break
    print(f"  {label}: best val NLL = {best_nll:.4f} (ep {best_ep})")
    return best_nll

# standardise targets for flow
ym_pdc = y_pdc.mean(); ys_pdc = y_pdc.std()
ym_sap = y_sap.mean(); ys_sap = y_sap.std()
yn_pdc = ((y_pdc - ym_pdc) / ys_pdc).astype(np.float32)
yn_sap = ((y_sap - ym_sap) / ys_sap).astype(np.float32)

idx = np.arange(len(X))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=42)
X_f  = X.astype(np.float32)

nll_pdc = train_small_flow(X_f[tr_idx], yn_pdc[tr_idx], X_f[va_idx], yn_pdc[va_idx], "PDCSAP")
nll_sap = train_small_flow(X_f[tr_idx], yn_sap[tr_idx], X_f[va_idx], yn_sap[va_idx], "SAP   ")

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*58)
print("SAP vs PDCSAP Learnability Experiment")
print("="*58)
print(f"Stars: {merged['tic_id'].nunique()}  Records: {len(merged):,}\n")

print(f"{'Metric':<35} {'PDCSAP':>10} {'SAP':>10}")
print("-"*58)
print(f"{'Median LOO scatter (%)':<35} {np.median(sc_pdc):>10.4f} {np.median(sc_sap):>10.4f}")
print(f"{'Mean LOO scatter (%)':<35} {sc_pdc.mean():>10.4f} {sc_sap.mean():>10.4f}")
print(f"{'LOO std (all records)':<35} {y_pdc.std()*100:>10.4f} {y_sap.std()*100:>10.4f}")
print(f"{'Linear R² (context features)':<35} {r2_pdc:>10.4f} {r2_sap:>10.4f}")
print(f"{'Residual std after linear fit':<35} {resid_pdc.std()*100:>10.4f} {resid_sap.std()*100:>10.4f}")
print(f"{'NSF val NLL (small flow)':<35} {nll_pdc:>10.4f} {nll_sap:>10.4f}")
print("="*58)

frac_pdc = r2_pdc
frac_sap = r2_sap
print(f"\nConclusion:")
print(f"  Context features explain {frac_pdc*100:.1f}% of PDCSAP LOO variance")
print(f"  Context features explain {frac_sap*100:.1f}% of SAP LOO variance")
winner = "SAP" if frac_sap > frac_pdc else "PDCSAP"
print(f"  → {winner} is more learnable from context features")
print(f"\n  SAP signal is {y_sap.std()/y_pdc.std():.1f}× larger than PDCSAP signal")
print(f"  SAP residual floor is {resid_sap.std()/resid_pdc.std():.1f}× larger than PDCSAP residual floor")

# save for inspection
merged[["tic_id","sector","cam","ccd","loo_pdc","loo_sap"]].to_csv(
    "eval/sap_vs_pdcsap_loo.csv", index=False)
print(f"\nPer-sector LOO values saved → eval/sap_vs_pdcsap_loo.csv")
