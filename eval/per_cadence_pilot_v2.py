"""
Per-cadence time-domain feature validation — v2, PROPERLY RANDOM sample.

v1 (overnight) stratified 50/50 worst-residual/random and found r=0.16-0.18 for
max_jump/outlier_frac, but with n=80 and a biased sampling design the SE was too
large to trust (~0.2). This version draws a PURE random sample of test stars
(no stratification by residual) at a larger N, so the resulting correlation
estimate is actually interpretable at face value.
"""

import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

T0 = time.time()
def log(m): print(m, flush=True)

preds = pd.read_parquet("/tmp/hgb_v2_preds.parquet")
orig = pd.read_parquet("training_data_v3_tic.parquet", columns=["tic_id","sector","cam","ccd","flux_offset_loo"])
orig["tic_id"]=orig["tic_id"].astype("int64"); orig["sector"]=orig["sector"].astype("int64")
preds["tic_id"]=preds["tic_id"].astype("int64"); preds["sector"]=preds["sector"].astype("int64")
df = preds.merge(orig, on=["tic_id","sector","cam","ccd"], how="left")
df = df.dropna(subset=["flux_offset_loo","pred_hgb_v2"])
df = df[df["flux_offset_loo"].between(0.85,1.15)]
df["resid"] = df["flux_offset_loo"]/df["pred_hgb_v2"] - 1.0

test_tics = set(np.load("models/hgb_v2/test_tic_ids.npy").tolist())
test_df = df[df.tic_id.isin(test_tics)].copy()

N_TARGET = 300
rng = np.random.default_rng(7)
one_per_tic = test_df.groupby("tic_id").apply(lambda g: g.sample(1, random_state=7), include_groups=False).reset_index()
target_obs = one_per_tic.sample(n=min(N_TARGET, len(one_per_tic)), random_state=7)[["tic_id","sector","resid"]].reset_index(drop=True)
log(f"PURE RANDOM target: {len(target_obs)} stars, one sector each")

import lightkurve as lk

def cadence_features(flux, quality=None):
    flux = np.asarray(flux, dtype=float)
    mask = np.isfinite(flux)
    if quality is not None:
        mask &= (np.asarray(quality) == 0)
    flux = flux[mask]
    if len(flux) < 50:
        return None
    med = np.median(flux)
    flux_norm = flux / med
    x = np.arange(len(flux_norm))
    slope = np.polyfit(x, flux_norm, 1)[0] * len(flux_norm)
    diffs = np.diff(flux_norm)
    local_mad = np.median(np.abs(diffs - np.median(diffs))) * 1.4826 + 1e-9
    max_jump = np.max(np.abs(diffs)) / local_mad
    global_mad = np.median(np.abs(flux_norm - np.median(flux_norm))) * 1.4826 + 1e-9
    outlier_frac = (np.abs(flux_norm - np.median(flux_norm)) > 4*global_mad).mean()
    # additional feature: rolling-window variance ratio (first half vs second half of sector)
    half = len(flux_norm)//2
    var_ratio = (np.var(flux_norm[half:]) + 1e-12) / (np.var(flux_norm[:half]) + 1e-12)
    return dict(trend_slope=slope, max_jump=max_jump, outlier_frac=outlier_frac,
                var_ratio=var_ratio, n_cadences=len(flux_norm))

OUT_PATH = "eval/per_cadence_pilot_v2_results.parquet"
results = []
n_ok, n_fail = 0, 0
for i, (tic, sec, resid) in enumerate(target_obs.itertuples(index=False)):
    try:
        sr = lk.search_lightcurve(f"TIC {int(tic)}", mission="TESS", author="TESS-SPOC", sector=int(sec))
        if len(sr) == 0:
            n_fail += 1
            continue
        lc = sr[0].download(download_dir="tess_cache", quality_bitmask="none")
        if lc is None or "pdcsap_flux" not in lc.colnames:
            n_fail += 1
            continue
        feats = cadence_features(lc["pdcsap_flux"].value, lc["quality"].value if "quality" in lc.colnames else None)
        if feats is None:
            n_fail += 1
            continue
        feats.update(tic_id=tic, sector=sec, resid=resid)
        results.append(feats)
        n_ok += 1
    except Exception:
        n_fail += 1
    if (i+1) % 20 == 0:
        log(f"  {i+1}/{len(target_obs)} processed  ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")
        if results:
            pd.DataFrame(results).to_parquet(OUT_PATH)

log(f"\nDone. ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")

if n_ok < 50:
    log("Too few successful downloads for a reliable correlation check.")
    sys.exit(0)

res_df = pd.DataFrame(results)
res_df.to_parquet(OUT_PATH)
log(f"Saved -> {OUT_PATH} ({len(res_df)} rows)")

print("\n=== PURE RANDOM sample: correlation of cadence-shape features with |model residual| ===")
res_df["abs_resid"] = res_df["resid"].abs()
n = len(res_df)
se = 1/np.sqrt(n-3)
print(f"n={n}  approx SE on r = {se:.4f}  (95% CI roughly ± {1.96*se:.4f})")
for f in ["trend_slope","max_jump","outlier_frac","var_ratio"]:
    r = res_df[f].corr(res_df["abs_resid"])
    sig = "SIGNIFICANT" if abs(r) > 1.96*se else "not significant at 95%"
    print(f"  r({f}, |resid|) = {r:+.4f}   [{sig}]")
