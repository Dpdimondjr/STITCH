"""
Per-cadence time-domain feature feasibility pilot.

Goal: test whether within-sector cadence SHAPE (trend slope, discontinuity size,
outlier burst fraction) carries signal beyond the blunt per-sector scalars we
already have (jitter_rms, pdcvar, cdpp1_0), by checking correlation with the
current model's residual.

Downloads a modest sample (prioritizing worst-residual stars from the failure
analysis, to maximize chance of finding signal) via lightkurve, extracts simple
cadence features per sector, and correlates with existing hgb_v2 residual.

This is a FEASIBILITY CHECK, not a production feature build. Network-bound,
designed to run alongside CPU-bound training jobs without resource contention.
"""

import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

T0 = time.time()
def log(m): print(m, flush=True)

# ── 1. Pick target TICs: worst-residual stars + a random comparison sample ──

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

N_WORST = 40
N_RANDOM = 40
worst_obs = test_df.reindex(test_df["resid"].abs().sort_values(ascending=False).index).head(200)
worst_pairs = worst_obs[["tic_id","sector","resid"]].drop_duplicates(subset=["tic_id"]).head(N_WORST)
rng = np.random.default_rng(42)
other = test_df[~test_df.tic_id.isin(worst_pairs.tic_id)]
other_one_per_tic = other.groupby("tic_id").apply(lambda g: g.sample(1, random_state=42), include_groups=False).reset_index()
random_pairs = other_one_per_tic.sample(n=min(N_RANDOM, len(other_one_per_tic)), random_state=42)[["tic_id","sector","resid"]]

# ONE (tic, sector) pair per star — not all sectors — to keep this tractable
target_obs = pd.concat([worst_pairs, random_pairs], ignore_index=True)
log(f"Target observations: {len(target_obs)} (one sector per star: {len(worst_pairs)} worst-residual, {len(random_pairs)} random)")

# ── 2. Download + extract cadence features ───────────────────────────────────

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
    # trend slope (linear fit over the sector)
    slope = np.polyfit(x, flux_norm, 1)[0] * len(flux_norm)  # total fractional drift over sector
    # discontinuity: largest single-cadence jump relative to local MAD
    diffs = np.diff(flux_norm)
    local_mad = np.median(np.abs(diffs - np.median(diffs))) * 1.4826 + 1e-9
    max_jump = np.max(np.abs(diffs)) / local_mad
    # outlier burst fraction: fraction of points >4 MAD from rolling median
    global_mad = np.median(np.abs(flux_norm - med/med)) * 1.4826 + 1e-9
    outlier_frac = (np.abs(flux_norm - np.median(flux_norm)) > 4*global_mad).mean()
    return dict(trend_slope=slope, max_jump=max_jump, outlier_frac=outlier_frac, n_cadences=len(flux_norm))

OUT_PATH = "eval/per_cadence_pilot_results.parquet"
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
    except Exception as e:
        n_fail += 1
    if (i+1) % 10 == 0:
        log(f"  {i+1}/{len(target_obs)} processed  ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")
        if results:
            pd.DataFrame(results).to_parquet(OUT_PATH)  # incremental save

log(f"\nDone. ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")

if n_ok < 20:
    log("Too few successful downloads for a meaningful correlation check. Pilot inconclusive on feasibility.")
    sys.exit(0)

res_df = pd.DataFrame(results)
res_df.to_parquet(OUT_PATH)
log(f"Saved -> {OUT_PATH} ({len(res_df)} rows)")

print("\n=== Correlation of cadence-shape features with |model residual| ===")
res_df["abs_resid"] = res_df["resid"].abs()
for f in ["trend_slope","max_jump","outlier_frac"]:
    r = res_df[f].corr(res_df["abs_resid"])
    print(f"  r({f}, |resid|) = {r:.4f}")

print(f"\nn_cadences stats: {res_df['n_cadences'].describe()}")
