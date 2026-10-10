"""
Focal-plane pixel-level feature feasibility pilot — the MVP first step toward
a "model the field, not the star" architecture.

Rationale: every feature used so far (KNN, destripe, pointing, contratio) is
derived from APERTURE-SUMMED flux or a single pipeline-computed centroid. None
of them look at the actual PIXEL DISTRIBUTION. A contaminating neighbor source
creates a visible secondary peak or asymmetry in the pixel cutout that aperture
photometry blends together — this pilot tests whether that raw shape information
carries signal the existing tabular features (including contratio/numcont) miss.

Downloads TESS-SPOC Target Pixel Files (11x11 cutouts, native cadence) for a
random sample of test stars, extracts simple per-sector shape features, and
checks:
  1. raw correlation with |residual|
  2. PARTIAL correlation controlling for contratio/numcont — i.e. does pixel
     shape add anything BEYOND what the catalog crowding features already give,
     or is it redundant?
"""

import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

T0 = time.time()
def log(m): print(m, flush=True)

preds = pd.read_parquet("/tmp/hgb_v2_preds.parquet")
orig = pd.read_parquet("training_data_v3_tic.parquet",
    columns=["tic_id","sector","cam","ccd","flux_offset_loo","contratio","numcont"])
orig["tic_id"]=orig["tic_id"].astype("int64"); orig["sector"]=orig["sector"].astype("int64")
preds["tic_id"]=preds["tic_id"].astype("int64"); preds["sector"]=preds["sector"].astype("int64")
df = preds.merge(orig, on=["tic_id","sector","cam","ccd"], how="left")
df = df.dropna(subset=["flux_offset_loo","pred_hgb_v2","contratio"])
df = df[df["flux_offset_loo"].between(0.85,1.15)]
df["resid"] = df["flux_offset_loo"]/df["pred_hgb_v2"] - 1.0

test_tics = set(np.load("models/hgb_v2/test_tic_ids.npy").tolist())
test_df = df[df.tic_id.isin(test_tics)].copy()

N_TARGET = 120
rng = np.random.default_rng(13)
one_per_tic = test_df.groupby("tic_id").apply(lambda g: g.sample(1, random_state=13), include_groups=False).reset_index()
target_obs = one_per_tic.sample(n=min(N_TARGET, len(one_per_tic)), random_state=13)[
    ["tic_id","sector","resid","contratio","numcont"]].reset_index(drop=True)
log(f"PURE RANDOM pixel-pilot target: {len(target_obs)} stars, one sector each")

import lightkurve as lk

def pixel_features(tpf):
    flux = tpf.flux.value  # (n_cadences, h, w)
    # drop cadences with any NaN
    good = np.isfinite(flux).all(axis=(1,2))
    flux = flux[good]
    if len(flux) < 20:
        return None
    h, w = flux.shape[1], flux.shape[2]
    cy, cx = h/2 - 0.5, w/2 - 0.5
    yy, xx = np.mgrid[0:h, 0:w]

    med_img = np.median(flux, axis=0)
    med_img_bgsub = med_img - np.percentile(med_img, 10)
    med_img_bgsub = np.clip(med_img_bgsub, 0, None)
    total = med_img_bgsub.sum()
    if total <= 0:
        return None

    # 1. flux-weighted centroid offset from cutout center (on the median image)
    cen_y = (yy * med_img_bgsub).sum() / total
    cen_x = (xx * med_img_bgsub).sum() / total
    centroid_offset = np.hypot(cen_y - cy, cen_x - cx)

    # 2. per-cadence centroid jitter (std of centroid across cadences) — pixel-level pointing noise
    cen_ys, cen_xs = [], []
    for i in range(0, len(flux), max(1, len(flux)//200)):  # subsample cadences for speed
        img = np.clip(flux[i] - np.percentile(flux[i], 10), 0, None)
        t = img.sum()
        if t <= 0: continue
        cen_ys.append((yy*img).sum()/t)
        cen_xs.append((xx*img).sum()/t)
    centroid_jitter = np.hypot(np.std(cen_ys), np.std(cen_xs)) if len(cen_ys) > 3 else np.nan

    # 3. asymmetry: compare flux-weighted second moments along two diagonals
    var_y = ((yy - cen_y)**2 * med_img_bgsub).sum() / total
    var_x = ((xx - cen_x)**2 * med_img_bgsub).sum() / total
    asymmetry = abs(var_y - var_x) / (var_y + var_x + 1e-9)

    # 4. secondary peak: mask out central 5x5 (target PSF core), look for a second bright blob
    core_mask = (np.abs(yy - cy) <= 2) & (np.abs(xx - cx) <= 2)
    outside = med_img_bgsub.copy()
    outside[core_mask] = 0
    peak_core = med_img_bgsub[core_mask].max() if core_mask.any() else 1.0
    secondary_peak_frac = outside.max() / (peak_core + 1e-9)

    return dict(centroid_offset=centroid_offset, centroid_jitter=centroid_jitter,
                asymmetry=asymmetry, secondary_peak_frac=secondary_peak_frac,
                n_cadences=len(flux))

OUT_PATH = "eval/pixel_level_pilot_results.parquet"
results = []
n_ok, n_fail = 0, 0
for i, (tic, sec, resid, contratio, numcont) in enumerate(target_obs.itertuples(index=False)):
    try:
        sr = lk.search_targetpixelfile(f"TIC {int(tic)}", mission="TESS", author="TESS-SPOC", sector=int(sec))
        if len(sr) == 0:
            n_fail += 1
            continue
        tpf = sr[0].download(download_dir="tess_cache")
        if tpf is None:
            n_fail += 1
            continue
        feats = pixel_features(tpf)
        if feats is None:
            n_fail += 1
            continue
        feats.update(tic_id=tic, sector=sec, resid=resid, contratio=contratio, numcont=numcont)
        results.append(feats)
        n_ok += 1
    except Exception:
        n_fail += 1
    if (i+1) % 10 == 0:
        log(f"  {i+1}/{len(target_obs)} processed  ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")
        if results:
            pd.DataFrame(results).to_parquet(OUT_PATH)

log(f"\nDone. ok={n_ok} fail={n_fail}  T={int(time.time()-T0)}s")

if n_ok < 30:
    log("Too few successful downloads for a reliable check.")
    sys.exit(0)

res_df = pd.DataFrame(results)
res_df.to_parquet(OUT_PATH)
log(f"Saved -> {OUT_PATH} ({len(res_df)} rows)")

res_df["abs_resid"] = res_df["resid"].abs()
n = len(res_df)
se = 1/np.sqrt(n-3)
print(f"\n=== PURE RANDOM pixel-feature correlations (n={n}, approx SE={se:.4f}) ===")
feats = ["centroid_offset","centroid_jitter","asymmetry","secondary_peak_frac"]
for f in feats:
    r = res_df[f].corr(res_df["abs_resid"])
    sig = "SIGNIFICANT" if abs(r) > 1.96*se else "not significant at 95%"
    print(f"  r({f}, |resid|) = {r:+.4f}   [{sig}]")

print("\n=== Partial correlation: does pixel shape add anything BEYOND contratio/numcont? ===")
from numpy.linalg import lstsq
X = res_df[["contratio","numcont"]].values
X = np.column_stack([X, np.ones(len(X))])
for f in feats:
    y = res_df[f].values
    coef, _, _, _ = lstsq(X, y, rcond=None)
    resid_f = y - X @ coef
    y2 = res_df["abs_resid"].values
    coef2, _, _, _ = lstsq(X, y2, rcond=None)
    resid_abs = y2 - X @ coef2
    partial_r = np.corrcoef(resid_f, resid_abs)[0,1]
    sig = "SIGNIFICANT" if abs(partial_r) > 1.96*se else "not significant at 95%"
    print(f"  partial_r({f}, |resid| | contratio,numcont) = {partial_r:+.4f}   [{sig}]")
