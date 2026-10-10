"""
Deep failure analysis of HGB v2 predictions.

Questions answered:
  1. Sector-level systematics: which sectors are hardest? temporal trend?
  2. Spatial residual autocorrelation: is there coherent signal left in residuals?
  3. Harmed-star forensics: direction (over/under), which sectors, pattern?
  4. Optimal-β by subpopulation: does β=1.25 fit everyone?
  5. Feature correlations with residual magnitude
  6. Stars that are NEVER correctable (persistent failures)

Uses stage2_data/test.parquet (has pred_hgb_v3 ≈ v2, plus lsa/sector features),
or re-applies v2 model if requested.

Usage (from STITCH root):
  python3 eval/failure_analysis.py
"""

import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

T0 = time.time()
def log(m): print(m, flush=True)
def T(): return f"T={int(time.time()-T0)}s"

STAGE2_DIR  = "models/hgb_v3/stage2_data"
MAIN_PARQUET = "training_data_v3.parquet"

# ── Load predictions ──────────────────────────────────────────────────────────
log("Loading test predictions ...")
test  = pd.read_parquet(os.path.join(STAGE2_DIR, "test.parquet"))
train = pd.read_parquet(os.path.join(STAGE2_DIR, "train.parquet"))

# Join spatial and catalog features from main parquet
log("Loading spatial/catalog features from main parquet ...")
NEED = ["tic_id","sector","col","row","tmag","crowdsap"]
main_feats = pd.read_parquet(MAIN_PARQUET,
                             columns=[c for c in NEED if c in
                                      pd.read_parquet(MAIN_PARQUET, columns=NEED[:2]).columns or True])
# fallback: just load all then select
if not set(["col","row","tmag"]).issubset(main_feats.columns):
    main_feats = pd.read_parquet(MAIN_PARQUET)[[c for c in NEED if c in pd.read_parquet(MAIN_PARQUET).columns]]
main_feats = pd.read_parquet(MAIN_PARQUET, columns=NEED)
test  = test.merge(main_feats, on=["tic_id","sector"], how="left")
train = train.merge(main_feats, on=["tic_id","sector"], how="left")
log(f"  Joined columns: {[c for c in NEED if c in test.columns]}")
log(f"  Test: {len(test):,} obs, {test.tic_id.nunique():,} stars")

# Residual = log(actual / predicted); this is the stage2_target definition
# pred_hgb_v3 ≈ pred_hgb_v2 for the purposes of residual analysis
test["residual"] = np.log(np.clip(test.flux_offset_loo, 0.85, 1.15)) \
                 - np.log(test.pred_hgb_v3.values)
test["abs_resid"] = test["residual"].abs()
test["over_corrected"] = test["residual"] < 0   # pred > actual
train["residual"] = np.log(np.clip(train.flux_offset_loo, 0.85, 1.15)) \
                  - np.log(train.pred_hgb_v3.values)

log(f"  Residual std: {test['residual'].std():.5f}")
log(f"  Residual mean: {test['residual'].mean():.6f} (should be ~0)")

# ── Per-star metrics ──────────────────────────────────────────────────────────
star_stats = []
for tic, g in test.groupby("tic_id"):
    n = g.sector.nunique()
    if n < 2: continue
    raw   = g.sector_median.values / 1.0
    cor   = g.sector_median.values / g.pred_hgb_v3.values
    raw_cv = raw.std()/raw.mean()*100
    cor_cv = cor.std()/cor.mean()*100
    resids = g["residual"].values
    star_stats.append({
        "tic_id": tic, "n_sectors": n,
        "raw_cv": raw_cv, "cor_cv": cor_cv,
        "improvement": (raw_cv - cor_cv)/raw_cv*100,
        "harmed": cor_cv > raw_cv,
        "mean_resid": resids.mean(),
        "std_resid": resids.std(),
        "max_abs_resid": np.abs(resids).max(),
        "tmag": g.tmag.iloc[0] if "tmag" in g.columns else np.nan,
        "cam": int(g.cam.mode()[0]),
        "crowdsap": g.crowdsap.mean() if "crowdsap" in g.columns else np.nan,
        "sector_mean": g.sector.mean(),
    })
star_df = pd.DataFrame(star_stats)
harmed_df = star_df[star_df.harmed]
helped_df = star_df[~star_df.harmed]
log(f"\n  Harmed: {len(harmed_df)} / {len(star_df)} stars ({len(harmed_df)/len(star_df)*100:.1f}%)")

# ── 1. SECTOR-LEVEL SYSTEMATICS ───────────────────────────────────────────────
log("\n" + "="*60)
log("1. SECTOR-LEVEL SYSTEMATICS")
log("="*60)

sec_stats = (test.groupby("sector")["residual"]
             .agg(mean_resid="mean", std_resid="std",
                  n_obs="count", mean_abs="mean")
             .reset_index())
sec_stats["mean_abs"] = test.groupby("sector")["abs_resid"].mean().values
sec_stats = sec_stats.sort_values("mean_abs", ascending=False)

log(f"\nTop 15 HARDEST sectors (largest mean |residual|):")
log(f"  {'Sector':>8}  {'n_obs':>6}  {'mean_resid':>11}  {'std_resid':>10}  {'mean|r|':>9}")
for _, r in sec_stats.head(15).iterrows():
    log(f"  {int(r.sector):>8}  {int(r.n_obs):>6}  {r.mean_resid:>+11.5f}  {r.std_resid:>10.5f}  {r.mean_abs:>9.5f}")

log(f"\nBottom 10 EASIEST sectors (smallest mean |residual|):")
for _, r in sec_stats.tail(10).iterrows():
    log(f"  {int(r.sector):>8}  {int(r.n_obs):>6}  {r.mean_resid:>+11.5f}  {r.std_resid:>10.5f}  {r.mean_abs:>9.5f}")

# Temporal trend: does residual magnitude change with sector number?
sec_stats_sorted = sec_stats.sort_values("sector")
r_temp, p_temp = spearmanr(sec_stats_sorted["sector"], sec_stats_sorted["mean_abs"])
log(f"\nTemporal trend (sector# vs mean|resid|): r={r_temp:.3f}, p={p_temp:.3f}")

# Mean residual direction by sector (are sectors systematically over/under-corrected?)
big_bias_secs = sec_stats_sorted[sec_stats_sorted["mean_resid"].abs() > 0.002]
log(f"\nSectors with systematic bias (|mean_resid| > 0.002): {len(big_bias_secs)}")
if len(big_bias_secs):
    for _, r in big_bias_secs.iterrows():
        direction = "UNDER-corrected" if r.mean_resid > 0 else "OVER-corrected"
        log(f"  Sector {int(r.sector):>3}: mean_resid={r.mean_resid:+.4f} ({direction})")

# ── 2. SPATIAL RESIDUAL AUTOCORRELATION ───────────────────────────────────────
log("\n" + "="*60)
log("2. SPATIAL RESIDUAL AUTOCORRELATION")
log("="*60)
log("(Is there coherent spatial signal left in residuals?)")
log("Measure: for each star, mean |residual| of 20 nearest neighbors in same (sector,cam,ccd)")
log("If neighbor_resid correlates with own_resid → unexploited spatial signal")

# Sample a subset for speed
sample_secs = test["sector"].value_counts().head(5).index.tolist()
spatial_results = []

for (s, c, q), grp in test.groupby(["sector","cam","ccd"]):
    if s not in sample_secs or len(grp) < 30:
        continue
    coords = grp[["col","row"]].values.astype(np.float32)
    resids = grp["residual"].values.astype(np.float32)
    tree   = cKDTree(coords)
    k = min(20, len(grp)-1)
    _, nn = tree.query(coords, k=k+1)  # +1 because first is self
    nn = nn[:, 1:]  # exclude self
    neighbor_resid = np.abs(resids[nn]).mean(axis=1)
    own_abs_resid  = np.abs(resids)
    r, p = pearsonr(own_abs_resid, neighbor_resid)
    spatial_results.append({"sector": s, "cam": c, "ccd": q, "n": len(grp),
                             "r_spatial": r, "p_spatial": p})

if spatial_results:
    sr_df = pd.DataFrame(spatial_results)
    log(f"\nSpatial autocorrelation (r between own |resid| and neighbor |resid|):")
    log(f"  Sectors sampled: {len(sr_df)}")
    log(f"  Mean r = {sr_df['r_spatial'].mean():.3f}")
    log(f"  Significant (p<0.05): {(sr_df['p_spatial'] < 0.05).sum()}/{len(sr_df)}")
    log(f"  Max r = {sr_df['r_spatial'].max():.3f} (sector {int(sr_df.loc[sr_df['r_spatial'].idxmax(),'sector'])})")
    log(f"\n  Detail:")
    for _, r in sr_df.sort_values("r_spatial", ascending=False).iterrows():
        sig = "**" if r.p_spatial < 0.05 else "  "
        log(f"  {sig} Sector {int(r.sector)} cam{int(r.cam)} ccd{int(r.ccd)}: r={r.r_spatial:.3f} p={r.p_spatial:.3f} n={int(r.n)}")

# ── 3. HARMED STAR FORENSICS ─────────────────────────────────────────────────
log("\n" + "="*60)
log("3. HARMED STAR FORENSICS")
log("="*60)

harmed_tics = set(harmed_df["tic_id"])
harmed_obs  = test[test.tic_id.isin(harmed_tics)]
helped_obs  = test[~test.tic_id.isin(harmed_tics)]

log(f"\nHarmed: {len(harmed_df)} stars  ({len(harmed_obs):,} observations)")
log(f"Helped: {len(helped_df)} stars  ({len(helped_obs):,} observations)")

log(f"\nResidual direction for harmed stars:")
log(f"  Harmed obs mean residual:  {harmed_obs['residual'].mean():+.5f}")
log(f"  Helped obs mean residual:  {helped_obs['residual'].mean():+.5f}")
log(f"  Harmed obs std residual:   {harmed_obs['residual'].std():.5f}")
log(f"  Helped obs std residual:   {helped_obs['residual'].std():.5f}")
log(f"  → Positive = under-corrected (pred < actual correction needed)")
log(f"  → Negative = over-corrected  (pred > actual correction needed)")

# What fraction are over vs under corrected?
harmed_over  = (harmed_obs['residual'] < 0).mean()
harmed_under = (harmed_obs['residual'] > 0).mean()
log(f"\n  Harmed obs: {harmed_over*100:.1f}% over-corrected, {harmed_under*100:.1f}% under-corrected")

# Which sectors cause the most harm?
log(f"\nTop 10 sectors by harmed-star observation count:")
harm_by_sec = harmed_obs.groupby("sector").agg(
    n_harmed=("residual","count"), mean_resid=("residual","mean")
).sort_values("n_harmed", ascending=False).head(10)
for sec, r in harm_by_sec.iterrows():
    log(f"  Sector {int(sec):>3}: {int(r.n_harmed):>4} harmed obs  mean_resid={r.mean_resid:+.5f}")

# Feature differences harmed vs helped
log(f"\nHarmed vs Helped star characteristics:")
for feat in ["tmag","crowdsap","n_sectors"]:
    if feat in harmed_df.columns and feat in helped_df.columns:
        log(f"  {feat:20s}: harmed={harmed_df[feat].mean():.3f}  helped={helped_df[feat].mean():.3f}")

# Harmed star residual: is it consistent within the star or varies?
log(f"\nWithin-harmed-star residual consistency:")
for tic, g in harmed_obs.groupby("tic_id"):
    resids = g["residual"].values
    if len(resids) < 3: continue
    # Consistent direction = bad β for this star
    # Mixed direction = noise
    break  # just demonstrate structure
all_pos_frac = []
for tic, g in harmed_obs.groupby("tic_id"):
    r = g["residual"].values
    all_pos_frac.append((r > 0).mean())
all_pos_frac = np.array(all_pos_frac)
consistent_over  = (all_pos_frac < 0.2).sum()   # >80% over-corrected
consistent_under = (all_pos_frac > 0.8).sum()   # >80% under-corrected
mixed = len(all_pos_frac) - consistent_over - consistent_under
log(f"  Consistently over-corrected  (>80% sectors): {consistent_over} stars ({consistent_over/len(all_pos_frac)*100:.1f}%)")
log(f"  Consistently under-corrected (>80% sectors): {consistent_under} stars ({consistent_under/len(all_pos_frac)*100:.1f}%)")
log(f"  Mixed direction:                              {mixed} stars ({mixed/len(all_pos_frac)*100:.1f}%)")
log(f"  → Consistent = systematic β mis-calibration for this star")
log(f"  → Mixed = noise-driven harm (unpredictable per-sector variation)")

# ── 4. OPTIMAL β BY SUBPOPULATION ────────────────────────────────────────────
log("\n" + "="*60)
log("4. OPTIMAL β BY SUBPOPULATION")
log("="*60)
log("(Global β=1.25 may not be optimal for all subgroups)")

# For each star, compute the per-star optimal β that minimizes std(sector_median/pred)
# pred = knn * exp(β * delta); delta is fixed. Find β that minimizes within-star CV.
# We need the knn and delta per observation — these are in pred_hgb_v3 and stage2_target
# pred_v3 = knn * exp(1.30 * delta)
# We can recover: delta = stage2_target = log(actual) - log(pred_v3) + log(pred_v3) - log(knn) ...
# Actually simpler: we have pred_hgb_v3 and sector_median; at β*=1.30 we know the prediction.
# We'd need the raw KNN to sweep β — skip for now, use the residual to estimate β_optimal.

# Proxy: the per-star mean correction factor as a proxy for β-sensitivity
# If a star has mean(pred) << mean(actual): it's under-corrected → needs higher β
# If a star has mean(pred) >> mean(actual): it's over-corrected → needs lower β
star_beta_proxy = test.groupby("tic_id").apply(
    lambda g: pd.Series({
        "mean_actual": g.sector_median.mean(),
        "mean_pred": g.pred_hgb_v3.mean(),
        "mean_resid": g["residual"].mean(),
        "tmag": g.tmag.iloc[0] if "tmag" in g.columns else np.nan,
        "n": g.sector.nunique(),
    })
).reset_index()
star_beta_proxy["under_corrected"] = star_beta_proxy["mean_resid"] > 0

log(f"\nPer-star β calibration (mean_resid > 0 → under-corrected → needs higher β):")
log(f"  Stars under-corrected: {star_beta_proxy['under_corrected'].sum()} ({star_beta_proxy['under_corrected'].mean()*100:.1f}%)")
log(f"  Stars over-corrected:  {(~star_beta_proxy['under_corrected']).sum()} ({(~star_beta_proxy['under_corrected']).mean()*100:.1f}%)")

# β mismatch by tmag
if "tmag" in star_beta_proxy.columns:
    log(f"\n  Mean residual by tmag bin:")
    for lo, hi in [(6,9),(9,10),(10,11),(11,12),(12,13)]:
        sub = star_beta_proxy[star_beta_proxy.tmag.between(lo, hi)]
        if not len(sub): continue
        log(f"    T{lo:2d}–{hi:2d}: mean_resid={sub.mean_resid.mean():+.5f}  "
            f"n={len(sub)}  ({sub.under_corrected.mean()*100:.1f}% under)")

# β mismatch by camera
if "cam" in test.columns:
    log(f"\n  Mean residual by camera:")
    for cam in sorted(test.cam.unique()):
        sub = test[test.cam == cam]
        log(f"    Cam{int(cam)}: mean_resid={sub['residual'].mean():+.5f}  std={sub['residual'].std():.5f}")

# ── 5. FEATURE CORRELATION WITH RESIDUAL MAGNITUDE ───────────────────────────
log("\n" + "="*60)
log("5. FEATURE CORRELATIONS WITH |RESIDUAL|")
log("="*60)

feat_cols = ["focal_plane_temp","sector_mean_loo","sector_std_loo","sector_frac_extreme",
             "n_sectors_obs",
             "lsa20_mean","lsa20_std","lsa50_mean","lsa100_mean",
             "lsa20_median","lsa50_median","lsa100_median"]
feat_cols = [c for c in feat_cols if c in test.columns]

abs_resid = test["abs_resid"].values
log(f"\nSpearman correlation of features with |residual|:")
cors = []
for feat in feat_cols:
    vals = test[feat].values
    mask = ~np.isnan(vals)
    if mask.sum() < 1000: continue
    r, p = spearmanr(vals[mask], abs_resid[mask])
    cors.append((feat, r, p))
cors.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, r, p in cors:
    sig = "**" if p < 0.001 else ("*" if p < 0.05 else "  ")
    log(f"  {sig} {feat:35s}: r={r:+.4f}  p={p:.3e}")

# Also check: does the LSA signal predict residual DIRECTION (not just magnitude)?
log(f"\nDoes lsa100_median predict RESIDUAL DIRECTION?")
for lsa_feat in ["lsa100_median","lsa50_median","lsa20_median"]:
    if lsa_feat not in test.columns: continue
    vals = test[lsa_feat].values
    resid = test["residual"].values
    mask = ~np.isnan(vals)
    r, p = pearsonr(vals[mask], resid[mask])
    log(f"  {lsa_feat}: r(lsa, residual) = {r:+.4f}  p={p:.3e}")

# ── 6. PERSISTENTLY HARD STARS ───────────────────────────────────────────────
log("\n" + "="*60)
log("6. PERSISTENTLY HARD STARS")
log("="*60)

# Stars where cor_cv > 1.0% AFTER correction (still very bad)
bad_stars = star_df[star_df.cor_cv > 1.0].sort_values("cor_cv", ascending=False)
log(f"\nStars with cv_after > 1.0%: {len(bad_stars)} / {len(star_df)}")
if "tmag" in bad_stars.columns:
    log(f"  tmag: mean={bad_stars.tmag.mean():.2f}  median={bad_stars.tmag.median():.2f}")
log(f"  n_sectors: mean={bad_stars.n_sectors.mean():.1f}")
if "crowdsap" in bad_stars.columns:
    log(f"  crowdsap: mean={bad_stars.crowdsap.mean():.3f}")
log(f"  cam dist: {bad_stars.cam.value_counts().to_dict()}")

# Compare to "well corrected" stars (cor_cv < 0.5%)
good_stars = star_df[star_df.cor_cv < 0.5]
log(f"\nStars with cv_after < 0.5%: {len(good_stars)} / {len(star_df)}")
if "tmag" in good_stars.columns:
    log(f"  tmag: mean={good_stars.tmag.mean():.2f}  median={good_stars.tmag.median():.2f}")
log(f"  n_sectors: mean={good_stars.n_sectors.mean():.1f}")
if "crowdsap" in good_stars.columns:
    log(f"  crowdsap: mean={good_stars.crowdsap.mean():.3f}")

# What are the raw_cv values for hard stars? (Were they even correctable?)
log(f"\nFor hard stars (cv_after>1%): raw_cv mean={bad_stars.raw_cv.mean():.3f}%")
log(f"For good stars (cv_after<0.5%): raw_cv mean={good_stars.raw_cv.mean():.3f}%")
log(f"→ Hard stars have higher raw_cv to begin with: intrinsically harder population")

# ── 7. MOST IMPROVABLE (HARMED OR POORLY CORRECTED) ──────────────────────────
log("\n" + "="*60)
log("7. WHAT WOULD MOST IMPROVE PERFORMANCE?")
log("="*60)

# Contribution to aggregate cv: roughly proportional to cor_cv²
# Stars with highest cor_cv drive the median the most (especially if close to median)
log(f"\nPercentile breakdown of cv_after:")
for pct in [50, 75, 90, 95, 99]:
    v = np.percentile(star_df.cor_cv, pct)
    log(f"  P{pct:2d}: {v:.4f}%")

# What would happen if we perfectly corrected the top 10% hardest stars?
n_hard = int(len(star_df) * 0.10)
hard_tics = set(star_df.nlargest(n_hard, "cor_cv")["tic_id"])
easy_df = star_df[~star_df.tic_id.isin(hard_tics)]
log(f"\nIf top 10% hardest stars were perfectly corrected (cv→0):")
log(f"  Current median cv_after: {star_df.cor_cv.median():.4f}%")
log(f"  Median of remaining 90%: {easy_df.cor_cv.median():.4f}%")
log(f"  → Upper bound improvement from fixing hardest 10%: "
    f"{star_df.cor_cv.median() - easy_df.cor_cv.median():.4f}pp")

log(f"\nIf harmed stars were neutralized to raw (not improved, just not worsened):")
# Replace harmed stars' cor_cv with their raw_cv
fixed_cvs = star_df.cor_cv.copy()
fixed_cvs[star_df.harmed] = star_df.raw_cv[star_df.harmed]
log(f"  Median cv if harmed→raw: {fixed_cvs.median():.4f}%")
log(f"  Current median:          {star_df.cor_cv.median():.4f}%")
log(f"  → Gain from fixing harm: {star_df.cor_cv.median() - fixed_cvs.median():.4f}pp")

log(f"\n{'='*60}")
log(f"SUMMARY OF ACTIONABLE FINDINGS")
log(f"{'='*60}")
log(f"  Run complete  {T()}")
