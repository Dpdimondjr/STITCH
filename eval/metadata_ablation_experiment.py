"""Leakage-safe metadata ablations for the STITCH v3 observation table.

All models predict the residual around a K=20 median made exclusively from
training-star labels. Training queries exclude their own TIC. Model shrinkage
is selected on validation stars and applied once to held-out test stars.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

SEED = 20260916
KEYS = ["sector", "cam", "ccd"]
OUT = Path("eval/metadata_ablation_results")

CORE = ["col", "row", "delta_sub_col", "delta_sub_row", "sector", "cam",
        "ccd", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
        "pdc_noi", "pr_wght2", "gaiarp"]
SPATIAL = ["knn20_median", "knn20_mad", "knn20_radius", "reference_count"]
TEMP = ["focal_plane_temp", "focal_plane_temp_std"]
PARTIAL = ["median_sap_bkg", "p90_sap_bkg", "bkg_rms",
           "scatter_flag_frac", "flfrcsap", "teff", "pdc_corp", "pdc_totp"]
PARTIAL_GROUPS = {
    "background": ["median_sap_bkg", "p90_sap_bkg", "bkg_rms", "scatter_flag_frac"],
    "aperture": ["flfrcsap"],
    "stellar_teff": ["teff"],
    "pdc_diagnostics": ["pdc_corp", "pdc_totp"],
}
ROBUST = ["cdpp1_0", "pdcvar", "jitter_rms", "pdc_noi", "pr_wght2"]


def split_stars(df):
    stars = (df.groupby("tic_id").cam.agg(lambda x: x.mode().iloc[0])
             .rename("cam").reset_index())
    train, temp = train_test_split(stars, test_size=.2, stratify=stars.cam,
                                   random_state=42)
    val, test = train_test_split(temp, test_size=.5, stratify=temp.cam,
                                 random_state=42)
    return tuple(set(x.tic_id) for x in (train, val, test))


def spatial_features(reference, query):
    """Return robust K=20 summaries; never use the query star's labels."""
    query = query.reset_index(drop=True)
    cols = SPATIAL
    output = np.full((len(query), len(cols)), np.nan, dtype=np.float32)
    output[:, -1] = 0
    groups = dict(tuple(reference.groupby(KEYS, sort=False)))
    for key, qg in query.groupby(KEYS, sort=False):
        ref = groups.get(key)
        if ref is None or len(ref) < 2:
            continue
        coords = ref[["col", "row"]].to_numpy(np.float32)
        ids = ref.tic_id.to_numpy()
        vals = ref.flux_offset_loo.to_numpy(float)
        extra = int(ref.tic_id.value_counts().max())
        take = min(20 + extra, len(ref))
        dist, ind = cKDTree(coords).query(qg[["col", "row"]].to_numpy(np.float32), k=take)
        if take == 1:
            dist, ind = dist[:, None], ind[:, None]
        for pos, qid, nn, dd in zip(qg.index, qg.tic_id, ind, dist):
            keep = ids[nn] != qid
            nn, dd = nn[keep][:20], dd[keep][:20]
            output[pos, 3] = len(nn)
            if len(nn) == 20:
                x = vals[nn]
                med = np.median(x)
                output[pos, :3] = med, np.median(np.abs(x-med)), dd[-1]
    return pd.DataFrame(output, columns=cols)


def robustify(train, *others):
    """Training-derived clipping/log transforms plus explicit missing flags."""
    frames = [x.copy() for x in (train,) + others]
    for col in ROBUST:
        finite = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        lo, hi = finite.quantile([.005, .995])
        # Signed log is monotonic and defined for anomalous negative values.
        for f in frames:
            z = pd.to_numeric(f[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            f[col] = np.sign(z.clip(lo, hi)) * np.log1p(np.abs(z.clip(lo, hi)))
            f[col + "_missing"] = z.isna().astype(np.int8)
    return frames


def scatter_table(frame, predictions):
    out = []
    sizes = frame.groupby("tic_id").size()
    for name, offset in predictions.items():
        x = frame[["tic_id", "sector_median"]].copy()
        x["corrected"] = x.sector_median / np.clip(offset, .85, 1.15)
        s = x.groupby("tic_id").corrected.agg(["mean", "std"])
        out.append((s["std"] / s["mean"]).rename(name))
    per = pd.concat(out, axis=1).dropna()
    return per[sizes.reindex(per.index).ge(5)]


def tune_alpha(frame, pred):
    best = None
    for alpha in np.linspace(0, 1.25, 26):
        score = scatter_table(frame, {"x": 1 + alpha*(pred-1)}).x.median()
        if best is None or score < best[1]:
            best = float(alpha), float(score)
    return best


def fit_predict(train, val, test, ftr, fv, ft, cols, tag):
    base_tr = ftr.knn20_median.to_numpy()
    good = np.isfinite(base_tr) & np.isfinite(train.flux_offset_loo)
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=.06, max_iter=250,
        max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=.1,
        random_state=SEED, early_stopping=False)
    print("fit", tag, int(good.sum()), len(cols), flush=True)
    model.fit(pd.concat([train.reset_index(drop=True), ftr], axis=1).loc[good, cols],
              train.flux_offset_loo.to_numpy()[good] - base_tr[good])
    pv = fv.knn20_median.to_numpy() + model.predict(
        pd.concat([val.reset_index(drop=True), fv], axis=1)[cols])
    pt = ft.knn20_median.to_numpy() + model.predict(
        pd.concat([test.reset_index(drop=True), ft], axis=1)[cols])
    alpha, vscore = tune_alpha(val, pv)
    return 1 + alpha*(pt-1), {"alpha": alpha, "validation_scatter_pct": 100*vscore}


def paired_bootstrap(per, candidate, baseline, n=2000):
    rng = np.random.default_rng(SEED)
    x, y = per[baseline].to_numpy(), per[candidate].to_numpy()
    gains = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(per), len(per))
        gains[i] = 100*(1 - np.median(y[ix])/np.median(x[ix]))
    return {"gain_pct": float(100*(1-np.median(y)/np.median(x))),
            "ci95": [float(q) for q in np.quantile(gains, [.025, .975])],
            "fraction_stars_improved_pct": float(100*np.mean(y < x))}


def evaluate(frame, predictions):
    per = scatter_table(frame, {"raw": np.ones(len(frame)), **predictions})
    summary = {c: {"median_scatter_pct": float(100*per[c].median()),
                   "reduction_from_raw_pct": float(100*(1-per[c].median()/per.raw.median())),
                   "stars_worse_than_raw_pct": float(100*np.mean(per[c] > per.raw))}
               for c in per}
    return per, summary


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    df = pd.read_parquet("training_data_v3.parquet")
    df = df.dropna(subset=["col", "row", "flux_offset_loo", "sector_median"])
    # Deliberately retain the target tails; absolute-error models are robust.
    df = df[df.n_sectors_total.ge(4) & df.tmag.le(13) &
            df.flux_offset_loo.gt(0)].copy().reset_index(drop=True)
    train_ids, val_ids, test_ids = split_stars(df)
    masks = {"train": df.tic_id.isin(train_ids), "val": df.tic_id.isin(val_ids),
             "test": df.tic_id.isin(test_ids)}
    frames = {k: df[v].reset_index(drop=True) for k, v in masks.items()}
    cache = OUT / "spatial_features.parquet"
    if cache.exists():
        sf = pd.read_parquet(cache)
        features = {k: sf[sf.split.eq(k)].drop(columns="split").reset_index(drop=True)
                    for k in frames}
    else:
        ref = frames["train"]
        features = {}
        blocks = []
        for k, x in frames.items():
            print("spatial", k, len(x), flush=True)
            features[k] = spatial_features(ref, x)
            blocks.append(features[k].assign(split=k))
        pd.concat(blocks, ignore_index=True).to_parquet(cache, index=False)

    tr, va, te = frames["train"], frames["val"], frames["test"]
    ftr, fv, ft = features["train"], features["val"], features["test"]
    core_cols = CORE + SPATIAL
    predictions = {"median_knn": ft.knn20_median.to_numpy()}
    validation = {}
    base_pred, validation["baseline"] = fit_predict(tr, va, te, ftr, fv, ft, core_cols, "baseline")
    predictions["baseline"] = base_pred

    # Robust transforms and missing indicators, thresholds learned on train.
    rtr, rv, rt = robustify(tr, va, te)
    robust_cols = core_cols + [c + "_missing" for c in ROBUST]
    predictions["robust_missing"] , validation["robust_missing"] = fit_predict(
        rtr, rv, rt, ftr, fv, ft, robust_cols, "robust_missing")

    predictions["temperature"], validation["temperature"] = fit_predict(
        tr, va, te, ftr, fv, ft, core_cols + TEMP, "temperature")
    predictions["partial_missing_aware"], validation["partial_missing_aware"] = fit_predict(
        tr, va, te, ftr, fv, ft, core_cols + PARTIAL + [c+"_missing" for c in PARTIAL],
        "partial_missing_aware") if False else (None, None)
    # Explicit flags (HGB also has native NaN routing).
    trm, vam, tem = tr.copy(), va.copy(), te.copy()
    for c in PARTIAL:
        for x in (trm, vam, tem): x[c+"_missing"] = x[c].isna().astype(np.int8)
    predictions["partial_missing_aware"], validation["partial_missing_aware"] = fit_predict(
        trm, vam, tem, ftr, fv, ft, core_cols + PARTIAL + [c+"_missing" for c in PARTIAL],
        "partial_missing_aware")

    # Identical complete-case train/val/test populations for partial metadata.
    complete = {k: frames[k][frames[k][PARTIAL].notna().all(axis=1)].reset_index(drop=True)
                for k in frames}
    cf = {k: features[k].loc[frames[k][PARTIAL].notna().all(axis=1).to_numpy()].reset_index(drop=True)
          for k in frames}
    cpred = {"median_knn": cf["test"].knn20_median.to_numpy()}
    complete_models = [("complete_baseline", core_cols),
                       ("complete_partial", core_cols + PARTIAL)]
    complete_models += [("complete_"+name, core_cols+cols)
                        for name, cols in PARTIAL_GROUPS.items()]
    for tag, cols in complete_models:
        cpred[tag], validation[tag] = fit_predict(
            complete["train"], complete["val"], complete["test"],
            cf["train"], cf["val"], cf["test"], cols, tag)
    cper, csummary = evaluate(complete["test"], cpred)
    # A gated deployment avoids forcing one tree ensemble to represent two
    # measurement regimes. Use the complete-case model where all fields exist,
    # and the all-population baseline otherwise. Each component's shrinkage was
    # independently fixed using validation stars above.
    gated = base_pred.copy()
    complete_test_mask = te[PARTIAL].notna().all(axis=1).to_numpy()
    gated[complete_test_mask] = cpred["complete_partial"]
    predictions["partial_gated"] = gated
    per, summary = evaluate(te, predictions)
    comparisons = {
        "robust_missing_vs_baseline": paired_bootstrap(per, "robust_missing", "baseline"),
        "temperature_vs_baseline": paired_bootstrap(per, "temperature", "baseline"),
        "partial_missing_aware_vs_baseline": paired_bootstrap(per, "partial_missing_aware", "baseline"),
        "partial_gated_vs_baseline": paired_bootstrap(per, "partial_gated", "baseline"),
        "complete_partial_vs_complete_baseline": paired_bootstrap(
            cper, "complete_partial", "complete_baseline")}
    for name in PARTIAL_GROUPS:
        comparisons["complete_"+name+"_vs_complete_baseline"] = paired_bootstrap(
            cper, "complete_"+name, "complete_baseline")

    payload = {"method": {"target_clipping": False, "loss": "absolute_error",
               "reference_policy": "training stars only; own TIC excluded",
               "shrinkage": "validation-selected only", "seed": SEED},
               "rows": {k: len(v) for k,v in frames.items()},
               "stars": {k: int(v.tic_id.nunique()) for k,v in frames.items()},
               "complete_case_rows": {k: len(v) for k,v in complete.items()},
               "coverage_pct": {c: float(100*df[c].notna().mean()) for c in TEMP+PARTIAL},
               "validation": validation, "test": summary,
               "complete_case_test": csummary, "comparisons": comparisons,
               "elapsed_seconds": time.time()-t0}
    (OUT/"results.json").write_text(json.dumps(payload, indent=2))
    per.to_csv(OUT/"per_star_scatter.csv")
    cper.to_csv(OUT/"complete_case_per_star_scatter.csv")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
