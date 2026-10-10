"""Leakage-safe latent detector-regime experiment.

The scientifically tempting star x sector factorization is not deployable to
unseen stars: their row factors are undefined unless their measured fluxes are
used at inference.  This experiment therefore learns modes of the residual
sector x (camera, CCD) calibration matrix from training stars only.  Validation
selects rank and shrinkage; test-star labels never define a mode or feature.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

from pairwise_global_calibration import BASE_COLS, FEATURE_COLS, paired_bootstrap
from pairwise_global_robust import fit_robust_global, group_median
from residual_multiscale_experiment import split_stars, spatial_features, scatter_table


SEED = 20260916
OUT = Path("eval/latent_mode_results")
REF_LOWER, REF_UPPER = 0.5, 1.5


def build_regime_modes(frame, residual, ranks=(0, 1, 2, 4, 8), ridge_count=100):
    """Return deployable sector-detector tables based only on training rows.

    Rank zero is an additive sector + detector model. Higher ranks reconstruct
    the remaining interaction matrix by SVD. Cell estimates are shrunk by the
    number of contributing stars before decomposition.
    """
    z = frame[["sector", "cam", "ccd", "tic_id"]].copy()
    z["detector"] = (z.cam.astype(int)-1)*4 + (z.ccd.astype(int)-1)
    z["residual"] = residual
    cell = z.groupby(["sector", "detector"]).residual.agg(["median", "count"])
    sectors = np.sort(z.sector.unique())
    detectors = np.arange(16)
    value = cell["median"].unstack().reindex(index=sectors, columns=detectors)
    count = cell["count"].unstack().reindex(index=sectors, columns=detectors).fillna(0)

    # Robust additive effects, iterated on observed cell medians.
    sec = value.median(axis=1, skipna=True).fillna(0)
    det = pd.Series(0.0, index=detectors)
    for _ in range(8):
        det = value.sub(sec, axis=0).median(axis=0, skipna=True).fillna(0)
        sec = value.sub(det, axis=1).median(axis=1, skipna=True).fillna(0)
    additive = pd.DataFrame(sec.to_numpy()[:, None] + det.to_numpy()[None, :],
                            index=sectors, columns=detectors)
    interaction = value - additive
    shrink = count / (count + ridge_count)
    dense = interaction.fillna(0).to_numpy() * shrink.to_numpy()
    u, s, vt = np.linalg.svd(dense, full_matrices=False)
    tables = {}
    for rank in ranks:
        recon = np.zeros_like(dense) if rank == 0 else (u[:, :rank]*s[:rank]) @ vt[:rank]
        tables[rank] = additive + pd.DataFrame(recon, index=sectors, columns=detectors)
    diagnostics = {
        "n_sectors": int(len(sectors)), "n_detectors": 16,
        "observed_cells": int(value.notna().sum().sum()),
        "cell_fraction_observed": float(value.notna().mean().mean()),
        "median_training_stars_per_cell": float(count.where(count.gt(0)).stack().median()),
        "singular_values": [float(x) for x in s[:12]],
        "variance_fraction_first_modes": [float(np.sum(s[:k]**2)/np.sum(s**2))
                                           for k in (1, 2, 4, 8)],
    }
    return tables, diagnostics


def lookup_mode(frame, table):
    det = (frame.cam.astype(int)-1)*4 + (frame.ccd.astype(int)-1)
    row = table.index.get_indexer(frame.sector)
    col = table.columns.get_indexer(det)
    out = np.zeros(len(frame), dtype=float)
    good = (row >= 0) & (col >= 0)
    out[good] = table.to_numpy()[row[good], col[good]]
    return out, int((~good).sum())


def tune_mode(frame, log_base, model_residual, mode):
    best = None
    # Preserve the anchor model's validation-selected strength (1.30 in the
    # robust experiment) while validation selects only the new mode strength.
    beta = 1.30
    for gamma in np.linspace(-1.0, 2.0, 61):
        pred = np.exp(log_base + beta*model_residual + gamma*mode)
        score = float(scatter_table(frame, {"candidate": pred}).candidate.median())
        if best is None or score < best[2]:
            best = (float(beta), float(gamma), score)
    return best


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    df = pd.read_parquet("training_data_v3.parquet")
    finite = (np.isfinite(df.sector_median) & df.sector_median.gt(0) &
              np.isfinite(df.flux_offset_loo) & df.n_sectors_total.ge(4) &
              df.tmag.le(13) & df.col.notna() & df.row.notna())
    df = df[finite].copy().reset_index(drop=True)
    train_ids, val_ids, test_ids = split_stars(df)
    masks = {"train": df.tic_id.isin(train_ids), "val": df.tic_id.isin(val_ids),
             "test": df.tic_id.isin(test_ids)}
    reference = df[masks["train"]].copy()
    reference["flux_offset_loo"] = reference.flux_offset_loo.clip(REF_LOWER, REF_UPPER)
    frames, feats, X, unsupported = {}, {}, {}, {}
    for name, mask in masks.items():
        frames[name] = df[mask].reset_index(drop=True)
        print("spatial features", name, len(frames[name]), flush=True)
        feats[name] = spatial_features(reference, df[mask])
        X[name] = pd.concat([frames[name][BASE_COLS], feats[name]], axis=1)[FEATURE_COLS]
        good = np.isfinite(feats[name].knn20_median.to_numpy())
        unsupported[name] = int((~good).sum())
        frames[name] = frames[name].loc[good].reset_index(drop=True)
        feats[name] = feats[name].loc[good].reset_index(drop=True)
        X[name] = X[name].loc[good].reset_index(drop=True)

    base = {k: feats[k].knn20_median.to_numpy() for k in frames}
    model, history = fit_robust_global(
        X["train"], np.log(frames["train"].sector_median.to_numpy()),
        np.log(base["train"]), frames["train"].tic_id.to_numpy())
    model_resid = {k: model.predict(X[k]) for k in frames}

    # Residual after the shared model, with each training star's unknown
    # luminosity removed. No validation/test observation enters this matrix.
    response = np.log(frames["train"].sector_median.to_numpy()) - np.log(base["train"])
    nuisance = group_median(response - model_resid["train"], frames["train"].tic_id.to_numpy())
    train_centered = response - model_resid["train"] - nuisance
    tables, mode_diag = build_regime_modes(frames["train"], train_centered)

    # Quantify why ordinary star x sector matrix completion is non-deployable.
    nstars = frames["train"].tic_id.nunique(); nsectors = frames["train"].sector.nunique()
    star_sector = frames["train"][["tic_id", "sector"]].drop_duplicates()
    overlap_diag = {
        "train_star_sector_matrix_shape": [int(nstars), int(nsectors)],
        "observed_entries": int(len(star_sector)),
        "density_pct": float(100*len(star_sector)/(nstars*nsectors)),
        "median_sectors_per_training_star": float(star_sector.groupby("tic_id").size().median()),
        "test_stars_with_learnable_training_row_factor": 0,
        "reason": "TIC-disjoint split: every test star is a cold-start row",
    }

    candidates = {}; tuning = {}; missing = {}
    logbase = {k: np.log(base[k]) for k in frames}
    anchor_val = np.exp(logbase["val"] + 1.30*model_resid["val"])
    anchor_test = np.exp(logbase["test"] + 1.30*model_resid["test"])
    for rank, table in tables.items():
        mv, missv = lookup_mode(frames["val"], table)
        mt, misst = lookup_mode(frames["test"], table)
        beta, gamma, score = tune_mode(frames["val"], logbase["val"], model_resid["val"], mv)
        name = f"regime_rank{rank}"
        candidates[name] = np.exp(logbase["test"] + beta*model_resid["test"] + gamma*mt)
        tuning[name] = {"beta": beta, "gamma": gamma, "validation_scatter_pct": 100*score}
        missing[name] = {"validation_rows": missv, "test_rows": misst}

    per = scatter_table(frames["test"], {"raw": np.ones(len(frames["test"])),
        "median_knn": base["test"], "robust_global_anchor": anchor_test, **candidates})
    summary = {c: {"median_scatter_pct": float(100*per[c].median()),
        "reduction_from_raw_pct": float(100*(1-per[c].median()/per.raw.median())),
        "stars_worse_than_raw_pct": float(100*(per[c]>per.raw).mean())} for c in per}
    best_name = min(tuning, key=lambda x: tuning[x]["validation_scatter_pct"])
    comparison = {
        "selected_on_validation": best_name,
        "gain_vs_anchor_pct": float(100*(1-per[best_name].median()/per.robust_global_anchor.median())),
        "gain_vs_anchor_bootstrap95": paired_bootstrap(per, best_name, "robust_global_anchor"),
        "better_star_fraction_pct": float(100*(per[best_name] < per.robust_global_anchor).mean()),
    }
    payload = {"method": "training-only low-rank sector x camera/CCD residual modes",
      "selection": "untruncated positive finite sector flux; n_sectors>=4; tmag<=13",
      "leakage_policy": "modes use training-star residuals only; TIC-disjoint validation/test; no query-star observations are predictors",
      "rows_after_support": {k: len(v) for k,v in frames.items()},
      "stars_after_support": {k: int(v.tic_id.nunique()) for k,v in frames.items()},
      "unsupported_rows_removed": unsupported, "star_sector_identifiability": overlap_diag,
      "regime_matrix": mode_diag, "anchor_history": history,
      "tuning": tuning, "unseen_regime_rows": missing, "test": summary,
      "comparison": comparison, "common_test_stars": int(len(per)),
      "elapsed_seconds": time.time()-t0}
    (OUT/"results.json").write_text(json.dumps(payload, indent=2))
    per.to_csv(OUT/"per_star_scatter.csv")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
