#!/usr/bin/env python3
"""Cache-native, star-disjoint SPOC feature cohort experiment.

PDCSAP is used only to construct the response and evaluation statistic. Predictors are
instrument/header, aperture, pointing, background/quality, and provenance quantities.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from sklearn.ensemble import HistGradientBoostingRegressor

from extract_spoc_trial import aperture_features, robust_stats

NAME_RE = re.compile(r"_(\d+)-s(\d+)_tess_v\d+_lc\.fits$")

CORE = ["sector", "camera", "ccd", "tmag", "crowdsap", "mom_col", "mom_row",
        "subpixel_col", "subpixel_row"]
APERTURE = ["flfrcsap", "ap_npix", "ap_width", "ap_height", "ap_perimeter",
            "ap_compactness", "target_to_ap_centroid"]
POINTING = ["pos_corr1_rms", "pos_corr2_rms", "pos_corr1_span", "pos_corr2_span",
            "pos_corr_major_rms", "pos_corr_minor_rms", "pos_corr_corr"]
BACKGROUND = ["log_sap_bkg_median", "log_sap_bkg_mad", "quality_nonzero_frac",
              "finite_pdcsap_frac", "n_good"]
PROVENANCE = ["data_rel", "cadence_days", "num_frames", "proc_major", "proc_minor",
              "proc_patch", "proc_date", "pdc_msmap"]
GROUPS = {
    "core": CORE,
    "core+aperture": CORE + APERTURE,
    "core+aperture+pointing": CORE + APERTURE + POINTING,
    "core+aperture+pointing+background": CORE + APERTURE + POINTING + BACKGROUND,
    "all_safe": CORE + APERTURE + POINTING + BACKGROUND + PROVENANCE,
}


def stable_u01(value: int, salt: str) -> float:
    raw = hashlib.sha256(f"{salt}:{value}".encode()).digest()[:8]
    return int.from_bytes(raw, "big") / 2**64


def proc_parts(value) -> tuple[float, float, float, float]:
    m = re.search(r"spoc-(\d+)\.(\d+)\.(\d+)-(\d{8})", str(value))
    return tuple(map(float, m.groups())) if m else (np.nan,) * 4


def extract_one(item: tuple[str, int, int]) -> dict:
    path, tic, sector = item
    out = {"path": path, "tic_id": tic, "sector": sector, "error": ""}
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as h:
            p, tab, ahdu = h[0].header, h[1], h[2]
            th, d = tab.header, tab.data
            flux = np.asarray(d["PDCSAP_FLUX"], float)
            q = np.asarray(d["QUALITY"], np.int64)
            finite = np.isfinite(flux) & (flux > 0)
            good = finite & (q == 0)
            use = good if good.sum() >= 50 else finite
            if use.sum() < 50:
                raise ValueError("fewer than 50 finite PDCSAP cadences")
            out.update(pdcsap_median=float(np.median(flux[use])), n_cadences=len(flux),
                       n_good=int(good.sum()), finite_pdcsap_frac=float(finite.mean()),
                       quality_nonzero_frac=float((q != 0).mean()),
                       camera=p.get("CAMERA"), ccd=p.get("CCD"), tmag=p.get("TESSMAG"),
                       crowdsap=th.get("CROWDSAP"), flfrcsap=th.get("FLFRCSAP"),
                       data_rel=p.get("DATA_REL"), cadence_days=th.get("TIMEDEL"),
                       num_frames=th.get("NUM_FRM"), pdc_msmap=float(th.get("PDCMETHD") == "msMAP"))
            a, b, c, dt = proc_parts(p.get("PROCVER"))
            out.update(proc_major=a, proc_minor=b, proc_patch=c, proc_date=dt)
            for col, name in (("SAP_BKG", "sap_bkg"), ("POS_CORR1", "pos_corr1"),
                              ("POS_CORR2", "pos_corr2")):
                out.update(robust_stats(np.asarray(d[col], float)[use], name))
            x = np.asarray(d["POS_CORR1"], float)[use]
            y = np.asarray(d["POS_CORR2"], float)[use]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                cov = np.cov(x[valid], y[valid], ddof=0)
                eig = np.maximum(np.linalg.eigvalsh(cov), 0)
                out.update(pos_corr_major_rms=float(np.sqrt(eig[-1])),
                           pos_corr_minor_rms=float(np.sqrt(eig[0])),
                           pos_corr_corr=float(np.corrcoef(x[valid], y[valid])[0, 1]))
            out.update(pos_corr1_span=out["pos_corr1_p95"] - out["pos_corr1_p05"],
                       pos_corr2_span=out["pos_corr2_p95"] - out["pos_corr2_p05"])
            out.update(aperture_features(ahdu.data))
            mc = np.asarray(d["MOM_CENTR1"], float)[use]
            mr = np.asarray(d["MOM_CENTR2"], float)[use]
            out["mom_col"] = float(np.nanmedian(mc)); out["mom_row"] = float(np.nanmedian(mr))
            out["subpixel_col"] = out["mom_col"] % 1; out["subpixel_row"] = out["mom_row"] % 1
            col0, row0 = ahdu.header.get("CRVAL1P"), ahdu.header.get("CRVAL2P")
            out["target_to_ap_centroid"] = float(np.hypot(out["mom_col"] - col0 - out["ap_centroid_x"],
                                                           out["mom_row"] - row0 - out["ap_centroid_y"]))
            out["log_sap_bkg_median"] = float(np.log1p(max(out["sap_bkg_median"], 0)))
            out["log_sap_bkg_mad"] = float(np.log1p(max(out["sap_bkg_mad"], 0)))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def prepare(frame: pd.DataFrame, columns: list[str], train_tics: set[int]):
    x = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
    tr = frame.tic_id.isin(train_tics)
    meds = x.loc[tr].median()
    missing = x.isna().astype(float).add_suffix("__missing")
    x = x.fillna(meds).fillna(0.0)
    # Preserve missingness only where it occurs; constant flags add no information.
    missing = missing.loc[:, missing.loc[tr].nunique().gt(1)]
    return pd.concat([x, missing], axis=1), meds.to_dict()


def metric(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    raw, cor = star_metric(frame, prediction)
    return {"n_stars": int(len(raw)), "raw_cv_pct": float(100 * raw.median()),
            "corrected_cv_pct": float(100 * cor.median()),
            "reduction_pct": float(100 * (1 - cor.median() / raw.median())),
            "stars_worse_pct": float(100 * (cor > raw).mean())}


def star_metric(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[pd.Series, pd.Series]:
    z = frame[["tic_id", "pdcsap_median"]].copy()
    z["pred"] = prediction
    z["pred"] -= z.groupby("tic_id")["pred"].transform("mean")
    z["corrected"] = z.pdcsap_median / np.exp(z.pred)
    raw = z.groupby("tic_id").pdcsap_median.agg(lambda v: np.std(v, ddof=0) / np.mean(v))
    cor = z.groupby("tic_id").corrected.agg(lambda v: np.std(v, ddof=0) / np.mean(v))
    return raw, cor


def bootstrap_gain(a: pd.Series, b: pd.Series, seed: int = 931, draws: int = 4000) -> dict:
    """Relative residual-scatter gain of b over a, bootstrapped by held-out star."""
    common = a.index.intersection(b.index); av, bv = a[common].to_numpy(), b[common].to_numpy()
    rng = np.random.default_rng(seed); values = np.empty(draws)
    for i in range(draws):
        ix = rng.integers(0, len(common), len(common))
        values[i] = 100 * (1 - np.median(bv[ix]) / np.median(av[ix]))
    return {"gain_pct": float(100 * (1 - np.median(bv) / np.median(av))),
            "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])]}


class ScoreNet(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n, 64), torch.nn.SiLU(),
                                       torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def pairs(indices: np.ndarray, tic: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); aa, bb = [], []
    for star in np.unique(tic[indices]):
        ix = indices[tic[indices] == star]
        pp = np.asarray(list(itertools.combinations(ix, 2)), dtype=int)
        if len(pp) > cap: pp = pp[rng.choice(len(pp), cap, replace=False)]
        if len(pp): aa.extend(pp[:, 0]); bb.extend(pp[:, 1])
    return np.asarray(aa), np.asarray(bb)


def pairwise_fit_predict(x: np.ndarray, y: np.ndarray, tic: np.ndarray, split: np.ndarray,
                         seed: int = 42) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed); torch.set_num_threads(4)
    trix, vaix = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    mean, std = x[trix].mean(0), x[trix].std(0); std[std < 1e-8] = 1
    xs = ((x - mean) / std).astype("float32"); ys = y.astype("float32")
    ta, tb = pairs(trix, tic, 40, seed); va, vb = pairs(vaix, tic, 40, seed + 1)
    model = ScoreNet(x.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tx = torch.from_numpy(xs); ty = torch.from_numpy(ys)
    best, best_state, stale = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(120):
        order = rng.permutation(len(ta)); model.train()
        for start in range(0, len(order), 2048):
            k = order[start:start + 2048]; pred = model(tx[ta[k]]) - model(tx[tb[k]])
            loss = torch.nn.functional.huber_loss(pred, ty[ta[k]] - ty[tb[k]], delta=0.01)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(tx[va]) - model(tx[vb]); score = torch.mean(torch.abs(vp - (ty[va] - ty[vb]))).item()
        if score < best - 1e-6:
            best, stale = score, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= 15: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred = model(tx).numpy()
    return pred, {"best_val_pair_mae": best, "epochs": epoch + 1, "train_pairs": len(ta), "val_pairs": len(va)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="tess_cache/mastDownload")
    ap.add_argument("--output", default="eval/spoc_cohort_results")
    ap.add_argument("--n-tics", type=int, default=1200)
    ap.add_argument("--min-cached-sectors", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--extract-only", action="store_true",
                    help="Build the cache-native feature table without fitting models")
    args = ap.parse_args(); outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    feature_path = outdir / "cohort_features.parquet"
    if args.reuse and feature_path.exists():
        frame = pd.read_parquet(feature_path); extraction_meta = json.loads((outdir / "extraction.json").read_text())
    else:
        started = time.perf_counter(); records = []
        for p in Path(args.cache).rglob("*_lc.fits"):
            m = NAME_RE.search(p.name)
            if m: records.append((str(p), int(m.group(1)), int(m.group(2))))
        files = pd.DataFrame(records, columns=["path", "tic_id", "sector"])
        counts = files.groupby("tic_id").sector.nunique()
        eligible = counts[counts >= args.min_cached_sectors].index.tolist()
        chosen = sorted(eligible, key=lambda t: stable_u01(t, "cohort"))[:args.n_tics]
        selected = files[files.tic_id.isin(chosen)].sort_values(["tic_id", "sector"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(extract_one, selected.itertuples(index=False, name=None)))
        frame = pd.DataFrame(rows)
        elapsed = time.perf_counter() - started
        extraction_meta = {"cache_products": len(files), "eligible_tics": len(eligible),
                           "chosen_tics": len(chosen), "selected_products": len(selected),
                           "readable_products": int(frame.error.eq("").sum()), "elapsed_seconds": elapsed}
        frame.to_parquet(feature_path, index=False)
        (outdir / "extraction.json").write_text(json.dumps(extraction_meta, indent=2) + "\n")
    if args.extract_only:
        print(json.dumps(extraction_meta, indent=2))
        return
    good = frame[frame.error.eq("") & frame.pdcsap_median.gt(0)].copy()
    keep = good.groupby("tic_id").size(); good = good[good.tic_id.isin(keep[keep >= 4].index)].copy()
    good["log_flux"] = np.log(good.pdcsap_median)
    good["label"] = good.log_flux - good.groupby("tic_id").log_flux.transform("mean")
    u = good.tic_id.map(lambda t: stable_u01(int(t), "split"))
    good["split"] = np.where(u < .70, "train", np.where(u < .85, "val", "test"))
    train_tics = set(good.loc[good.split.eq("train"), "tic_id"].astype(int))
    test = good.split.eq("test"); results = {"raw": metric(good[test], np.zeros(test.sum()))}
    raw_star, _ = star_metric(good[test], np.zeros(test.sum()))
    per_star = pd.DataFrame({"tic_id": raw_star.index, "raw_cv": raw_star.values}).set_index("tic_id")
    histories = {}
    for name, columns in GROUPS.items():
        xdf, _ = prepare(good, columns, train_tics); x = xdf.to_numpy(float)
        train = good.split.eq("train"); val = good.split.eq("val")
        global_model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
            learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.,
            early_stopping=True, validation_fraction=None, random_state=42)
        global_model.fit(x[train], good.loc[train, "label"])
        gp = global_model.predict(x[test]); results[f"global/{name}"] = metric(good[test], gp)
        _, gc = star_metric(good[test], gp); per_star[f"global/{name}"] = gc
        pp, hist = pairwise_fit_predict(x, good.label.to_numpy(), good.tic_id.to_numpy(), good.split.to_numpy())
        results[f"pairwise/{name}"] = metric(good[test], pp[test]); histories[name] = hist
        _, pc = star_metric(good[test], pp[test]); per_star[f"pairwise/{name}"] = pc
    comparisons = {}
    names = list(GROUPS)
    for family in ("global", "pairwise"):
        comparisons[f"{family}/core_vs_raw"] = bootstrap_gain(per_star.raw_cv, per_star[f"{family}/core"])
        for before, after in zip(names, names[1:]):
            comparisons[f"{family}/{after}_vs_{before}"] = bootstrap_gain(
                per_star[f"{family}/{before}"], per_star[f"{family}/{after}"])
    per_star.reset_index().to_parquet(outdir / "per_star_metrics.parquet", index=False)
    payload = {"extraction": extraction_meta, "analysis_products": len(good),
               "analysis_tics": int(good.tic_id.nunique()),
               "split_tics": good.groupby("split").tic_id.nunique().to_dict(),
               "results": results, "paired_bootstrap": comparisons, "pairwise_histories": histories}
    (outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
