#!/usr/bin/env python3
"""Extract physically motivated SPOC LC features from a bounded local sample.

The script never downloads data.  It parses TIC/sector from cached filenames, matches
them to a parquet catalogue, takes a deterministic sector/camera/CCD-stratified
sample, and writes one row per LC product.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

NAME_RE = re.compile(r"_(\d+)-s(\d+)_tess_v\d+_lc\.fits$")


def robust_stats(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"{prefix}_{s}": np.nan for s in ("median", "mad", "rms", "p05", "p95")}
    med = np.median(x)
    return {
        f"{prefix}_median": float(med),
        f"{prefix}_mad": float(1.4826 * np.median(np.abs(x - med))),
        f"{prefix}_rms": float(np.sqrt(np.mean((x - med) ** 2))),
        f"{prefix}_p05": float(np.percentile(x, 5)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
    }


def aperture_features(mask: np.ndarray) -> dict[str, float]:
    # SPOC aperture bit 2 identifies pixels in the optimal photometric aperture.
    ap = (np.asarray(mask, dtype=np.int64) & 2) != 0
    yy, xx = np.where(ap)
    if not len(xx):
        return {k: np.nan for k in ("ap_npix", "ap_width", "ap_height", "ap_perimeter",
                                    "ap_compactness", "ap_centroid_x", "ap_centroid_y")}
    perimeter = 0
    for y, x in zip(yy, xx):
        perimeter += sum(y + dy < 0 or y + dy >= ap.shape[0] or x + dx < 0 or
                         x + dx >= ap.shape[1] or not ap[y + dy, x + dx]
                         for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    area = len(xx)
    return {
        "ap_npix": float(area), "ap_width": float(xx.max() - xx.min() + 1),
        "ap_height": float(yy.max() - yy.min() + 1), "ap_perimeter": float(perimeter),
        "ap_compactness": float(4 * np.pi * area / perimeter**2),
        "ap_centroid_x": float(xx.mean()), "ap_centroid_y": float(yy.mean()),
    }


def extract_one(item: tuple[str, int, int]) -> dict:
    path, tic, sector = item
    out: dict = {"tic_id": tic, "sector": sector, "path": path, "error": ""}
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
            p, lc, ap = hdus[0].header, hdus[1], hdus[2]
            h = lc.header
            out.update({
                "camera_fits": p.get("CAMERA"), "ccd_fits": p.get("CCD"),
                "procver": p.get("PROCVER"), "data_rel": p.get("DATA_REL"),
                "pdc_method": h.get("PDCMETHD"), "cadence_days": h.get("TIMEDEL"),
                "num_frames": h.get("NUM_FRM"), "crowdsap_fits": h.get("CROWDSAP"),
                "flfrcsap_fits": h.get("FLFRCSAP"), "tmag_fits": p.get("TESSMAG"),
                "teff_fits": p.get("TEFF"), "cutout_col0": ap.header.get("CRVAL1P"),
                "cutout_row0": ap.header.get("CRVAL2P"),
            })
            d = lc.data
            q = np.asarray(d["QUALITY"], dtype=np.int64)
            finite_flux = np.isfinite(d["PDCSAP_FLUX"])
            good = (q == 0) & finite_flux
            use = good if good.sum() >= 10 else finite_flux
            out.update({"n_cadences": int(len(q)), "n_good": int(good.sum()),
                        "quality_nonzero_frac": float(np.mean(q != 0)),
                        "finite_pdcsap_frac": float(np.mean(finite_flux))})
            for col, name in (("SAP_BKG", "sap_bkg"), ("POS_CORR1", "pos_corr1"),
                              ("POS_CORR2", "pos_corr2")):
                out.update(robust_stats(np.asarray(d[col])[use], name))
            x, y = np.asarray(d["POS_CORR1"], float)[use], np.asarray(d["POS_CORR2"], float)[use]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                cov = np.cov(x[ok], y[ok], ddof=0)
                eig = np.linalg.eigvalsh(cov)
                out.update(pos_corr_cov=float(cov[0, 1]), pos_corr_corr=float(np.corrcoef(x[ok], y[ok])[0, 1]),
                           pos_corr_major_rms=float(np.sqrt(max(eig[-1], 0))),
                           pos_corr_minor_rms=float(np.sqrt(max(eig[0], 0))))
            else:
                out.update(pos_corr_cov=np.nan, pos_corr_corr=np.nan,
                           pos_corr_major_rms=np.nan, pos_corr_minor_rms=np.nan)
            dx = np.asarray(d["MOM_CENTR1"], float)[use] - np.asarray(d["PSF_CENTR1"], float)[use]
            dy = np.asarray(d["MOM_CENTR2"], float)[use] - np.asarray(d["PSF_CENTR2"], float)[use]
            out.update(robust_stats(np.hypot(dx, dy), "mom_psf_sep"))
            out.update(aperture_features(ap.data))
    except Exception as exc:  # retain failures for an auditable coverage denominator
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--cache", default="tess_cache/mastDownload")
    pa.add_argument("--catalog", default="training_data_v3.parquet")
    pa.add_argument("--output", default="eval/spoc_feature_trial/features.parquet")
    pa.add_argument("--max-products", type=int, default=2000)
    pa.add_argument("--workers", type=int, default=8)
    args = pa.parse_args()
    started = time.perf_counter()
    files = []
    for p in Path(args.cache).rglob("*_lc.fits"):
        m = NAME_RE.search(p.name)
        if m:
            files.append((str(p), int(m.group(1)), int(m.group(2))))
    scan_seconds = time.perf_counter() - started
    catalog = pd.read_parquet(args.catalog, columns=["tic_id", "sector", "cam", "ccd", "col", "row"])
    catalog = catalog.dropna(subset=["tic_id", "sector"]).copy()
    catalog[["tic_id", "sector"]] = catalog[["tic_id", "sector"]].astype("int64")
    cat = catalog.drop_duplicates(["tic_id", "sector"])
    local = pd.DataFrame(files, columns=["path", "tic_id", "sector"])
    matched = local.merge(cat, on=["tic_id", "sector"], how="inner", validate="many_to_one")
    # Balanced allocation across detector/sector cells; deterministic final cap.
    matched = matched.sort_values(["sector", "cam", "ccd", "tic_id", "path"])
    cells = matched.groupby(["sector", "cam", "ccd"], dropna=False, group_keys=False)
    per_cell = max(1, int(np.ceil(args.max_products / max(cells.ngroups, 1))))
    selected = cells.head(per_cell).head(args.max_products).copy()
    tasks = list(selected[["path", "tic_id", "sector"]].itertuples(index=False, name=None))
    extract_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(extract_one, tasks))
    extract_seconds = time.perf_counter() - extract_started
    out = pd.DataFrame(rows).merge(selected.drop(columns="path"), on=["tic_id", "sector"], how="left")
    # Target location relative to the cutout and distance to its rectangular edge.
    out["target_local_x"] = out["col"] - out["cutout_col0"]
    out["target_local_y"] = out["row"] - out["cutout_row0"]
    out["target_to_ap_centroid"] = np.hypot(out.target_local_x - out.ap_centroid_x,
                                             out.target_local_y - out.ap_centroid_y)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    meta = {
        "catalog": args.catalog, "cache_products": len(local),
        "unique_cache_tic_sector": int(local[["tic_id", "sector"]].drop_duplicates().shape[0]),
        "matched_products": len(matched), "matched_unique_tic_sector": int(matched[["tic_id", "sector"]].drop_duplicates().shape[0]),
        "selected": len(selected), "successful": int(out.error.eq("").sum()),
        "scan_seconds": scan_seconds, "extract_seconds": extract_seconds,
        "products_per_second": len(selected) / extract_seconds if extract_seconds else None,
        "workers": args.workers,
    }
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
