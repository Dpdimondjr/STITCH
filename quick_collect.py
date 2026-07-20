"""
Fast data collection for 4 sectors — enough to populate the heatmap
with real star measurements and get multi-sector gradient variation.

Runs in ~20–40 minutes depending on MAST response.
Results feed directly into visualize_stitch.py.
"""

import os, warnings, threading
import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.mast import Observations
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SECTOR_RANGE = range(68, 72)   # 4 sectors: 68, 69, 70, 71
CAMERA       = 4
CCD          = 1
N_SCREEN     = 120             # LC headers to screen per sector
N_STABLE     = 8              # stable stars to keep per sector
TMAG_MIN     = 10.0
TMAG_MAX     = 13.5
N_WORKERS    = 3
CACHE        = "./tess_cache"
os.makedirs(CACHE, exist_ok=True)

mast_sem = threading.Semaphore(2)


def screen_sector(sector):
    cache_file = os.path.join(CACHE, f"headers_s{sector:04d}_n{N_SCREEN}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        n = len(df[(df["camera"] == CAMERA) & (df["ccd"] == CCD)])
        print(f"  [s{sector}] cache hit ({n} on Cam{CAMERA}/CCD{CCD})")
        return df

    print(f"  [s{sector}] querying MAST...")
    with mast_sem:
        try:
            obs = Observations.query_criteria(
                obs_collection="TESS", sequence_number=sector,
                provenance_name="SPOC", dataproduct_type="timeseries",
            )
        except Exception as e:
            print(f"  [s{sector}] MAST query failed: {e}")
            return pd.DataFrame()

    if len(obs) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(sector)
    idx = rng.choice(len(obs), size=min(N_SCREEN, len(obs)), replace=False)

    with mast_sem:
        try:
            products = Observations.get_product_list(obs[idx])
        except Exception as e:
            print(f"  [s{sector}] product list failed: {e}")
            return pd.DataFrame()

    lc_prods = products[products["productSubGroupDescription"] == "LC"]
    if len(lc_prods) == 0:
        return pd.DataFrame()

    with mast_sem:
        try:
            manifest = Observations.download_products(
                lc_prods, download_dir=CACHE, cache=True
            )
        except Exception as e:
            print(f"  [s{sector}] download failed: {e}")
            return pd.DataFrame()

    records = []
    for row in manifest:
        path = row["Local Path"]
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path, memmap=True) as hdul:
                h0, h1 = hdul[0].header, hdul[1].header
                tmag = h0.get("TESSMAG")
                cdpp = h1.get("CDPP1_0")
                if tmag is None or cdpp is None:
                    continue
                records.append({
                    "tic_id":   h0.get("TICID"),
                    "tmag":     tmag,
                    "cdpp1_0":  cdpp,
                    "pdcvar":   h1.get("PDCVAR"),
                    "crowdsap": h0.get("CROWDSAP"),
                    "camera":   h0.get("CAMERA"),
                    "ccd":      h0.get("CCD"),
                    "sector":   h0.get("SECTOR"),
                })
        except Exception:
            pass

    df = pd.DataFrame(records)
    df.to_csv(cache_file, index=False)
    n = len(df[(df["camera"] == CAMERA) & (df["ccd"] == CCD)])
    print(f"  [s{sector}] screened {len(df)} files, {n} on Cam{CAMERA}/CCD{CCD}")
    return df


def get_star_offsets(tic, sector):
    try:
        sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC")
        if len(sr) == 0:
            return None
        lc_col = sr.download_all(download_dir=CACHE)
        if not lc_col or len(lc_col) == 0:
            return None

        all_flux    = np.concatenate([lc.flux.value for lc in lc_col])
        global_med  = np.nanmedian(all_flux)
        if global_med == 0 or np.isnan(global_med):
            return None

        for lc in lc_col:
            if (lc.meta.get("SECTOR") == sector and
                    lc.meta.get("CAMERA") == CAMERA and
                    lc.meta.get("CCD")    == CCD):
                flux_offset = np.nanmedian(lc.flux.value) / global_med
                try:
                    pc1 = lc["pos_corr1"].value.astype(float)
                    pc2 = lc["pos_corr2"].value.astype(float)
                    jitter_rms = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
                except Exception:
                    jitter_rms = np.nan
                return {"flux_offset": flux_offset, "jitter_rms": jitter_rms,
                        "n_sectors": len(lc_col)}
        return None
    except Exception:
        return None


def get_tpf_position(tic, sector):
    cache_file = os.path.join(CACHE, f"tpf_pos_tic{tic}_s{sector:04d}.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file).iloc[0].to_dict()
    try:
        sr = lk.search_targetpixelfile(f"TIC {tic}", sector=sector, author="SPOC")
        if len(sr) == 0:
            return None
        tpf = sr[0].download(download_dir=CACHE)

        col_c = tpf.column + tpf.shape[2] / 2.0
        row_c = tpf.row    + tpf.shape[1] / 2.0
        pc1   = tpf.hdu[1].data["POS_CORR1"].astype(float)
        pc2   = tpf.hdu[1].data["POS_CORR2"].astype(float)
        result = {
            "col":           col_c,
            "row":           row_c,
            "delta_sub_col": (col_c + np.nanmedian(pc1)) % 1,
            "delta_sub_row": (row_c + np.nanmedian(pc2)) % 1,
            "aap":           int(tpf.pipeline_mask.sum()),
        }
        pd.DataFrame([result]).to_csv(cache_file, index=False)
        return result
    except Exception:
        return None


def process_sector(sector):
    out_cache = os.path.join(CACHE, f"sector_result_s{sector:04d}_cam{CAMERA}_ccd{CCD}.csv")
    if os.path.exists(out_cache):
        df = pd.read_csv(out_cache)
        print(f"  [s{sector}] result cache hit ({len(df)} stars)")
        return df

    headers = screen_sector(sector)
    if headers.empty:
        return pd.DataFrame()

    candidates = (
        headers
        .dropna(subset=["cdpp1_0", "tmag", "camera", "ccd"])
        .query(f"camera == {CAMERA} and ccd == {CCD}")
        .query(f"{TMAG_MIN} <= tmag <= {TMAG_MAX}")
        .sort_values("cdpp1_0")
        .head(N_STABLE)
    )

    if candidates.empty:
        print(f"  [s{sector}] no candidates after filtering")
        return pd.DataFrame()

    print(f"  [s{sector}] {len(candidates)} candidates — fetching offsets + positions...")

    records = []
    for _, star in candidates.iterrows():
        tic     = int(star["tic_id"])
        offsets = get_star_offsets(tic, sector)
        if offsets is None:
            continue
        pos = get_tpf_position(tic, sector)
        if pos is None:
            continue
        records.append({
            "tic_id":        tic,
            "sector":        sector,
            "tmag":          star["tmag"],
            "cdpp1_0":       star["cdpp1_0"],
            "crowdsap":      star.get("crowdsap"),
            "col":           pos["col"],
            "row":           pos["row"],
            "delta_sub_col": pos["delta_sub_col"],
            "delta_sub_row": pos["delta_sub_row"],
            "aap":           pos["aap"],
            "flux_offset":   offsets["flux_offset"],
            "jitter_rms":    offsets["jitter_rms"],
            "n_sectors":     offsets["n_sectors"],
        })

    df = pd.DataFrame(records)
    df.to_csv(out_cache, index=False)
    print(f"  [s{sector}] done — {len(df)} stars")
    return df


if __name__ == "__main__":
    print(f"Collecting sectors {list(SECTOR_RANGE)} "
          f"(N_SCREEN={N_SCREEN}, N_STABLE={N_STABLE})...\n")

    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_sector, s): s for s in SECTOR_RANGE}
        for future in as_completed(futures):
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                print(f"  sector error: {e}")

    if not results:
        print("No results. Try widening TMAG range or increasing N_SCREEN.")
        raise SystemExit(1)

    combined = pd.concat(results, ignore_index=True)
    out = os.path.join(CACHE, f"all_sectors_cam{CAMERA}_ccd{CCD}.csv")
    combined.to_csv(out, index=False)

    print(f"\nSectors found: {sorted(combined['sector'].unique())}")
    print(f"Total (star, sector) pairs: {len(combined)}")
    print(f"Unique stars: {combined['tic_id'].nunique()}")
    print(f"Flux offset range: {combined['flux_offset'].min():.4f} – "
          f"{combined['flux_offset'].max():.4f}")
    print(f"\nSaved → {out}")
    print("Now run: python3 visualize_stitch.py")
