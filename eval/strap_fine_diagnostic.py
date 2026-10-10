"""
Fine-resolution strap diagnostic — tests for NARROW, sharp fixed-row/column
artifacts (true CCD strapping), as distinct from the smooth large-scale
gradient already found in the coarse (64-bin) marginal profile.

Approach: bin at fine resolution (every N_PIX pixels), then DETREND by
subtracting a smooth (low-order) fit, leaving only sharp local deviations.
A real strap shows up as an isolated spike/dip in the detrended residual
that is large relative to the local noise and persists at the same
row/col location — check by also running on independent sub-splits.
"""

import numpy as np
import pandas as pd

PARQUET = "training_data_v3_tic.parquet"
N_PIX = 4           # fine bin width in pixels
SMOOTH_WINDOW = 15  # bins, for rolling-median detrend
MIN_N_PER_BIN = 15

print(f"Loading {PARQUET} ...", flush=True)
df = pd.read_parquet(PARQUET, columns=["tic_id","sector","cam","ccd","col","row","flux_offset_loo","n_sectors_total"])
df = df.dropna(subset=["col","row","flux_offset_loo"])
df = df[df["flux_offset_loo"].between(0.85, 1.15)]
df = df[df["n_sectors_total"] >= 4]
print(f"  {len(df):,} observations after cleaning")

def fine_profile(g, axis, n_pix):
    bins = np.arange(g[axis].min(), g[axis].max() + n_pix, n_pix)
    g = g.copy()
    g["_bin"] = pd.cut(g[axis], bins, labels=False)
    prof = g.groupby("_bin")["flux_offset_loo"].agg(["median", "count"])
    prof = prof[prof["count"] >= MIN_N_PER_BIN]
    return prof, bins

def detrend(prof, window):
    smooth = prof["median"].rolling(window, center=True, min_periods=window//2).median()
    resid = prof["median"] - smooth
    return resid

print(f"\n{'Cam':>4} {'CCD':>4} {'Axis':>4} {'nBins':>6} | {'resid std':>10} {'max |resid|':>12} {'spike loc':>10} {'spike val':>10}")
print("-" * 80)

spike_report = []
for (cam, ccd), g in df.groupby(["cam", "ccd"]):
    if len(g) < 5000:
        continue
    for axis in ["row", "col"]:
        prof, bins = fine_profile(g, axis, N_PIX)
        if len(prof) < SMOOTH_WINDOW * 2:
            continue
        resid = detrend(prof, SMOOTH_WINDOW).dropna()
        if len(resid) == 0:
            continue
        resid_std = resid.std()
        max_abs = resid.abs().max()
        spike_bin = resid.abs().idxmax()
        spike_val = resid.loc[spike_bin]
        spike_pos = bins[int(spike_bin)] if int(spike_bin) < len(bins) else np.nan
        snr = max_abs / resid_std if resid_std > 0 else np.nan
        flag = " <<<" if snr > 5 else (" <<" if snr > 3.5 else "")
        print(f"{int(cam):>4} {int(ccd):>4} {axis:>4} {len(prof):>6} | {resid_std:>10.5f} {max_abs:>12.5f} {spike_pos:>10.1f} {spike_val:>+10.5f}{flag}")
        spike_report.append(dict(cam=cam, ccd=ccd, axis=axis, resid=resid, bins=bins, snr=snr))

print("\nresid = fine-bin median minus a 15-bin (60px) rolling-median smooth trend.")
print("Large std + an isolated outlier bin (not just high overall std) = candidate strap location.")
print("A smoothly elevated std with NO single dominant spike = just noisier fine-binning, not a strap.")

import pickle
with open("eval/banding_results/strap_fine_profiles.pkl", "wb") as f:
    pickle.dump(spike_report, f)
print("\nSaved -> eval/banding_results/strap_fine_profiles.pkl")
