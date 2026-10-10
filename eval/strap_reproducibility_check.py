"""
Reproducibility check for candidate strap spikes found in strap_fine_diagnostic.py.

Splits stars into two independent halves (by tic_id parity) and recomputes the
fine-grained detrended column/row profile in EACH half separately. A genuine
fixed detector artifact must appear at the SAME location with the SAME sign in
both halves. A statistical fluke will not reproduce.

Also trims edge bins before spike-finding to avoid rolling-median boundary
artifacts contaminating the result.
"""

import numpy as np
import pandas as pd

PARQUET = "training_data_v3_tic.parquet"
N_PIX = 4
SMOOTH_WINDOW = 15
MIN_N_PER_BIN = 10
EDGE_TRIM_BINS = 10   # drop this many bins from each end before spike search

print(f"Loading {PARQUET} ...", flush=True)
df = pd.read_parquet(PARQUET, columns=["tic_id","sector","cam","ccd","col","row","flux_offset_loo","n_sectors_total"])
df = df.dropna(subset=["col","row","flux_offset_loo"])
df = df[df["flux_offset_loo"].between(0.85, 1.15)]
df = df[df["n_sectors_total"] >= 4]
df["half"] = (df["tic_id"] % 2).astype(int)
print(f"  {len(df):,} observations, half A: {(df.half==0).sum():,}  half B: {(df.half==1).sum():,}")

def fine_profile(g, axis, n_pix):
    bins = np.arange(g[axis].min(), g[axis].max() + n_pix, n_pix)
    g = g.copy()
    g["_bin"] = pd.cut(g[axis], bins, labels=False)
    prof = g.groupby("_bin")["flux_offset_loo"].agg(["median", "count"])
    prof = prof[prof["count"] >= MIN_N_PER_BIN]
    return prof, bins

def detrend(prof, window):
    smooth = prof["median"].rolling(window, center=True, min_periods=window//2).median()
    return prof["median"] - smooth

print(f"\n{'Cam':>4} {'CCD':>4} {'Axis':>4} | {'A: loc':>8} {'A: val':>9} | {'B: loc':>8} {'B: val':>9} | {'match?':>8}")
print("-" * 80)

confirmed = []
for (cam, ccd), g_all in df.groupby(["cam", "ccd"]):
    if len(g_all) < 8000:
        continue
    for axis in ["row", "col"]:
        locs, vals = {}, {}
        ok = True
        for half in [0, 1]:
            g = g_all[g_all["half"] == half]
            prof, bins = fine_profile(g, axis, N_PIX)
            if len(prof) < SMOOTH_WINDOW * 2 + 2 * EDGE_TRIM_BINS:
                ok = False
                break
            resid = detrend(prof, SMOOTH_WINDOW).dropna()
            # trim edges by bin index value, not position in series
            lo_bin, hi_bin = prof.index.min() + EDGE_TRIM_BINS, prof.index.max() - EDGE_TRIM_BINS
            resid = resid[(resid.index >= lo_bin) & (resid.index <= hi_bin)]
            if len(resid) == 0:
                ok = False
                break
            spike_bin = resid.abs().idxmax()
            locs[half] = bins[int(spike_bin)]
            vals[half] = resid.loc[spike_bin]
        if not ok:
            continue
        # "match" = within 3 fine-bins (12px) of each other AND same sign
        close = abs(locs[0] - locs[1]) <= (N_PIX * 3)
        same_sign = np.sign(vals[0]) == np.sign(vals[1])
        match = close and same_sign
        flag = " *** REPRODUCES ***" if match else ""
        print(f"{int(cam):>4} {int(ccd):>4} {axis:>4} | {locs[0]:>8.1f} {vals[0]:>+9.5f} | {locs[1]:>8.1f} {vals[1]:>+9.5f} | {'YES' if match else 'no':>8}{flag}")
        if match:
            confirmed.append((cam, ccd, axis, locs[0], locs[1], vals[0], vals[1]))

print(f"\n{len(confirmed)} candidate strap locations reproduce independently across random star halves.")
if confirmed:
    print("These are strong candidates for genuine fixed detector artifacts:")
    for c in confirmed:
        print(f"  Cam{int(c[0])} CCD{int(c[1])} {c[2]}: loc~{(c[3]+c[4])/2:.0f}  amplitude~{(c[5]+c[6])/2:+.4f}")
