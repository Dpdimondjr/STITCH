"""
Strap/banding diagnostic — isolates FIXED detector strapping from sector-varying
pointing/crowding bands.

Rationale: KNN-100 is an isotropic (2D disk) estimator. A true CCD strap artifact
is anisotropic (constant along the strap, sharp across it) — KNN structurally
dilutes it and can't be trusted to reveal it. The correct diagnostic is a 1D
marginal (collapse over the orthogonal axis), averaged ACROSS SECTORS:

  - A FIXED strap (same row/col, same sign, every sector) REINFORCES under a
    many-sector average.
  - A sector-flipping pointing/crowding band (confirmed null in prior analysis)
    CANCELS under a many-sector average.

So: whatever survives the cross-sector average of the row-only / col-only
marginal profile is evidence for genuine, correctable fixed strapping.

Uses flux_offset_loo (per-star LOO-normalized ratio) so per-star brightness is
already removed; only sector/detector-level structure remains.
"""

import numpy as np
import pandas as pd

PARQUET = "training_data_v3_tic.parquet"
N_ROW_BINS = 64
N_COL_BINS = 64
MIN_N_PER_BIN = 20

print(f"Loading {PARQUET} ...", flush=True)
df = pd.read_parquet(PARQUET, columns=["tic_id","sector","cam","ccd","col","row","flux_offset_loo","n_sectors_total"])
df = df.dropna(subset=["col","row","flux_offset_loo"])
df = df[df["flux_offset_loo"].between(0.85, 1.15)]
df = df[df["n_sectors_total"] >= 4]
print(f"  {len(df):,} observations after cleaning")

results = {}

for (cam, ccd), g in df.groupby(["cam", "ccd"]):
    if len(g) < 2000:
        continue
    n_sectors = g["sector"].nunique()

    # Row marginal: bin by row, collapse over column AND sector
    row_bins = np.linspace(g["row"].min(), g["row"].max(), N_ROW_BINS + 1)
    g = g.copy()
    g["row_bin"] = pd.cut(g["row"], row_bins, labels=False)
    row_prof = g.groupby("row_bin")["flux_offset_loo"].agg(["median", "count"])
    row_prof = row_prof[row_prof["count"] >= MIN_N_PER_BIN]

    # Column marginal: bin by column, collapse over row AND sector
    col_bins = np.linspace(g["col"].min(), g["col"].max(), N_COL_BINS + 1)
    g["col_bin"] = pd.cut(g["col"], col_bins, labels=False)
    col_prof = g.groupby("col_bin")["flux_offset_loo"].agg(["median", "count"])
    col_prof = col_prof[col_prof["count"] >= MIN_N_PER_BIN]

    # Signal strength: range and std of the marginal median, relative to
    # what you'd expect from pure sampling noise (SEM of the median ~ MAD/sqrt(n))
    row_signal = row_prof["median"].std()
    col_signal = col_prof["median"].std()
    row_range  = row_prof["median"].max() - row_prof["median"].min()
    col_range  = col_prof["median"].max() - col_prof["median"].min()

    # Expected noise floor per bin (robust SEM), averaged
    overall_mad = (df["flux_offset_loo"] - 1.0).abs().median() * 1.4826
    row_noise_floor = np.median(overall_mad / np.sqrt(row_prof["count"].values))
    col_noise_floor = np.median(overall_mad / np.sqrt(col_prof["count"].values))

    results[(cam, ccd)] = dict(
        n_sectors=n_sectors, n_obs=len(g),
        row_signal=row_signal, row_noise_floor=row_noise_floor, row_range=row_range,
        col_signal=col_signal, col_noise_floor=col_noise_floor, col_range=col_range,
        row_prof=row_prof, col_prof=col_prof, row_bins=row_bins, col_bins=col_bins,
    )

print(f"\n{'Cam':>4} {'CCD':>4} {'nSec':>5} {'nObs':>8} | {'row σ':>8} {'row noise':>10} {'row SNR':>8} {'row rng':>8} | {'col σ':>8} {'col noise':>10} {'col SNR':>8} {'col rng':>8}")
print("-" * 115)
for (cam, ccd), r in sorted(results.items()):
    row_snr = r["row_signal"] / r["row_noise_floor"]
    col_snr = r["col_signal"] / r["col_noise_floor"]
    flag_row = " <<" if row_snr > 2.0 else ""
    flag_col = " <<" if col_snr > 2.0 else ""
    print(f"{int(cam):>4} {int(ccd):>4} {r['n_sectors']:>5} {r['n_obs']:>8,} | "
          f"{r['row_signal']:>8.5f} {r['row_noise_floor']:>10.5f} {row_snr:>7.2f}{flag_row:3s} {r['row_range']:>8.5f} | "
          f"{r['col_signal']:>8.5f} {r['col_noise_floor']:>10.5f} {col_snr:>7.2f}{flag_col:3s} {r['col_range']:>8.5f}")

print("\nSNR = cross-sector-averaged marginal std / expected sampling-noise floor.")
print("SNR >> 1 means the marginal profile has structure too large to be noise —")
print("i.e. survives averaging across many sectors = candidate FIXED strap.")

# Save detailed profiles for the highest-SNR cam/ccd for visualization
import pickle
with open("eval/banding_results/strap_marginal_profiles.pkl", "wb") as f:
    pickle.dump(results, f)
print("\nSaved detailed profiles -> eval/banding_results/strap_marginal_profiles.pkl")
