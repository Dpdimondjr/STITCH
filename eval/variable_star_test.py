"""
Variable star preservation test.

Downloads SPOC lightcurves for known variable stars, computes per-sector medians,
runs STITCH, and plots before vs after to check whether astrophysical variability
is preserved by the correction.

The spatial context features (sector_ccd_mean_loo, spatial_knn_mean_loo) are
derived from the quiet-star parquet — the same neighbours STITCH would have seen
for any star on that CCD/sector.

Usage:
  python3 eval/variable_star_test.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
"""

import sys, warnings, os
import numpy as np
import pandas as pd
import torch, zuko
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")

# Known variable stars: (TIC, common name, var type, approx period)
TARGETS = [
    (229804573, "RR Lyr",         "RRab",       0.57),   # archetype RR Lyrae, period < 1d
    (441420236, "V* V Boo",       "SRb",        258.0),  # semi-regular, ~258d
    (372919087, "V* Y Tau",       "SRb",        241.0),  # semi-regular
    (289349670, "V* R Dor",       "SRb",        175.0),  # semi-regular, bright
    (141306459, "V* W Vir",       "CWB",        17.3),   # Population II Cepheid ~17d
    (167602025, "V* AW Lac",      "EW",         6.9),    # W UMa eclipsing binary
    (150428135, "V* EX Dra",      "EA",         5.06),   # Algol-type EB
    (264614817, "V* V523 Cas",    "ROT",        1.88),   # rotating spotted
]

print(f"Model: {MODEL}  Parquet: {PARQUET}")

# ── Load model ─────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"], bins=cfg["bins"])
flow.load_state_dict(ckpt["model_state"]); flow.eval()
means=ckpt["means"]; stds=ckpt["stds"]; y_mean=ckpt["y_mean"]; y_std=ckpt["y_std"]
CONTINUOUS=ckpt["continuous_cols"]; CAM_COLS=ckpt["cam_cols"]; CCD_COLS=ckpt["ccd_cols"]

device=(torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
flow=flow.to(device)

# ── Load quiet-star parquet (for spatial context) ──────────────────────────────
print("Loading parquet …", flush=True)
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col]=df[col].fillna(df[col].median())
df = df.reset_index(drop=True)

# Precompute sector_ccd_mean (non-LOO for context lookup — LOO is star-level, not needed here)
_t="flux_offset_loo"
df["sector_ccd_mean"]=df.groupby(["sector","cam","ccd"])[_t].transform("mean")
print(f"  Quiet stars: {df['tic_id'].nunique():,}  records: {len(df):,}")

# ── Download SPOC lightcurves for each variable star ──────────────────────────
from lightkurve import search_lightcurve
import lightkurve as lk

def get_sector_data(tic_id, name):
    """Download SPOC 2-min lightcurves, return per-sector summary DataFrame."""
    print(f"  Searching MAST for TIC {tic_id} ({name}) …", flush=True)
    try:
        results = search_lightcurve(f"TIC {tic_id}", mission="TESS",
                                    author="SPOC", exptime=120)
        if len(results) == 0:
            print(f"    No SPOC 2-min data found.")
            return None
        print(f"    Found {len(results)} sector(s). Downloading …", flush=True)
        lcs = results.download_all(quality_bitmask="default")
    except Exception as e:
        print(f"    Download failed: {e}")
        return None

    rows = []
    for lc in lcs:
        try:
            lc_clean = lc.remove_nans().remove_outliers(sigma=5)
            if len(lc_clean) < 100:
                continue
            flux = lc_clean.flux.value
            sector = int(lc.meta.get("SECTOR", -1))
            cam    = int(lc.meta.get("CAMERA", -1))
            ccd    = int(lc.meta.get("CCD", -1))
            col    = float(lc.meta.get("COLUMN", np.nan))
            row    = float(lc.meta.get("ROW", np.nan))
            tmag   = float(lc.meta.get("TESSMAG", np.nan))
            gaiarp = float(lc.meta.get("GAIARP", lc.meta.get("GAIAmag", np.nan)))
            crowdsap  = float(lc.meta.get("CROWDSAP", 1.0))
            cdpp1_0   = float(lc.meta.get("CDPP1_0",  np.nan))
            pdcvar    = float(lc.meta.get("PDCVAR",   np.nan))
            jitter_rms= float(lc.meta.get("JITTRMS",  np.nan))
            pdc_noi   = float(lc.meta.get("PDC_NOI",  np.nan))
            pr_wght2  = float(lc.meta.get("PRWGHT2",  np.nan))
            # subpixel
            col_p = lc.meta.get("COLUMN_RP", col)
            row_p = lc.meta.get("ROW_RP",    row)
            delta_sub_col = float(col_p) - float(col) if col_p else 0.0
            delta_sub_row = float(row_p) - float(row) if row_p else 0.0

            sector_median = float(np.nanmedian(flux))
            rows.append(dict(
                tic_id=tic_id, sector=sector, cam=cam, ccd=ccd,
                col=col, row=row, delta_sub_col=delta_sub_col, delta_sub_row=delta_sub_row,
                tmag=tmag, gaiarp=gaiarp, crowdsap=crowdsap,
                cdpp1_0=cdpp1_0, pdcvar=pdcvar, jitter_rms=jitter_rms,
                pdc_noi=pdc_noi, pr_wght2=pr_wght2,
                sector_median=sector_median,
            ))
        except Exception as e:
            print(f"    Sector error: {e}")
            continue

    if not rows:
        return None
    return pd.DataFrame(rows)


def add_spatial_context(vdf):
    """Add sector_ccd_mean_loo and spatial_knn_mean_loo from quiet-star parquet."""
    vdf = vdf.copy()

    # sector_ccd_mean: mean of quiet stars on same sector/cam/ccd
    ccd_means = (df.groupby(["sector","cam","ccd"])["flux_offset_loo"]
                   .mean().rename("sector_ccd_mean_loo").reset_index())
    vdf = vdf.merge(ccd_means, on=["sector","cam","ccd"], how="left")
    vdf["sector_ccd_mean_loo"] = vdf["sector_ccd_mean_loo"].fillna(1.0)

    # spatial_knn_mean: K=20 nearest quiet neighbours in col/row space per sector/cam/ccd
    _K = 20
    knn_vals = np.full(len(vdf), np.nan)
    for i, row_v in vdf.iterrows():
        key = (row_v["sector"], row_v["cam"], row_v["ccd"])
        mask = ((df["sector"]==key[0])&(df["cam"]==key[1])&(df["ccd"]==key[2]))
        nbrs = df[mask]
        if len(nbrs) < 2:
            continue
        coords = nbrs[["col","row"]].values.astype(np.float32)
        vals   = nbrs["flux_offset_loo"].values
        qpt = np.array([[row_v["col"], row_v["row"]]], dtype=np.float32)
        if not np.all(np.isfinite(qpt)):
            continue
        finite_mask = np.all(np.isfinite(coords), axis=1)
        coords_f = coords[finite_mask]; vals_f = vals[finite_mask]
        if len(coords_f) < 2:
            continue
        k = min(_K, len(coords_f))
        _, nn = cKDTree(coords_f).query(qpt, k=k)
        knn_vals[i] = vals_f[nn[0]].mean()
    vdf["spatial_knn_mean_loo"] = knn_vals
    vdf["spatial_knn_mean_loo"] = vdf["spatial_knn_mean_loo"].fillna(vdf["sector_ccd_mean_loo"])

    # perstar_gaia_offset: for variable stars, use 1.0 (we don't have a quiet reference)
    # (the model was trained with this as gaia-normalised mean; for variables it's noisy)
    vdf["perstar_gaia_offset"] = 1.0

    return vdf


def run_stitch(vdf):
    """Run STITCH inference on a variable-star sector DataFrame."""
    vdf = vdf.copy().reset_index(drop=True)
    # Fill missing continuous features with parquet medians
    for col in CONTINUOUS:
        if col not in vdf.columns:
            vdf[col] = df[col].median() if col in df.columns else 0.0
        vdf[col] = vdf[col].fillna(df[col].median() if col in df.columns else 0.0)

    cont = (vdf[CONTINUOUS].reindex(columns=CONTINUOUS, fill_value=0) - means) / stds
    cam_oh = pd.get_dummies(vdf["cam"].astype(int), prefix="cam").reindex(columns=CAM_COLS, fill_value=0)
    ccd_oh = pd.get_dummies(vdf["ccd"].astype(int), prefix="ccd").reindex(columns=CCD_COLS, fill_value=0)
    C = torch.tensor(
        pd.concat([cont.reset_index(drop=True),
                   cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)], axis=1).values.astype("float32")
    ).to(device)

    with torch.no_grad():
        samp = flow(C).sample((300,)).squeeze(-1)
        mu_z = samp.mean(0).cpu().numpy()
        sig_z= samp.std(0).cpu().numpy()

    pred_raw = mu_z*y_std + y_mean
    pred_std = sig_z*y_std
    weight   = 1.0/(1.0+5.0*pred_std)
    pred     = weight*pred_raw + (1-weight)*1.0
    vdf["pred_offset"] = pred
    vdf["weight"]      = weight
    vdf["pred_std"]    = pred_std
    return vdf


# ── Main loop ──────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"

results = []
for tic, name, vartype, period in TARGETS:
    print(f"\n── {name} (TIC {tic}, {vartype}, P~{period}d) ──")
    vdf = get_sector_data(tic, name)
    if vdf is None or len(vdf) < 3:
        print(f"  Skipped (insufficient data)")
        continue
    print(f"  Got {len(vdf)} sectors: {sorted(vdf['sector'].unique())}")
    vdf = add_spatial_context(vdf)
    vdf = run_stitch(vdf)
    vdf["name"] = name
    vdf["vartype"] = vartype
    vdf["period"] = period
    results.append(vdf)
    print(f"  Mean weight: {vdf['weight'].mean():.3f}  pred_std: {vdf['pred_std'].mean():.4f}")

if not results:
    print("No data downloaded. Check network / TIC IDs.")
    sys.exit(1)

all_df = pd.concat(results, ignore_index=True)
all_df.to_csv("eval/variable_star_test.csv", index=False)
print(f"\nSaved → eval/variable_star_test.csv")

# ── Plot ───────────────────────────────────────────────────────────────────────
SECTOR_DAYS = 27.4
n = len(results)
ncols = 2; nrows = (n+1)//2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5*nrows), facecolor=BG)
axes = np.array(axes).flatten()

for ax, vdf in zip(axes, results):
    vdf = vdf.sort_values("sector")
    t   = vdf["sector"].values * SECTOR_DAYS
    raw = vdf["sector_median"].values
    po  = vdf["pred_offset"].values
    wt  = vdf["weight"].mean()
    ps  = vdf["pred_std"].mean()

    raw_norm  = raw / raw.mean()
    corr      = raw / po
    corr_norm = corr / corr.mean()

    raw_sc  = raw_norm.std()*100
    corr_sc = corr_norm.std()*100
    name = vdf["name"].iloc[0]
    vtype= vdf["vartype"].iloc[0]
    per  = vdf["period"].iloc[0]
    nsec = len(vdf)

    ax.set_facecolor(SURF)
    ax.plot(t, raw_norm,  color="#ef4444", lw=1.5, alpha=0.85,
            label=f"Raw  σ={raw_sc:.1f}%")
    ax.plot(t, corr_norm, color="#4f8ef7", lw=1.5, alpha=0.9,
            label=f"STITCH  σ={corr_sc:.1f}%")
    ax.axhline(1, color=DIM, lw=0.5, ls="--")
    ax.set_title(f"{name}  [{vtype}, P~{per}d, {nsec} sectors]\n"
                 f"weight={wt:.3f}  pred_std={ps:.4f}",
                 color=TEXT, fontsize=9, pad=6)
    ax.set_xlabel("Time (days)", color=MUT, fontsize=8)
    ax.set_ylabel("Norm. flux", color=MUT, fontsize=8)
    ax.tick_params(colors=MUT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, lw=0.4, zorder=0); ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT,
              facecolor="#0f1c30", edgecolor=DIM)

for ax in axes[len(results):]:
    ax.set_visible(False)

fig.suptitle("Known variable stars: raw vs STITCH sector-median time series",
             color=TEXT, fontsize=12, y=1.01)
plt.tight_layout()
out = "eval/variable_star_test.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
