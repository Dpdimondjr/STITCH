"""
Download 3 test-set stars and produce a 3-row before/after light curve grid.
Each row = one star; left = raw PDCSAP, right = STITCH corrected.
All individual 2-min cadence points shown.
"""

import warnings, numpy as np, pandas as pd, torch, zuko
import matplotlib, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lightkurve as lk
from tess_stars2px import tess_stars2px_function_entry

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

CACHE_DIR = "./tess_cache"

# Three test stars: varied scatter levels and cameras
TARGETS = [
    288470684,   # 5.1% → 1.8%, 12 sectors, Cam 3
    237203089,   # 3.9% → 0.8%, 10 sectors (most dramatic %)
    47733410,    # 3.8% → 1.3%, 10 sectors, Tmag 11.7 (brighter)
]

# ── Style ─────────────────────────────────────────────────────────────────────
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
MUTED    = "#898781"
GRID     = "#e1e0d9"
C_BEFORE = "#898781"
C_AFTER  = "#2a78d6"
CAM_COLS = {1:"#e05c3a", 2:"#2a78d6", 3:"#1baf7a", 4:"#c47900"}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
ckpt = torch.load("stitch_nsf.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["flow_config"]
flow = zuko.flows.NSF(
    features=cfg["features"], context=cfg["context"],
    transforms=cfg["transforms"], hidden_features=cfg["hidden_features"],
    bins=cfg["bins"],
)
flow.load_state_dict(ckpt["model_state"])
flow.eval()

means      = ckpt["means"]
stds       = ckpt["stds"]
y_mean     = ckpt["y_mean"]
y_std      = ckpt["y_std"]
CONTINUOUS = ckpt["continuous_cols"]
CAM_COLS_OHE = ckpt["cam_cols"]
CCD_COLS_OHE = ckpt["ccd_cols"]


def run_star(tic_id):
    print(f"\n── TIC {tic_id} ──────────────────────")
    sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
    try:
        sr = sr[sr.exptime.value >= 100]
    except Exception:
        pass
    print(f"  {len(sr)} SPOC sectors found")

    lc_list = []
    for i in range(len(sr)):
        try:
            lc = sr[i].download(download_dir=CACHE_DIR)
            if lc is not None:
                lc_list.append(lc)
                print(f"  S{lc.meta.get('SECTOR')} ✓", end="", flush=True)
        except Exception as e:
            print(f"  S{i} ✗", end="", flush=True)
    print()

    if not lc_list:
        return None

    ra  = float(lc_list[0].meta["RA_OBJ"])
    dec = float(lc_list[0].meta["DEC_OBJ"])
    _, _, _, out_sec, out_cam, out_ccd, out_col, out_row, _ = \
        tess_stars2px_function_entry(tic_id, ra, dec)
    pos = {int(s): (int(c1), int(c2), float(cl), float(rw))
           for s, c1, c2, cl, rw in zip(out_sec, out_cam, out_ccd, out_col, out_row)}

    meta_rows, lcs_valid = [], []
    for lc in lc_list:
        sec = lc.meta.get("SECTOR")
        if sec is None or int(sec) not in pos:
            continue
        cam_tp, ccd_tp, col, row = pos[int(sec)]
        try:
            pc1 = lc["pos_corr1"].value.astype(float)
            pc2 = lc["pos_corr2"].value.astype(float)
            pc1_med = float(np.nanmedian(pc1))
            pc2_med = float(np.nanmedian(pc2))
            jitter  = float(np.sqrt(np.nanvar(pc1) + np.nanvar(pc2)))
        except Exception:
            pc1_med = pc2_med = jitter = np.nan
        sec_med = float(np.nanmedian(lc.flux.value))
        def _flt(v):
            try: return float(v)
            except: return np.nan
        meta_rows.append({
            "sector": float(sec), "cam": cam_tp, "ccd": ccd_tp,
            "col": col, "row": row,
            "delta_sub_col": (col + pc1_med) % 1.0,
            "delta_sub_row": (row + pc2_med) % 1.0,
            "tmag":     _flt(lc.meta.get("TESSMAG")),
            "crowdsap": _flt(lc.meta.get("CROWDSAP")),
            "cdpp1_0":  _flt(lc.meta.get("CDPP1_0")),
            "pdcvar":   _flt(lc.meta.get("PDCVAR")),
            "jitter_rms": jitter,
            "log_sector_median": np.log1p(max(sec_med, 0)),
        })
        lcs_valid.append(lc)

    if not lcs_valid:
        return None

    meta_df = pd.DataFrame(meta_rows)
    meta_df["n_sectors_total"] = float(len(meta_rows))
    for col in CONTINUOUS:
        if col in meta_df.columns and meta_df[col].isna().any():
            fallback = float(means[col]) if hasattr(means, '__getitem__') else float(np.array(means)[list(CONTINUOUS).index(col)])
            meta_df[col] = meta_df[col].fillna(fallback)

    cont   = (meta_df[CONTINUOUS] - means) / stds
    cam_oh = pd.get_dummies(meta_df["cam"].astype(int), prefix="cam").reindex(columns=CAM_COLS_OHE, fill_value=0)
    ccd_oh = pd.get_dummies(meta_df["ccd"].astype(int), prefix="ccd").reindex(columns=CCD_COLS_OHE, fill_value=0)
    C = pd.concat([cont.reset_index(drop=True), cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

    with torch.no_grad():
        samples = flow(torch.tensor(C)).sample((500,)).squeeze(-1)
        pred_z  = samples.mean(0).numpy()
        pred_z_std = samples.std(0).numpy()

    pred_raw = pred_z * y_std + y_mean
    weight   = 1.0 / (1.0 + 5.0 * pred_z_std * y_std)
    pred_off = 1.0 * (1 - weight) + pred_raw * weight

    # Build per-cadence arrays
    sector_data = []
    sector_meds = []
    for lc, meta, off in zip(lcs_valid, meta_rows, pred_off):
        flux = lc.flux.value.astype(float)
        time = lc.time.value.astype(float)
        mask = np.isfinite(flux) & np.isfinite(time)
        flux, time = flux[mask], time[mask]
        if len(flux) == 0:
            continue
        med = float(np.nanmedian(flux))
        sector_meds.append(med)
        sector_data.append((time, flux, med, off, int(meta["cam"])))

    global_ref = float(np.mean(sector_meds))
    raw_sec  = np.array([med/global_ref for _, _, med, _, _ in sector_data])
    cor_sec  = np.array([(med/off)/global_ref for _, _, med, off, _ in sector_data])
    scatter_before = raw_sec.std() * 100
    scatter_after  = cor_sec.std() * 100
    improv = (scatter_before - scatter_after) / scatter_before * 100

    tmag = meta_df["tmag"].mean()
    print(f"  Tmag {tmag:.1f} · {len(sector_data)} sectors · "
          f"scatter {scatter_before:.2f}% → {scatter_after:.2f}% ({improv:.0f}% reduction)")

    return dict(tic_id=tic_id, tmag=tmag, n_sectors=len(sector_data),
                scatter_before=scatter_before, scatter_after=scatter_after, improv=improv,
                global_ref=global_ref, sector_data=sector_data)


# ── Run all stars ─────────────────────────────────────────────────────────────
results = []
for tic in TARGETS:
    r = run_star(tic)
    if r:
        results.append(r)

if not results:
    print("No data downloaded.")
    exit(1)

# ── Figure ────────────────────────────────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(n, 2, figsize=(16, 3.2 * n), facecolor=SURFACE)
if n == 1:
    axes = [axes]

fig.subplots_adjust(hspace=0.42, wspace=0.06, left=0.07, right=0.98, top=0.93, bottom=0.06)

for row_i, r in enumerate(results):
    ax_b = axes[row_i][0]
    ax_a = axes[row_i][1]

    # Y range: use sector medians ± 4× their std, ignore extreme outlier sectors
    raw_meds = np.array([med/r["global_ref"] for _, _, med, _, _ in r["sector_data"]])
    med_center = np.median(raw_meds)
    med_spread = raw_meds.std()
    ylo = med_center - max(med_spread * 4.5, 0.015)
    yhi = med_center + max(med_spread * 4.5, 0.015)
    pad = (yhi - ylo) * 0.12
    ylo -= pad; yhi += pad

    for ax in (ax_b, ax_a):
        ax.set_facecolor(SURFACE)
        ax.set_ylim(ylo, yhi)
        ax.axhline(1.0, color=GRID, lw=1.0, ls="--", zorder=0)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.yaxis.set_tick_params(labelcolor=MUTED, labelsize=8)
        ax.xaxis.set_tick_params(labelcolor=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)

    # Plot cadence points coloured by camera
    for time, flux, med, off, cam in r["sector_data"]:
        col = CAM_COLS.get(cam, MUTED)
        raw_n = flux / r["global_ref"]
        cor_n = (flux / off) / r["global_ref"]
        ax_b.plot(time, raw_n, "-", lw=0.5, color=col, alpha=0.55, rasterized=True)
        ax_a.plot(time, cor_n, "-", lw=0.5, color=col, alpha=0.55, rasterized=True)

    # Sector median markers
    for time, flux, med, off, cam in r["sector_data"]:
        col = CAM_COLS.get(cam, MUTED)
        t_mid = float(np.median(time))
        ax_b.plot(t_mid, med/r["global_ref"], "D", color=col, ms=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
        ax_a.plot(t_mid, (med/off)/r["global_ref"], "D", color=col, ms=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)

    # Row label
    ax_b.set_ylabel(f"TIC {r['tic_id']}\nTmag {r['tmag']:.1f}", color=INK, fontsize=9,
                    fontweight="600", labelpad=6)
    ax_a.set_ylabel("")
    ax_a.yaxis.set_tick_params(labelleft=False)

    # Scatter annotation
    ax_b.text(0.98, 0.05, f"σ = {r['scatter_before']:.2f}%", transform=ax_b.transAxes,
              fontsize=9, color=MUTED, ha="right", va="bottom")
    ax_a.text(0.98, 0.05, f"σ = {r['scatter_after']:.2f}%  (−{r['improv']:.0f}%)",
              transform=ax_a.transAxes, fontsize=9, color=C_AFTER, ha="right", va="bottom",
              fontweight="600")

    if row_i == 0:
        ax_b.set_title("Before STITCH", color=INK, fontsize=11, fontweight="600", pad=6)
        ax_a.set_title("After STITCH", color=INK, fontsize=11, fontweight="600", pad=6)

    if row_i == n - 1:
        ax_b.set_xlabel("BTJD (days)", color=MUTED, fontsize=9)
        ax_a.set_xlabel("BTJD (days)", color=MUTED, fontsize=9)

# Camera legend at top right
cam_handles = [Line2D([0],[0], color=c, lw=2.5, label=f"Cam {k}") for k,c in CAM_COLS.items()]
fig.legend(handles=cam_handles, loc="upper right", fontsize=8, ncol=4,
           framealpha=0.9, edgecolor=GRID,
           bbox_to_anchor=(0.98, 0.99))

fig.suptitle("STITCH — before vs. after correction · 2-min SPOC PDCSAP light curves",
             color=INK, fontsize=12, fontweight="700", x=0.5, y=0.995)

out = "stitch_lightcurve_grid.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"\nSaved → {out}")
