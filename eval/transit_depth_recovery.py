"""
Transit depth recovery — using raw (unnormalized) PDCSAP flux.

STITCH's pred_offset was trained on raw sector medians (e-/s). To apply it
correctly, we need the raw flux, not per-sector normalized flux.

Test:
  1. Download raw PDCSAP lightcurves (no normalize())
  2. Compute sector median in raw units, masking transit windows
  3. Run STITCH to get pred_offset per sector
  4. Corrected flux = raw_flux / pred_offset  (all sectors → same Gaia-calibrated level)
  5. Compare per-sector OOT median scatter: raw vs STITCH
  6. Phase-fold the global lightcurve and measure transit depth

Usage:
  python3 eval/transit_depth_recovery.py stitch_nsf_knn.pt training_data_topup_pdc.parquet
"""

import sys, warnings
import numpy as np
import pandas as pd
import torch, zuko
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightkurve as lk

warnings.filterwarnings("ignore")

MODEL   = next((s for s in sys.argv[1:] if s.endswith(".pt")),      "stitch_nsf_knn.pt")
PARQUET = next((s for s in sys.argv[1:] if s.endswith(".parquet")), "training_data_topup_pdc.parquet")

PLANETS = [
    # (TIC, name, period_days, T0_BTJD, duration_hr, depth_ppm, source)
    (261136679, "pi Men c",   6.267832, 1325.504, 2.03,  890, "Huang+2018"),
    (307210830, "L 98-59 c",  3.690621, 1367.946, 1.05,  660, "Cloutier+2019"),
    (52368076,  "TOI-125 b",  4.654980, 1326.370, 2.15, 1530, "Quinn+2019"),
    (410153553, "LHS 3844 b", 0.462969, 1325.350, 1.05, 4700, "Vanderspek+2019"),
    (117689799, "TOI-700 d", 37.424500, 1325.610, 2.93,  570, "Gilbert+2020"),
]

print(f"Model: {MODEL}  Parquet: {PARQUET}\n")

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

# ── Load quiet-star parquet for spatial context ────────────────────────────────
print("Loading parquet …", flush=True)
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col","row","flux_offset","sector_median"])
df = df[(df["flux_offset"]>0.85)&(df["flux_offset"]<1.15)]
df = df[df["sector_median"]>0]
for col in CONTINUOUS:
    if col in df.columns and df[col].isna().any():
        df[col]=df[col].fillna(df[col].median())
df = df.reset_index(drop=True)
print(f"  {df['tic_id'].nunique():,} quiet stars, sectors {df['sector'].min()}–{df['sector'].max()}")

def transit_mask(t, t0, period, duration_hr):
    dur_days = duration_hr / 24.0
    phase = ((t - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)
    return np.abs(phase) < (0.5 * dur_days / period)

def add_spatial_context(vdf):
    vdf = vdf.copy()
    ccd_means = (df.groupby(["sector","cam","ccd"])["flux_offset_loo"]
                   .mean().rename("sector_ccd_mean_loo").reset_index())
    vdf = vdf.merge(ccd_means, on=["sector","cam","ccd"], how="left")
    vdf["sector_ccd_mean_loo"] = vdf["sector_ccd_mean_loo"].fillna(1.0)
    _K = 20
    knn_vals = np.full(len(vdf), np.nan)
    for i, row_v in vdf.iterrows():
        mask = ((df["sector"]==row_v["sector"])&(df["cam"]==row_v["cam"])&(df["ccd"]==row_v["ccd"]))
        nbrs = df[mask]
        if len(nbrs) < 2: continue
        coords = nbrs[["col","row"]].values.astype(np.float32)
        qpt = np.array([[row_v["col"], row_v["row"]]], dtype=np.float32)
        if not np.all(np.isfinite(qpt)): continue
        fin = np.all(np.isfinite(coords), axis=1)
        if fin.sum() < 2: continue
        k = min(_K, fin.sum())
        _, nn = cKDTree(coords[fin]).query(qpt, k=k)
        knn_vals[i] = nbrs["flux_offset_loo"].values[fin][nn[0]].mean()
    vdf["spatial_knn_mean_loo"] = knn_vals
    vdf["spatial_knn_mean_loo"] = vdf["spatial_knn_mean_loo"].fillna(vdf["sector_ccd_mean_loo"])
    vdf["perstar_gaia_offset"] = 1.0
    return vdf

def run_stitch(vdf):
    vdf = vdf.copy().reset_index(drop=True)
    for col in CONTINUOUS:
        if col not in vdf.columns:
            vdf[col] = df[col].median() if col in df.columns else 0.0
        vdf[col] = pd.to_numeric(vdf[col], errors="coerce").fillna(
            df[col].median() if col in df.columns else 0.0)
    cont   = (vdf[CONTINUOUS].reindex(columns=CONTINUOUS,fill_value=0)-means)/stds
    cam_oh = pd.get_dummies(vdf["cam"].astype(int),prefix="cam").reindex(columns=CAM_COLS,fill_value=0)
    ccd_oh = pd.get_dummies(vdf["ccd"].astype(int),prefix="ccd").reindex(columns=CCD_COLS,fill_value=0)
    C = torch.tensor(
        pd.concat([cont.reset_index(drop=True),cam_oh.reset_index(drop=True),
                   ccd_oh.reset_index(drop=True)],axis=1).values.astype("float32")).to(device)
    with torch.no_grad():
        samp=flow(C).sample((300,)).squeeze(-1)
        mu_z=samp.mean(0).cpu().numpy(); sig_z=samp.std(0).cpu().numpy()
    pred_raw=mu_z*y_std+y_mean; pred_std=sig_z*y_std
    weight=1.0/(1.0+5.0*pred_std)
    vdf["pred_offset"]=weight*pred_raw+(1-weight)*1.0
    vdf["weight"]=weight; vdf["pred_std"]=pred_std
    return vdf

def folded_depth(t, f, t0, period, dur_hr, nbins=150):
    dur_days = dur_hr/24.0; half_phase = 0.5*dur_days/period
    phase = ((t-t0)%period)/period
    phase = np.where(phase>0.5,phase-1,phase)
    in_mask  = np.abs(phase)<half_phase
    oot_mask = (np.abs(phase)>half_phase)&(np.abs(phase)<5*half_phase)
    if in_mask.sum()<5 or oot_mask.sum()<20: return np.nan, np.nan, phase, f
    f_in=float(np.nanmedian(f[in_mask])); f_oot=float(np.nanmedian(f[oot_mask]))
    depth_ppm=(1-f_in/f_oot)*1e6 if f_oot>0 else np.nan
    oot_std=float(np.nanstd(f[oot_mask])/f_oot*1e6) if f_oot>0 else np.nan
    return depth_ppm, oot_std, phase, f

# ── Main loop ──────────────────────────────────────────────────────────────────
BG,SURF,TEXT,MUT,DIM="#0b0f1a","#0f1826","#c4d4ee","#7a9cc4","#1c2d45"
C_RAW,C_ST="#ef4444","#4f8ef7"
results = []

for tic, name, period, t0, dur_hr, pub_ppm, source in PLANETS:
    print(f"── {name} (TIC {tic}, P={period}d, depth={pub_ppm} ppm) ──")
    try:
        search = lk.search_lightcurve(f"TIC {tic}", mission="TESS",
                                      author="SPOC", exptime=120)
        if len(search)==0: print("  No data\n"); continue
        lcs = search.download_all(quality_bitmask="default")
    except Exception as e:
        print(f"  Download failed: {e}\n"); continue

    sector_rows = []; lc_data = []
    for lc in lcs:
        try:
            # Raw flux — no normalize()
            t_arr = np.array(lc.time.btjd, dtype=float)
            f_arr = np.array(lc.flux.value, dtype=float)
            good  = np.isfinite(t_arr) & np.isfinite(f_arr) & (f_arr>0)
            t_arr, f_arr = t_arr[good], f_arr[good]
            if len(t_arr)<100: continue

            sector=int(lc.meta.get("SECTOR",-1))
            cam   =int(lc.meta.get("CAMERA",-1))
            ccd   =int(lc.meta.get("CCD",  -1))
            col   =float(lc.meta.get("COLUMN",np.nan))
            row   =float(lc.meta.get("ROW",  np.nan))

            # Sector median excluding transit windows
            tmask = transit_mask(t_arr, t0, period, dur_hr)
            oot   = f_arr[~tmask]
            sect_med = float(np.nanmedian(oot)) if len(oot)>10 else float(np.nanmedian(f_arr))

            sector_rows.append(dict(
                tic_id=tic, sector=sector, cam=cam, ccd=ccd, col=col, row=row,
                tmag      =float(lc.meta.get("TESSMAG", np.nan)),
                gaiarp    =float(lc.meta.get("GAIARP",  np.nan)),
                crowdsap  =float(lc.meta.get("CROWDSAP",1.0)),
                cdpp1_0   =float(lc.meta.get("CDPP1_0", np.nan)),
                pdcvar    =float(lc.meta.get("PDCVAR",  np.nan)),
                jitter_rms=float(lc.meta.get("JITTRMS", np.nan)),
                pdc_noi   =float(lc.meta.get("PDC_NOI", np.nan)),
                pr_wght2  =float(lc.meta.get("PRWGHT2", np.nan)),
                delta_sub_col=0.0, delta_sub_row=0.0,
                sector_median=sect_med,
            ))
            lc_data.append((sector, t_arr, f_arr, sect_med))
        except: continue

    if len(sector_rows)<2: print("  Too few sectors\n"); continue
    sdf = pd.DataFrame(sector_rows)
    print(f"  {len(sdf)} sectors, raw flux range: "
          f"{np.nanmedian([r[3] for r in lc_data]):.0f} e-/s")

    sdf = add_spatial_context(sdf)
    sdf = run_stitch(sdf)
    pred_map = dict(zip(sdf["sector"], sdf["pred_offset"]))
    wt_map   = dict(zip(sdf["sector"], sdf["weight"]))
    ps_map   = dict(zip(sdf["sector"], sdf["pred_std"]))

    print(f"  pred_offset: {sdf['pred_offset'].min():.4f} – {sdf['pred_offset'].max():.4f}"
          f"  (mean weight {sdf['weight'].mean():.3f})")

    # Per-sector OOT medians — before and after STITCH
    sect_nums, raw_meds, stitch_meds = [], [], []
    for sector, t_arr, f_arr, sect_med in lc_data:
        po = pred_map.get(sector, 1.0)
        tmask = transit_mask(t_arr, t0, period, dur_hr)
        oot_raw = f_arr[~tmask]
        raw_med_oot = float(np.nanmedian(oot_raw))
        stitch_med_oot = raw_med_oot / po
        sect_nums.append(sector)
        raw_meds.append(raw_med_oot)
        stitch_meds.append(stitch_med_oot)

    raw_meds    = np.array(raw_meds,    dtype=float)
    stitch_meds = np.array(stitch_meds, dtype=float)
    raw_cv    = raw_meds.std()    / raw_meds.mean()    * 100
    stitch_cv = stitch_meds.std() / stitch_meds.mean() * 100
    print(f"  OOT median CV:  raw={raw_cv:.2f}%  STITCH={stitch_cv:.2f}%")

    # Phase fold with per-sector OOT normalization to remove cross-sector baseline offsets.
    # Both raw and STITCH should give the same depth (STITCH scalar cancels in ratio).
    # This confirms depth preservation; the cross-sector CV is the meaningful STITCH test.
    t_all=[]; f_all_raw_n=[]; f_all_st_n=[]
    for sector, t_arr, f_arr, sect_med in lc_data:
        po = pred_map.get(sector, 1.0)
        tmask = transit_mask(t_arr, t0, period, dur_hr)
        oot_raw = float(np.nanmedian(f_arr[~tmask])) if (~tmask).sum()>10 else float(np.nanmedian(f_arr))
        oot_st  = oot_raw / po
        t_all.append(t_arr)
        f_all_raw_n.append(f_arr / oot_raw)          # per-sector OOT-normalised raw
        f_all_st_n.append((f_arr / po) / oot_st)     # per-sector OOT-normalised STITCH

    t_ph = np.concatenate(t_all)
    f_raw_n = np.concatenate(f_all_raw_n)
    f_st_n  = np.concatenate(f_all_st_n)

    d_raw, sc_raw, ph_raw, _ = folded_depth(t_ph, f_raw_n, t0, period, dur_hr)
    d_st,  sc_st,  ph_st,  _ = folded_depth(t_ph, f_st_n,  t0, period, dur_hr)
    ph_raw = ph_st = ((t_ph - t0) % period) / period
    ph_raw = np.where(ph_raw > 0.5, ph_raw - 1, ph_raw)
    ph_st  = ph_raw.copy()
    print(f"  Folded depth:   raw={d_raw:.0f} ppm (OOT σ={sc_raw:.0f})  "
          f"STITCH={d_st:.0f} ppm (OOT σ={sc_st:.0f})  pub={pub_ppm} ppm")

    results.append(dict(
        name=name, pub_ppm=pub_ppm, source=source, period=period,
        t0=t0, dur_hr=dur_hr, n_sectors=len(sdf),
        raw_cv=raw_cv, stitch_cv=stitch_cv,
        d_raw=d_raw, sc_raw=sc_raw, d_st=d_st, sc_st=sc_st,
        sect_nums=sect_nums, raw_meds=raw_meds, stitch_meds=stitch_meds,
        t_ph=t_ph, f_raw_n=f_raw_n, ph_raw=ph_raw,
        f_st_n=f_st_n, ph_st=ph_st,
    ))
    print()

if not results:
    print("No results."); sys.exit(1)

# ── Figure ─────────────────────────────────────────────────────────────────────
# 2 columns per planet: OOT medians | phase-folded transit (per-sector normalised)
n = len(results)
fig = plt.figure(figsize=(12, 4*n+2), facecolor=BG)
outer = fig.add_gridspec(n, 2, hspace=0.50, wspace=0.35)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    ax.set_xlabel(xlabel, color=MUT, fontsize=8)
    ax.set_ylabel(ylabel, color=MUT, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=9, pad=5)
    ax.tick_params(colors=MUT, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.yaxis.grid(True, color=DIM, lw=0.3, zorder=0); ax.set_axisbelow(True)

def bin_fold(ph, fl, dur_hr_v, period_v, nbins=120):
    dur_phase = 8 * (dur_hr_v/24) / period_v
    mask = np.abs(ph) < dur_phase
    if mask.sum() < 10: return np.array([]), np.array([]), np.array([])
    ph_m, fl_m = ph[mask], fl[mask]
    edges = np.linspace(ph_m.min(), ph_m.max(), nbins+1)
    bc, bm, be = [], [], []
    for i in range(nbins):
        m = (ph_m>=edges[i])&(ph_m<edges[i+1])
        if m.sum()<3: continue
        bc.append(0.5*(edges[i]+edges[i+1]))
        bm.append(np.nanmedian(fl_m[m]))
        be.append(np.nanstd(fl_m[m])/np.sqrt(m.sum()))
    return np.array(bc), np.array(bm), np.array(be)

for row_i, r in enumerate(results):
    ax0 = fig.add_subplot(outer[row_i, 0])
    ax1 = fig.add_subplot(outer[row_i, 1])

    # Panel 0: per-sector OOT medians (normalised to mean=1)
    sn = np.array(r["sect_nums"])
    rm = r["raw_meds"]    / r["raw_meds"].mean()
    sm = r["stitch_meds"] / r["stitch_meds"].mean()
    improvement = r["raw_cv"] - r["stitch_cv"]
    imp_str = f"{'▼' if improvement>0 else '▲'}{abs(improvement):.2f}%"
    ax0.scatter(sn, rm, color=C_RAW, s=28, zorder=3, label=f"Raw  σ={r['raw_cv']:.2f}%")
    ax0.scatter(sn, sm, color=C_ST,  s=28, zorder=4, marker="D",
                label=f"STITCH  σ={r['stitch_cv']:.2f}%  {imp_str}")
    ax0.axhline(1, color=DIM, lw=0.7, ls="--")
    ax0.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
    style(ax0, "TESS sector", "Norm. OOT median",
          f"{r['name']}  [{r['n_sectors']} sectors]\nCross-sector OOT consistency")

    # Panel 1: phase-folded transit (per-sector OOT-normalised raw; raw=STITCH since scalar cancels)
    ph = r["ph_raw"]
    fl = r["f_raw_n"]
    dur_phase = (r["dur_hr"]/24) / r["period"]
    win = 8*dur_phase
    mask = np.abs(ph)<win
    ax1.scatter(ph[mask], fl[mask], c=C_RAW, s=0.5, alpha=0.07, rasterized=True)
    bc, bm, be = bin_fold(ph, fl, r["dur_hr"], r["period"])
    if len(bc):
        ax1.plot(bc, bm, color=C_RAW, lw=1.5, zorder=5, label=f"Meas. {r['d_raw']:.0f} ppm")
    ax1.axhline(1-r["pub_ppm"]/1e6, color=TEXT, lw=1.0, ls="--", alpha=0.7,
                label=f"Published {r['pub_ppm']:.0f} ppm")
    ax1.axhline(1.0, color=DIM, lw=0.5, ls=":")
    ax1.axvline(-dur_phase/2, color=DIM, lw=0.6, ls=":", alpha=0.7)
    ax1.axvline( dur_phase/2, color=DIM, lw=0.6, ls=":", alpha=0.7)
    ax1.set_xlim(-win, win)
    margin = max(r["pub_ppm"]*2.5/1e6, 0.006)
    ax1.set_ylim(1-margin, 1+margin*0.3)
    ax1.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT, facecolor="#0f1c30", edgecolor=DIM)
    style(ax1, f"Phase  (P={r['period']:.4f}d)", "Per-sector norm. flux",
          f"Phase-folded transit  OOT σ={r['sc_raw']:.0f} ppm\n({r['source']})")

plt.suptitle(
    "STITCH transit validation  |  raw PDCSAP flux (no global normalize)\n"
    "Left: cross-sector OOT baseline consistency  ·  "
    "Right: phase-folded transit depth (per-sector OOT-normalised; depth identical ±STITCH)",
    color=TEXT, fontsize=9.5, y=1.005)

out = "eval/transit_depth_recovery.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")

# Summary
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
              for r in results]).to_csv("eval/transit_depth_recovery.csv", index=False)
print("Saved → eval/transit_depth_recovery.csv")
print(f"\n{'─'*80}")
print(f"{'Planet':<14} {'Secs':>4} {'Pub':>6} {'RawCV%':>7} {'StCV%':>7} "
      f"{'RawDep':>7} {'StDep':>7} {'OOTraw':>7} {'OOTst':>7}")
print(f"{'─'*80}")
for r in results:
    print(f"{r['name']:<14} {r['n_sectors']:>4} {r['pub_ppm']:>6.0f} "
          f"{r['raw_cv']:>7.2f} {r['stitch_cv']:>7.2f} "
          f"{r['d_raw']:>7.0f} {r['d_st']:>7.0f} "
          f"{r['sc_raw']:>7.0f} {r['sc_st']:>7.0f}")
