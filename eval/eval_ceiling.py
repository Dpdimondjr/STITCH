"""
Ceiling analysis for STITCH: how close are we to the best possible correction
given the LOO label framework?

Computes:
  - Oracle CV  : within-star CV when using true LOO labels as prediction
  - Model CV   : our best model (stitch_nsf_knn.pt)
  - Model efficiency : (raw - model) / (raw - oracle)
  - Breakdown by n_sectors, tmag, camera
  - Noise budget chart

Usage (from STITCH root):
  python3 eval/eval_ceiling.py
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

PARQUET   = next((a for a in sys.argv[1:] if a.endswith(".parquet")), "training_data_topup_pdc.parquet")
MODEL_KNN = "stitch_nsf_knn.pt"
OUT       = "stitch_ceiling_report.html"
MIN_SECTS = 4
TMAG_MAX  = 13.0
N_SAMPLES = 200

# ── Load & clean ───────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "flux_offset_loo"])
df = df[(df["flux_offset"]     > 0.85) & (df["flux_offset"]     < 1.15)]
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTS]
df = df[df["tmag"] <= TMAG_MAX]

if "gaiarp" in df.columns:
    df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")

df = df.reset_index(drop=True)
_grp = df.groupby(["sector", "cam", "ccd"])["flux_offset_loo"]
df["sector_ccd_mean_loo"] = (
    (_grp.transform("sum") - df["flux_offset_loo"]) /
    (_grp.transform("count") - 1).clip(lower=1)
)

print("Computing KNN feature …", flush=True)
_K, _knn = 20, np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2: continue
    _xy = _g[["col", "row"]].values.astype(np.float32)
    _v  = _g["flux_offset_loo"].values
    _k  = min(_K, _n - 1)
    _, _nn = cKDTree(_xy).query(_xy, k=_k + 1)
    _knn[_g.index.values] = _v[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = np.where(np.isnan(_knn), df["sector_ccd_mean_loo"], _knn)

for col in ["col", "row", "delta_sub_col", "delta_sub_row", "sector", "tmag",
            "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms", "pdc_noi", "pr_wght2"]:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Split ─────────────────────────────────────────────────────────────────────
sc = (df.groupby("tic_id")["cam"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index().rename(columns={"cam": "dom"}))
tr, tmp = train_test_split(sc["tic_id"], test_size=0.2, stratify=sc["dom"], random_state=42)
tmp_cam = sc[sc["tic_id"].isin(tmp)]["dom"]
vl, te  = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

test_df = df[df["tic_id"].isin(te)].copy().reset_index(drop=True)
print(f"  Test: {test_df['tic_id'].nunique():,} stars")

# ── Inference ─────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

def make_context(sdf, means, stds, cont_cols, cam_cols, ccd_cols):
    cont   = (sdf[cont_cols] - means) / stds
    cam_oh = pd.get_dummies(sdf["cam"].astype(int), prefix="cam").reindex(columns=cam_cols, fill_value=0)
    ccd_oh = pd.get_dummies(sdf["ccd"].astype(int), prefix="ccd").reindex(columns=ccd_cols, fill_value=0)
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

print(f"Running {MODEL_KNN} …", flush=True)
ck   = torch.load(MODEL_KNN, map_location="cpu", weights_only=False)
flow = zuko.flows.NSF(**ck["flow_config"])
flow.load_state_dict(ck["model_state"]); flow.to(device).eval()
C = torch.tensor(make_context(test_df, ck["means"], ck["stds"],
                               ck["continuous_cols"], ck["cam_cols"], ck["ccd_cols"])).to(device)
with torch.no_grad():
    mu = flow(C).sample((N_SAMPLES,)).squeeze(-1).mean(0).cpu().numpy()
pred_knn = mu * ck["y_std"] + ck["y_mean"]

# ── Build corrected columns ────────────────────────────────────────────────────
test_df["_raw"]    = test_df["sector_median"]
# Oracle: use true LOO label as prediction
# oracle_corrected_k = sector_median_k / flux_offset_loo_k = LOO mean of other sectors
test_df["_oracle"] = test_df["sector_median"] / test_df["flux_offset_loo"]
test_df["_knn"]    = test_df["sector_median"] / pred_knn

# ── Within-star CV ─────────────────────────────────────────────────────────────
def cv_stars(df, col, min_n=2):
    return df.groupby("tic_id")[col].apply(
        lambda x: float(x.std() / x.mean()) if len(x) >= min_n else np.nan)

cv_raw    = cv_stars(test_df, "_raw")
cv_oracle = cv_stars(test_df, "_oracle")
cv_knn    = cv_stars(test_df, "_knn")

common = cv_raw.dropna().index.intersection(cv_oracle.dropna().index).intersection(cv_knn.dropna().index)
cv_raw    = cv_raw[common]
cv_oracle = cv_oracle[common]
cv_knn    = cv_knn[common]

med_raw    = float(cv_raw.median())    * 100
med_oracle = float(cv_oracle.median()) * 100
med_knn    = float(cv_knn.median())    * 100

print(f"\n  Raw   : {med_raw:.4f}%")
print(f"  Oracle: {med_oracle:.4f}%  (LOO-label ceiling)")
print(f"  KNN   : {med_knn:.4f}%")

# Model efficiency = fraction of oracle-recoverable variance actually removed
efficiency_per_star = (cv_raw - cv_knn) / (cv_raw - cv_oracle).clip(lower=1e-6)
efficiency_per_star = efficiency_per_star.clip(0, 2)   # clip outliers for display
med_eff = float(efficiency_per_star.median()) * 100
print(f"\n  Median model efficiency: {med_eff:.1f}%")
print(f"  (fraction of oracle-recoverable CV reduction actually achieved)")

# What drives oracle residual? Larger for fewer sectors.
nsec = test_df.groupby("tic_id")["sector"].nunique()
nsec_common = nsec[common]
print(f"\n  Stars with 4-6  sectors: oracle {cv_oracle[nsec_common.between(4,6)].median()*100:.4f}%")
print(f"  Stars with 7-10 sectors: oracle {cv_oracle[nsec_common.between(7,10)].median()*100:.4f}%")
print(f"  Stars with 11+  sectors: oracle {cv_oracle[nsec_common>=11].median()*100:.4f}%")

# ── Breakdown tables ───────────────────────────────────────────────────────────
def breakdown(mask_ser, n_bin_name):
    rows = []
    for label, (lo, hi) in n_bin_name:
        m = mask_ser.between(lo, hi)
        tics = mask_ser[m].index.intersection(common)
        if len(tics) < 5:
            continue
        r = float(cv_raw[tics].median())    * 100
        o = float(cv_oracle[tics].median()) * 100
        k = float(cv_knn[tics].median())    * 100
        eff = float((cv_raw[tics] - cv_knn[tics]).median() /
                    (cv_raw[tics] - cv_oracle[tics]).clip(lower=1e-6).median()) * 100
        rows.append({"label": label, "n": int(len(tics)),
                     "raw": round(r,4), "oracle": round(o,4), "knn": round(k,4),
                     "eff": round(eff,1)})
    return rows

nsec_bd = breakdown(nsec_common, [
    ("4–6 sectors",  (4, 6)),
    ("7–10 sectors", (7, 10)),
    ("11–15 sectors",(11,15)),
    ("16+ sectors",  (16,999)),
])

tmag_star = test_df.groupby("tic_id")["tmag"].mean()
tmag_common = tmag_star[common]
tmag_bd = breakdown(tmag_common, [
    ("Tmag 5–9",   (5, 9)),
    ("Tmag 9–10",  (9, 10)),
    ("Tmag 10–11", (10,11)),
    ("Tmag 11–12", (11,12)),
    ("Tmag 12–13", (12,13)),
])

cam_star = test_df.groupby("tic_id")["cam"].agg(lambda x: int(x.mode()[0]))
cam_common = cam_star[common]
cam_bd = breakdown(cam_common, [(f"Cam {c}", (c,c)) for c in [1,2,3,4]])

# ── Histogram of efficiency per star ──────────────────────────────────────────
eff_pct = efficiency_per_star * 100
bins = np.arange(0, 205, 10)
hist_eff, edges = np.histogram(eff_pct.clip(0, 200), bins=bins)
hist_eff_data = [{"x": int(edges[i]), "n": int(hist_eff[i])} for i in range(len(hist_eff))]

# ── Expected oracle CV from theory: CV_oracle ≈ CV_raw / (N-1) ───────────────
# Generate theory curve
theory_n = list(range(4, 25))
theory_raw_assumed = med_raw  # assume median raw CV for illustration
theory_oracle = [round(theory_raw_assumed / (n - 1), 4) for n in theory_n]
# Observed oracle by n_sectors
obs_oracle_by_n = []
for n in theory_n:
    tics = nsec_common[nsec_common == n].index.intersection(common)
    if len(tics) < 10:
        continue
    obs_oracle_by_n.append({
        "n": n,
        "obs_oracle": round(float(cv_oracle[tics].median()) * 100, 4),
        "theory":     round(theory_raw_assumed / (n - 1), 4),
        "obs_knn":    round(float(cv_knn[tics].median())    * 100, 4),
        "obs_raw":    round(float(cv_raw[tics].median())    * 100, 4),
    })

def _j(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(type(o))

report_data = {
    "med_raw":    round(med_raw,    4),
    "med_oracle": round(med_oracle, 4),
    "med_knn":    round(med_knn,    4),
    "med_eff":    round(med_eff,    1),
    "n_stars":    int(len(common)),
    "nsec_bd":    nsec_bd,
    "tmag_bd":    tmag_bd,
    "cam_bd":     cam_bd,
    "hist_eff":   hist_eff_data,
    "oracle_by_n":obs_oracle_by_n,
    "pct_above_oracle": round(float((cv_knn > cv_oracle).mean()) * 100, 1),
    "pct_below_oracle": round(float((cv_knn <= cv_oracle).mean()) * 100, 1),
}
data_js = json.dumps(report_data, separators=(",", ":"), default=_j)
print(f"\n  Stars where model beats oracle: {report_data['pct_below_oracle']}% (shouldn't happen — label noise)")

# ── HTML report ────────────────────────────────────────────────────────────────
HTML = """\
<style>
:root{
  --bg:#f5f4f1;--bg2:#eceae5;--bg3:#e2e0da;
  --ink:#151410;--ink2:#48473f;--muted:#8a8980;
  --border:#d8d6ce;--accent:#1d6fb5;
  --good:#0a7a45;--warn:#b85c00;--ora:#8b3d00;
  --c-raw:#8a8980;--c-ora:#d97706;--c-knn:#059669;
  --font:system-ui,-apple-system,sans-serif;
  --mono:'SF Mono','Cascadia Code','Fira Code',ui-monospace,monospace;
  --c1:#d95b37;--c2:#1d6fb5;--c3:#16a34a;--c4:#b45309;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#0e1016;--bg2:#161820;--bg3:#1d2029;
  --ink:#d8d6ce;--ink2:#8a8980;--muted:#4a4a42;
  --border:#252830;--accent:#4d9de0;
  --good:#22c55e;--warn:#fb923c;--ora:#f59e0b;
  --c-raw:#6b7280;--c-ora:#f59e0b;--c-knn:#34d399;
}}
:root[data-theme=dark]{
  --bg:#0e1016;--bg2:#161820;--bg3:#1d2029;
  --ink:#d8d6ce;--ink2:#8a8980;--muted:#4a4a42;
  --border:#252830;--accent:#4d9de0;
  --good:#22c55e;--warn:#fb923c;--ora:#f59e0b;
  --c-raw:#6b7280;--c-ora:#f59e0b;--c-knn:#34d399;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:light dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font);font-size:13px;line-height:1.55;min-height:100vh}
.page{max-width:960px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:20px;font-weight:700;letter-spacing:-.03em;margin-bottom:4px}
.subtitle{font-size:12px;color:var(--muted);margin-bottom:28px}
h2{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.hero-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:28px}
.hero-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:14px 18px;flex:1;min-width:130px}
.hero-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.hero-val{font-size:24px;font-weight:700;letter-spacing:-.03em;font-variant-numeric:tabular-nums;margin-top:2px}
.hero-sub{font-size:11px;color:var(--muted);margin-top:2px}
.good{color:var(--good)} .warn{color:var(--warn)} .ora{color:var(--ora)}
.tbl-wrap{overflow-x:auto;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--bg2);padding:6px 10px;text-align:left;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--border);
  white-space:nowrap}
thead th.r{text-align:right}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--bg2)}
td{padding:5px 10px;vertical-align:middle;white-space:nowrap;font-variant-numeric:tabular-nums}
td.r{text-align:right} td.mono{font-family:var(--mono);font-size:11px}
.best{font-weight:700;color:var(--good)}
.pill{display:inline-block;font-size:10px;font-weight:600;padding:1px 7px;border-radius:8px;font-variant-numeric:tabular-nums}
.pill-g{background:rgba(10,122,69,.15);color:var(--good)}
.pill-a{background:rgba(217,119,6,.15);color:var(--ora)}
.chart-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:16px;margin-bottom:8px;overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:11px;margin:8px 0}
.leg{display:flex;align-items:center;gap:5px}
.lsq{width:12px;height:5px;border-radius:2px}
.callout{background:var(--bg2);border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
  padding:12px 16px;margin:16px 0;font-size:12px;line-height:1.6}
.callout b{color:var(--ink)}
.note{font-size:11px;color:var(--muted);margin-top:6px;line-height:1.5}
</style>

<div class="page">
  <h1>STITCH · Ceiling Analysis</h1>
  <div class="subtitle">How close is the KNN model to the best possible LOO-based correction?</div>

  <div class="hero-row" id="heroes"></div>

  <div class="callout" id="callout-top"></div>

  <h2>Noise Budget</h2>
  <div class="legend">
    <div class="leg"><div class="lsq" style="background:var(--c-raw)"></div>Raw (no correction)</div>
    <div class="leg"><div class="lsq" style="background:var(--c-ora)"></div>Oracle ceiling (LOO label as perfect predictor)</div>
    <div class="leg"><div class="lsq" style="background:var(--c-knn)"></div>KNN model</div>
  </div>
  <div class="chart-wrap" id="budget-chart"></div>
  <p class="note">Oracle ceiling = using the true LOO flux offset as the prediction (best any model could do given these labels). Residual oracle CV comes from label noise (LOO estimator variance, which shrinks with more sectors) and intrinsic stellar variability that the LOO framework captures but can't remove.</p>

  <h2>Oracle CV vs N sectors</h2>
  <div class="chart-wrap" id="nsec-chart"></div>
  <p class="note">The oracle ceiling improves predictably with the number of observed sectors because the LOO mean becomes a better estimate of the true mean. Theory line: CV<sub>oracle</sub> ≈ CV<sub>raw</sub> / (N−1). Stars below the theoretical line are genuinely variable; stars above it are well-behaved.</p>

  <h2>Model Efficiency by Group</h2>
  <p class="note" style="margin-bottom:12px">Efficiency = (CV<sub>raw</sub> − CV<sub>KNN</sub>) / (CV<sub>raw</sub> − CV<sub>oracle</sub>). 100% = model perfectly reaches oracle ceiling. Values above 100% are possible where the model overshoots (pushes CV below the oracle, usually due to label noise).</p>
  <div style="display:flex;gap:20px;flex-wrap:wrap">
    <div style="flex:1;min-width:240px">
      <div class="tbl-wrap"><table id="nsec-table"></table></div>
    </div>
    <div style="flex:1;min-width:240px">
      <div class="tbl-wrap"><table id="tmag-table"></table></div>
    </div>
    <div style="flex:1;min-width:200px">
      <div class="tbl-wrap"><table id="cam-table"></table></div>
    </div>
  </div>

  <h2>Per-Star Efficiency Distribution</h2>
  <div class="chart-wrap" id="eff-hist"></div>
  <p class="note" id="eff-note"></p>
</div>

<script>
const D = DATA_PLACEHOLDER;
const CAM_C = ['','#d95b37','#1d6fb5','#16a34a','#b45309'];

// ── Hero cards ────────────────────────────────────────────────────────────────
const gap_model  = D.med_raw - D.med_knn;
const gap_oracle = D.med_raw - D.med_oracle;
const ceiling_gap = D.med_knn - D.med_oracle;

document.getElementById('heroes').innerHTML = [
  {label:'Raw Median CV',    val: D.med_raw.toFixed(3)+'%',    sub:'before any correction',    cls:'warn'},
  {label:'Oracle Ceiling',   val: D.med_oracle.toFixed(3)+'%', sub:'LOO-label floor',           cls:'ora'},
  {label:'KNN Model CV',     val: D.med_knn.toFixed(3)+'%',    sub:'best current model',        cls:'good'},
  {label:'Model Efficiency', val: D.med_eff.toFixed(0)+'%',    sub:'of oracle gap closed',      cls:'good'},
  {label:'Gap to Oracle',    val: ceiling_gap.toFixed(3)+'%',  sub:'remaining improvement room',cls:''},
  {label:'Test Stars',       val: D.n_stars.toLocaleString(),   sub:'held-out set',              cls:''},
].map(h=>`<div class="hero-card">
  <div class="hero-label">${h.label}</div>
  <div class="hero-val ${h.cls}">${h.val}</div>
  <div class="hero-sub">${h.sub}</div>
</div>`).join('');

document.getElementById('callout-top').innerHTML =
  `The KNN model closes <b>${D.med_eff.toFixed(0)}%</b> of the gap between raw and oracle.
  Of the ${gap_oracle.toFixed(3)}% CV that's recoverable with perfect LOO-label prediction,
  the model captures ${gap_model.toFixed(3)}%, leaving ${ceiling_gap.toFixed(3)}% on the table.
  The oracle residual (${D.med_oracle.toFixed(3)}%) is label noise that shrinks as stars accumulate more sectors —
  it is not a fixed physical floor.`;

// ── Noise budget chart (horizontal stack bars) ────────────────────────────────
(function(){
  const groups = [
    {label: 'Raw',    vals:[{v:D.med_raw,   c:'var(--c-raw)'}]},
    {label: 'Oracle', vals:[{v:D.med_oracle,c:'var(--c-ora)'}]},
    {label: 'KNN',    vals:[{v:D.med_knn,   c:'var(--c-knn)'}]},
  ];
  const W=700, rowH=52, pad={l:80,r:120,t:12,b:16};
  const n = groups.length;
  const H = n*rowH + pad.t + pad.b;
  const pw = W - pad.l - pad.r;
  const maxV = D.med_raw * 1.05;
  function xp(v){ return pad.l + (v/maxV)*pw; }

  let s = `<svg width="100%" viewBox="0 0 ${W} ${H}" style="display:block">`;

  // Vertical gridlines
  [0, 0.5, 1.0, 1.5, D.med_raw].forEach(v => {
    const x = xp(v).toFixed(1);
    s += `<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H-pad.b}" stroke="currentColor" stroke-width="0.5" opacity="0.1"/>`;
    s += `<text x="${x}" y="${H}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.45">${v.toFixed(1)}%</text>`;
  });

  // Bars
  groups.forEach((g, i) => {
    const y0 = pad.t + i*rowH + 10;
    const bh = rowH - 20;
    const isBold = g.label === 'KNN';
    s += `<text x="${pad.l-8}" y="${y0+bh/2+4}" text-anchor="end" font-size="${isBold?12:11}" font-weight="${isBold?700:400}" fill="currentColor" opacity="${isBold?1:0.7}">${g.label}</text>`;

    // Draw each value segment
    let cx = pad.l;
    g.vals.forEach(seg => {
      const bw = (seg.v / maxV) * pw;
      s += `<rect x="${cx.toFixed(1)}" y="${y0}" width="${bw.toFixed(1)}" height="${bh}" fill="${seg.c}" rx="3" opacity="0.9"/>`;
      cx += bw;
    });

    // Value label
    const totalV = g.vals.reduce((a,b)=>a+b.v, 0);
    s += `<text x="${(xp(totalV)+8).toFixed(1)}" y="${y0+bh/2+4}" font-size="12" font-weight="${isBold?700:400}" fill="${g.vals[0].c}">${totalV.toFixed(3)}%</text>`;
  });

  // Bracket showing gaps
  // Gap: raw -> oracle (oracle residual floor)
  const yMid = pad.t + 3*rowH/2;
  const xRaw = xp(D.med_raw); const xOra = xp(D.med_oracle); const xKnn = xp(D.med_knn);

  s += '</svg>';
  document.getElementById('budget-chart').innerHTML = s;
})();

// ── Oracle CV vs N sectors chart ───────────────────────────────────────────────
(function(){
  if(!D.oracle_by_n || D.oracle_by_n.length < 2) return;
  const pts = D.oracle_by_n;
  const W=680, H=200, pad={l:45,r:20,t:20,b:35};
  const nMin = Math.min(...pts.map(p=>p.n));
  const nMax = Math.max(...pts.map(p=>p.n));
  const yMax = Math.max(...pts.map(p=>p.obs_raw), D.med_raw) * 1.1;
  const yMin = 0;
  function xp(n){ return pad.l + (n-nMin)/(nMax-nMin) * (W-pad.l-pad.r); }
  function yp(v){ return pad.t + (1-(v-yMin)/(yMax-yMin)) * (H-pad.t-pad.b); }

  let s = `<svg width="100%" viewBox="0 0 ${W} ${H}" style="display:block">`;
  // Grid
  [0,0.5,1.0,1.5,2.0].forEach(v=>{
    if(v>yMax) return;
    const y=yp(v).toFixed(1);
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="currentColor" stroke-width="0.5" opacity="0.1"/>`;
    s+=`<text x="${pad.l-4}" y="${parseFloat(y)+3}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.45">${v.toFixed(1)}%</text>`;
  });
  for(let n=nMin; n<=nMax; n+=2){
    const x=xp(n).toFixed(1);
    s+=`<text x="${x}" y="${H-pad.b+13}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.45">${n}</text>`;
  }
  s+=`<text x="${(pad.l+W-pad.r)/2}" y="${H-pad.b+27}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.35">N sectors observed</text>`;

  // Lines
  function line(key, col, dash=''){
    const d=pts.map((p,i)=>`${i?'L':'M'}${xp(p.n).toFixed(1)},${yp(p[key]).toFixed(1)}`).join('');
    s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="1.8" stroke-dasharray="${dash}" opacity="0.9"/>`;
    pts.forEach(p=>{
      s+=`<circle cx="${xp(p.n).toFixed(1)}" cy="${yp(p[key]).toFixed(1)}" r="2.5" fill="${col}" opacity="0.8"/>`;
    });
  }

  line('obs_raw',    'var(--c-raw)');
  line('obs_oracle', 'var(--c-ora)');
  line('theory',     'var(--c-ora)', '4 3');
  line('obs_knn',    'var(--c-knn)');

  // Legend
  [{k:'obs_raw','l':'Raw',c:'var(--c-raw)'},{k:'obs_oracle','l':'Oracle (observed)',c:'var(--c-ora)'},
   {k:'theory','l':'Oracle (theory: CVraw/(N-1))',c:'var(--c-ora)',d:'4 3'},{k:'obs_knn','l':'KNN',c:'var(--c-knn)'}
  ].forEach((leg,i)=>{
    const lx = pad.l + i*160;
    if(lx > W-80) return;
    s+=`<line x1="${lx}" y1="10" x2="${lx+16}" y2="10" stroke="${leg.c}" stroke-width="2" stroke-dasharray="${leg.d||''}"/>`;
    s+=`<text x="${lx+20}" y="13.5" font-size="9" fill="currentColor" opacity="0.7">${leg.l}</text>`;
  });

  s+='</svg>';
  document.getElementById('nsec-chart').innerHTML = s;
})();

// ── Breakdown tables ──────────────────────────────────────────────────────────
function mkTable(data, rowLabel, containerId){
  const hdr=`<thead><tr><th>${rowLabel}</th><th class="r">N</th>
    <th class="r" style="color:var(--c-raw)">Raw CV</th>
    <th class="r" style="color:var(--c-ora)">Oracle</th>
    <th class="r" style="color:var(--c-knn)">KNN</th>
    <th class="r">Efficiency</th></tr></thead>`;
  const body=data.map(r=>{
    const effCls = r.eff>=70?'pill-g':'pill-a';
    return`<tr>
      <td>${r.label}</td>
      <td class="r">${r.n.toLocaleString()}</td>
      <td class="r mono" style="color:var(--c-raw)">${r.raw.toFixed(3)}%</td>
      <td class="r mono" style="color:var(--c-ora)">${r.oracle.toFixed(3)}%</td>
      <td class="r mono" style="color:var(--c-knn)">${r.knn.toFixed(3)}%</td>
      <td class="r"><span class="pill ${effCls}">${r.eff.toFixed(0)}%</span></td>
    </tr>`;
  }).join('');
  document.getElementById(containerId).innerHTML=hdr+'<tbody>'+body+'</tbody>';
}
mkTable(D.nsec_bd, 'Sectors observed', 'nsec-table');
mkTable(D.tmag_bd, 'Brightness',        'tmag-table');
mkTable(D.cam_bd,  'Camera',            'cam-table');

// ── Efficiency histogram ──────────────────────────────────────────────────────
(function(){
  const H = D.hist_eff;
  const W=680, pH=180, pad={l:40,r:20,t:12,b:38};
  const pw=W-pad.l-pad.r, ph=pH-pad.t-pad.b;
  const maxN = Math.max(...H.map(b=>b.n));
  const xMin=H[0].x, xMax=H[H.length-1].x+10;
  function xp(v){ return pad.l + (v-xMin)/(xMax-xMin)*pw; }
  function yp(v){ return pad.t + (1-v/maxN)*ph; }
  const barW = pw/H.length - 1;

  let s=`<svg width="100%" viewBox="0 0 ${W} ${pH}" style="display:block">`;
  // Grid
  [0,0.25,0.5,0.75,1.0].forEach(f=>{
    const y=yp(f*maxN).toFixed(1);
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="currentColor" stroke-width="0.5" opacity="0.1"/>`;
  });
  // Reference line at 100%
  const x100 = xp(100).toFixed(1);
  s+=`<line x1="${x100}" y1="${pad.t}" x2="${x100}" y2="${pH-pad.b}" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.25"/>`;
  s+=`<text x="${parseFloat(x100)+4}" y="${pad.t+12}" font-size="9" fill="currentColor" opacity="0.5">100%</text>`;

  // Bars
  H.forEach((b,i)=>{
    if(b.n===0) return;
    const x=xp(b.x); const bh=pH-pad.b-yp(b.n);
    const col = b.x >= 100 ? 'var(--c-ora)' : b.x < 50 ? 'var(--c-raw)' : 'var(--c-knn)';
    s+=`<rect x="${x.toFixed(1)}" y="${yp(b.n).toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(1,bh).toFixed(1)}" fill="${col}" rx="1" opacity="0.8"/>`;
  });
  s+=`<line x1="${pad.l}" y1="${pH-pad.b}" x2="${W-pad.r}" y2="${pH-pad.b}" stroke="currentColor" stroke-width="0.5" opacity="0.15"/>`;
  // X labels
  [0,25,50,75,100,125,150,175,200].forEach(v=>{
    if(v<xMin||v>xMax) return;
    s+=`<text x="${xp(v).toFixed(1)}" y="${pH-pad.b+13}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.45">${v}%</text>`;
  });
  s+=`<text x="${(pad.l+W-pad.r)/2}" y="${pH-pad.b+28}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.35">Model efficiency % (fraction of oracle gap closed per star)</text>`;

  s+='</svg>';
  document.getElementById('eff-hist').innerHTML=s;

  const nAbove = D.hist_eff.filter(b=>b.x>=100).reduce((a,b)=>a+b.n,0);
  const tot    = D.hist_eff.reduce((a,b)=>a+b.n,0);
  const pAbove = (nAbove/tot*100).toFixed(1);
  document.getElementById('eff-note').textContent =
    `${pAbove}% of stars have efficiency > 100% — the model pushes CV below the oracle ceiling, `+
    `which happens because the oracle itself uses a noisy LOO mean (especially for low-N stars) `+
    `and our model can sometimes predict the true offset better than a 4-sector LOO average.`;
})();
</script>
"""

HTML = HTML.replace("DATA_PLACEHOLDER", data_js)
with open(OUT, "w") as f:
    f.write(HTML)
print(f"\nWritten → {OUT}  ({len(HTML)/1024:.1f} KB)")
