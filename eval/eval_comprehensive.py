"""
Comprehensive model evaluation for STITCH.
Compares GaiaFeat → NBR → KNN flow models plus two direct-feature baselines.
Generates stitch_eval_report.html.

Usage (from STITCH root):
  python3 eval/eval_comprehensive.py
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

PARQUET  = next((a for a in sys.argv[1:] if a.endswith(".parquet")), "training_data_topup_pdc.parquet")
OUT      = "stitch_eval_report.html"
MIN_SECTS = 4
TMAG_MAX  = 13.0
N_SAMPLES = 200

# Models in progression order (name, path)
MODELS = [
    ("LOO+GaiaFeat",     "stitch_nsf_gaia_feat.pt"),
    ("LOO+GaiaFeat+NBR", "stitch_nsf_nbr.pt"),
    ("LOO+GaiaFeat+KNN", "stitch_nsf_knn.pt"),
]

# ── Load & clean ───────────────────────────────────────────────────────────────
print(f"Loading {PARQUET} …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "flux_offset_loo"])
df = df[(df["flux_offset"]     > 0.85) & (df["flux_offset"]     < 1.15)]
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTS]
df = df[df["tmag"] <= TMAG_MAX]
print(f"  {len(df):,} records  ·  {df['tic_id'].nunique():,} stars")

# ── Feature engineering ────────────────────────────────────────────────────────
if "gaiarp" in df.columns:
    df["perstar_gaia_offset"] = df.groupby("tic_id")["flux_offset"].transform("mean")

df = df.reset_index(drop=True)
_grp = df.groupby(["sector", "cam", "ccd"])["flux_offset_loo"]
df["sector_ccd_mean_loo"] = (
    (_grp.transform("sum") - df["flux_offset_loo"]) /
    (_grp.transform("count") - 1).clip(lower=1)
)

print("Computing spatial KNN feature …", flush=True)
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

CONTINUOUS_BASE = ["col", "row", "delta_sub_col", "delta_sub_row",
                   "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
                   "pdc_noi", "pr_wght2"]
for col in CONTINUOUS_BASE:
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
print(f"  Test: {len(test_df):,} rows  ·  {test_df['tic_id'].nunique():,} stars\n")

# ── Inference helpers ─────────────────────────────────────────────────────────
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

def predict(path, sdf):
    ck   = torch.load(path, map_location="cpu", weights_only=False)
    flow = zuko.flows.NSF(**ck["flow_config"])
    flow.load_state_dict(ck["model_state"]); flow.to(device).eval()
    C = torch.tensor(make_context(sdf, ck["means"], ck["stds"],
                                  ck["continuous_cols"], ck["cam_cols"], ck["ccd_cols"])).to(device)
    with torch.no_grad():
        mu = flow(C).sample((N_SAMPLES,)).squeeze(-1).mean(0).cpu().numpy()
    return mu * ck["y_std"] + ck["y_mean"]

# ── Run models ────────────────────────────────────────────────────────────────
preds = {}
for name, path in MODELS:
    if not os.path.exists(path):
        print(f"  Skipping {name} ({path} missing)"); continue
    print(f"  Running {name} …", flush=True)
    preds[name] = predict(path, test_df)

# ── Build corrected columns ────────────────────────────────────────────────────
# Raw (no correction)
test_df["_raw"] = test_df["sector_median"]
# Zero-param direct baselines
test_df["_ccd_direct"] = test_df["sector_median"] / test_df["sector_ccd_mean_loo"]
test_df["_knn_direct"] = test_df["sector_median"] / test_df["spatial_knn_mean_loo"]
# Flow model corrections
for name, p in preds.items():
    test_df[f"_flow_{name}"] = test_df["sector_median"] / p

ALL_COLS = {
    "Raw (no correction)":  "_raw",
    "CCD-mean direct":      "_ccd_direct",
    "KNN-mean direct":      "_knn_direct",
}
for name in preds:
    ALL_COLS[name] = f"_flow_{name}"

# ── Within-star CV ─────────────────────────────────────────────────────────────
def cv_stars(df, col):
    return df.groupby("tic_id")[col].apply(
        lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan
    )

print("\nComputing within-star CV …")
cvs = {label: cv_stars(test_df, col) for label, col in ALL_COLS.items()}
cv_raw = cvs["Raw (no correction)"]

# ── Metrics per approach ───────────────────────────────────────────────────────
LOO_TRUE = test_df["flux_offset_loo"].values

def summarise(label, col, pred_arr=None):
    cv    = cvs[label]
    valid = cv.dropna()
    med   = float(valid.median()) * 100
    med_r = float(cv_raw.reindex(valid.index).median()) * 100
    red   = float((1 - med / med_r) * 100)
    pct_imp  = float((valid < cv_raw.reindex(valid.index)).mean() * 100)
    pct_hurt = float((valid > cv_raw.reindex(valid.index)).mean() * 100)
    if pred_arr is not None:
        ok  = ~np.isnan(pred_arr) & ~np.isnan(LOO_TRUE)
        mae = float(np.abs(LOO_TRUE[ok] - pred_arr[ok]).mean())
        r   = float(np.corrcoef(LOO_TRUE[ok], pred_arr[ok])[0, 1])
    else:
        mae, r = None, None
    return {"label": label, "med_cv": float(med), "cv_red": red,
            "pct_imp": pct_imp, "pct_hurt": pct_hurt, "mae": mae, "r": r}

rows_summary = []
rows_summary.append(summarise("Raw (no correction)", "_raw", pred_arr=None))
rows_summary.append(summarise("CCD-mean direct",     "_ccd_direct",
                              pred_arr=test_df["sector_ccd_mean_loo"].values))
rows_summary.append(summarise("KNN-mean direct",     "_knn_direct",
                              pred_arr=test_df["spatial_knn_mean_loo"].values))
for name, p in preds.items():
    rows_summary.append(summarise(name, f"_flow_{name}", pred_arr=p))

print("\n  Approach                   Med CV%   ↓vs raw   MAE-LOO   Pearson r   %improved  %hurt")
print("  " + "-"*90)
for r in rows_summary:
    mae_s = f"{r['mae']:.4f}" if r['mae'] is not None else "  n/a "
    r_s   = f"{r['r']:.3f}"  if r['r']   is not None else "  n/a"
    print(f"  {r['label']:<28} {r['med_cv']:>6.3f}%  {r['cv_red']:>6.1f}%  "
          f"{mae_s}   {r_s}       {r['pct_imp']:>5.1f}%     {r['pct_hurt']:>4.1f}%")

# ── Per-camera breakdown ───────────────────────────────────────────────────────
cam_rows = []
for cam in sorted(test_df["cam"].dropna().unique().astype(int)):
    mask = test_df["cam"].values == cam
    sub  = test_df[mask]
    entry = {"cam": cam}
    for label, col in ALL_COLS.items():
        cv = sub.groupby("tic_id")[col].apply(
            lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan
        ).dropna()
        entry[label] = float(cv.median()) * 100
    cam_rows.append(entry)

# ── Per-tmag breakdown ────────────────────────────────────────────────────────
tmag_bins = [(5, 9), (9, 10), (10, 11), (11, 12), (12, 13)]
tmag_rows = []
for lo, hi in tmag_bins:
    mask = (test_df["tmag"] >= lo) & (test_df["tmag"] < hi)
    sub  = test_df[mask]
    if sub["tic_id"].nunique() < 5:
        continue
    entry = {"bin": f"{lo}–{hi}", "n": int(sub["tic_id"].nunique())}
    for label, col in ALL_COLS.items():
        cv = sub.groupby("tic_id")[col].apply(
            lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan
        ).dropna()
        entry[label] = float(cv.median()) * 100 if len(cv) else 0
    tmag_rows.append(entry)

# ── Per-star improvement histogram for best model ────────────────────────────
best_name  = list(preds.keys())[-1]
best_cv    = cvs[best_name]
delta_cv   = (best_cv - cv_raw.reindex(best_cv.index)).dropna() * 100  # positive = hurt
bins       = np.arange(-2.0, 1.05, 0.1)
hist, edges = np.histogram(delta_cv.clip(-2.0, 1.0), bins=bins)
hist_data  = [{"x": round(float(edges[i]), 2), "w": round(float(edges[i+1]-edges[i]), 3),
               "n": int(hist[i])} for i in range(len(hist))]

# ── Per-star n_sectors breakdown ───────────────────────────────────────────────
nsec_star = test_df.groupby("tic_id")["sector"].nunique()
nsec_bins = [(4, 6), (7, 10), (11, 15), (16, 100)]
nsec_rows = []
for lo, hi in nsec_bins:
    tics = nsec_star[(nsec_star >= lo) & (nsec_star <= hi)].index
    sub  = test_df[test_df["tic_id"].isin(tics)]
    if len(tics) < 5: continue
    entry = {"bin": f"{lo}–{hi if hi < 100 else '+'}", "n": int(len(tics))}
    for label, col in ALL_COLS.items():
        cv = sub.groupby("tic_id")[col].apply(
            lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan
        ).dropna()
        entry[label] = float(cv.median()) * 100 if len(cv) else 0
    nsec_rows.append(entry)

# ── Pack everything for the report ─────────────────────────────────────────────
report_data = {
    "summary":   rows_summary,
    "cam":       cam_rows,
    "tmag":      tmag_rows,
    "nsec":      nsec_rows,
    "hist":      hist_data,
    "best_name": best_name,
    "n_stars":   int(test_df["tic_id"].nunique()),
    "raw_med_cv": float(cv_raw.dropna().median()) * 100,
    "best_med_cv": float(cvs[best_name].dropna().median()) * 100,
    "approaches": list(ALL_COLS.keys()),
}
def _j(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(type(o))
data_js = json.dumps(report_data, separators=(",", ":"), default=_j)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """\
<style>
:root{
  --bg:#f5f4f1;--bg2:#eceae5;--bg3:#e2e0da;
  --ink:#151410;--ink2:#48473f;--muted:#8a8980;
  --border:#d8d6ce;--accent:#1d6fb5;
  --good:#0a7a45;--warn:#b85c00;--neu:#48473f;
  --font:system-ui,-apple-system,sans-serif;
  --mono:'SF Mono','Cascadia Code','Fira Code',ui-monospace,monospace;
  --ma:#6d28d9;--mb:#047857;
  --c1:#d95b37;--c2:#1d6fb5;--c3:#16a34a;--c4:#b45309;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#0e1016;--bg2:#161820;--bg3:#1d2029;
  --ink:#d8d6ce;--ink2:#8a8980;--muted:#4a4a42;
  --border:#252830;--accent:#4d9de0;
  --good:#22c55e;--warn:#fb923c;--neu:#8a8980;
  --ma:#a78bfa;--mb:#34d399;
}}
:root[data-theme=dark]{
  --bg:#0e1016;--bg2:#161820;--bg3:#1d2029;
  --ink:#d8d6ce;--ink2:#8a8980;--muted:#4a4a42;
  --border:#252830;--accent:#4d9de0;
  --good:#22c55e;--warn:#fb923c;--neu:#8a8980;
  --ma:#a78bfa;--mb:#34d399;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:light dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font);font-size:13px;line-height:1.55;min-height:100vh}

.page{max-width:960px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:20px;font-weight:700;letter-spacing:-.03em;margin-bottom:4px}
.subtitle{font-size:12px;color:var(--muted);margin-bottom:28px}
h2{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--border)}
.hero-row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:28px}
.hero-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:14px 18px;flex:1;min-width:140px}
.hero-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.hero-val{font-size:26px;font-weight:700;letter-spacing:-.03em;font-variant-numeric:tabular-nums;margin-top:2px}
.hero-sub{font-size:11px;color:var(--muted);margin-top:2px}
.hero-good{color:var(--good)}
.hero-warn{color:var(--warn)}

/* Tables */
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
td.r{text-align:right}
td.mono{font-family:var(--mono);font-size:11px}
.best{font-weight:700;color:var(--good)}
.sec{color:var(--accent)}
.row-highlight{background:var(--bg3)}

/* Bar cells */
.bar-cell{display:flex;align-items:center;gap:7px}
.bar-bg{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden;min-width:60px}
.bar-fill{height:100%;border-radius:3px}

/* Delta pill */
.pill{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;
  border-radius:8px;font-variant-numeric:tabular-nums}
.pill-g{background:rgba(10,122,69,.14);color:var(--good)}
.pill-w{background:rgba(184,92,0,.14);color:var(--warn)}
.pill-n{background:var(--bg3);color:var(--muted)}

/* Charts */
.chart-wrap{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:16px;margin-bottom:8px;overflow-x:auto}
svg text{font-family:var(--font)}

/* Model legend */
.legend{display:flex;flex-wrap:wrap;gap:10px 18px;font-size:11px;margin:10px 0 4px}
.leg-item{display:flex;align-items:center;gap:5px}
.leg-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.leg-rect{width:12px;height:5px;border-radius:2px;flex-shrink:0}

.note{font-size:11px;color:var(--muted);margin-top:6px}
</style>

<div class="page">
  <h1>STITCH Model Evaluation</h1>
  <div class="subtitle" id="subtitle">Test set · N stars loading…</div>

  <div class="hero-row" id="hero-row"></div>

  <h2>Model Progression</h2>
  <div class="legend" id="model-legend"></div>
  <div class="chart-wrap" id="bar-chart"></div>
  <div class="tbl-wrap"><table id="summary-table"></table></div>
  <p class="note">CV = within-star coefficient of variation (std/mean of sector_median/prediction across sectors). Lower is better. MAE and Pearson r are vs the LOO flux offset label.</p>

  <h2>Per-Camera Breakdown</h2>
  <div class="chart-wrap" id="cam-chart"></div>
  <div class="tbl-wrap"><table id="cam-table"></table></div>

  <h2>Per-Brightness Breakdown</h2>
  <div class="tbl-wrap"><table id="tmag-table"></table></div>

  <h2>Per-Sectors-Observed Breakdown</h2>
  <div class="tbl-wrap"><table id="nsec-table"></table></div>

  <h2>Per-Star Improvement Distribution (best model vs raw)</h2>
  <div class="chart-wrap" id="hist-chart"></div>
  <p class="note" id="hist-note"></p>
</div>

<script>
const D = DATA_PLACEHOLDER;
const CAM_C = ['','#d95b37','#1d6fb5','#16a34a','#b45309'];

// Palette for approaches in order
const APPROACH_COLORS = {
  'Raw (no correction)':  '#8a8980',
  'CCD-mean direct':      '#c084fc',
  'KNN-mean direct':      '#60a5fa',
  'LOO+GaiaFeat':         '#f59e0b',
  'LOO+GaiaFeat+NBR':     '#7c3aed',
  'LOO+GaiaFeat+KNN':     '#059669',
};

function color(name){ return APPROACH_COLORS[name] || '#8a8980'; }

// ── Hero cards ─────────────────────────────────────────────────────────────
const raw_s   = D.summary.find(r=>r.label==='Raw (no correction)');
const best_s  = D.summary[D.summary.length-1];
const flowModels = D.summary.filter(r=>r.label.startsWith('LOO+'));
const gaia_s  = flowModels[0];

const heroData = [
  {label:'Test Stars', val: D.n_stars.toLocaleString(), sub:'held-out set', cls:''},
  {label:'Raw Median CV', val: raw_s.med_cv.toFixed(3)+'%', sub:'before correction', cls:'hero-warn'},
  {label:'Best Model CV', val: best_s.med_cv.toFixed(3)+'%', sub:best_s.label, cls:'hero-good'},
  {label:'Scatter Reduction', val: best_s.cv_red.toFixed(1)+'%', sub:'vs no correction', cls:'hero-good'},
  {label:'Stars Improved', val: best_s.pct_imp.toFixed(1)+'%', sub:'vs raw', cls:''},
  {label:'Stars Hurt', val: best_s.pct_hurt.toFixed(1)+'%', sub:'worse than raw', cls: best_s.pct_hurt>10?'hero-warn':''},
];
document.getElementById('hero-row').innerHTML = heroData.map(h=>
  `<div class="hero-card"><div class="hero-label">${h.label}</div>`+
  `<div class="hero-val ${h.cls}">${h.val}</div>`+
  `<div class="hero-sub">${h.sub}</div></div>`
).join('');

document.getElementById('subtitle').textContent =
  `Test set · ${D.n_stars.toLocaleString()} stars · ${D.approaches.length} approaches compared`;

// ── Model legend ────────────────────────────────────────────────────────────
document.getElementById('model-legend').innerHTML = D.approaches.map(a=>
  `<div class="leg-item"><div class="leg-rect" style="background:${color(a)}"></div>${a}</div>`
).join('');

// ── Summary bar chart ───────────────────────────────────────────────────────
(function(){
  const W=700, rowH=28, pad={l:180,r:80,t:10,b:20};
  const n = D.summary.length;
  const H = n*rowH + pad.t + pad.b;
  const pw = W - pad.l - pad.r;
  const maxRed = Math.max(...D.summary.map(r=>Math.max(0,r.cv_red)));
  function xp(v){ return pad.l + (v/maxRed)*pw; }

  let s = `<svg width="100%" viewBox="0 0 ${W} ${H}" style="display:block">`;
  // Grid
  [0,10,20,30,maxRed].forEach(v=>{
    const x = xp(v).toFixed(1);
    s+=`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H-pad.b}" stroke="currentColor" stroke-width="0.5" opacity="0.1"/>`;
    s+=`<text x="${x}" y="${H-pad.b+12}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.45">${v.toFixed(0)}%</text>`;
  });
  // Bars
  D.summary.forEach((r,i)=>{
    const y = pad.t + i*rowH;
    const cy = y + rowH/2;
    const bh = rowH*0.48, by = cy - bh/2;
    const red = Math.max(0, r.cv_red);
    const bw = (red/maxRed)*pw;
    const c = color(r.label);
    const isBest = i===D.summary.length-1;
    s+=`<text x="${pad.l-8}" y="${cy+4}" text-anchor="end" font-size="${isBest?11:10}" `+
       `font-weight="${isBest?700:400}" fill="currentColor" opacity="${isBest?1:0.75}">${r.label}</text>`;
    s+=`<rect x="${pad.l}" y="${by.toFixed(1)}" width="${Math.max(2,bw).toFixed(1)}" height="${bh.toFixed(1)}" `+
       `fill="${c}" rx="2" opacity="${isBest?1:0.75}"/>`;
    s+=`<text x="${(pad.l+Math.max(2,bw)+6).toFixed(1)}" y="${cy+4}" font-size="11" `+
       `font-weight="${isBest?700:400}" fill="${isBest?'var(--good)':'currentColor'}" opacity="${isBest?1:0.65}">`+
       `${red.toFixed(1)}%</text>`;
  });
  s+='</svg>';
  document.getElementById('bar-chart').innerHTML = s;
})();

// ── Summary table ──────────────────────────────────────────────────────────
(function(){
  const bestMed = Math.min(...D.summary.map(r=>r.med_cv));
  const hdr = `<thead><tr>
    <th>Approach</th>
    <th class="r">Median CV%</th>
    <th class="r">↓ vs raw</th>
    <th class="r">MAE (LOO)</th>
    <th class="r">Pearson r</th>
    <th class="r">% improved</th>
    <th class="r">% hurt</th>
  </tr></thead>`;
  const body = D.summary.map((r,i)=>{
    const isBest = i===D.summary.length-1;
    const isPrev = i===D.summary.length-2;
    const cls = isBest?'row-highlight':'';
    const medCls = r.med_cv===bestMed?'best':'';
    const hurtCls = r.pct_hurt>10?'hero-warn':'';
    const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color(r.label)};margin-right:5px;vertical-align:middle"></span>`;
    return `<tr class="${cls}">
      <td>${dot}${r.label}</td>
      <td class="r mono ${medCls}">${r.med_cv.toFixed(3)}%</td>
      <td class="r"><span class="pill ${r.cv_red>0?'pill-g':'pill-n'}">${r.cv_red>0?'↓'+r.cv_red.toFixed(1):r.cv_red.toFixed(1)}%</span></td>
      <td class="r mono">${r.mae!=null?r.mae.toFixed(4):'—'}</td>
      <td class="r mono">${r.r!=null?r.r.toFixed(3):'—'}</td>
      <td class="r">${r.pct_imp.toFixed(1)}%</td>
      <td class="r ${hurtCls}">${r.pct_hurt.toFixed(1)}%</td>
    </tr>`;
  }).join('');
  document.getElementById('summary-table').innerHTML = hdr + '<tbody>' + body + '</tbody>';
})();

// ── Camera chart (grouped bars) ────────────────────────────────────────────
(function(){
  const flowNames = D.approaches.filter(a=>a.startsWith('LOO+'));
  const rawName   = 'Raw (no correction)';
  const cols = [rawName, ...flowNames];
  const cams = D.cam;
  const nCam=cams.length, nCol=cols.length;
  const groupW=130, barW=Math.max(8, groupW/nCol - 2), gap=4;
  const W=720, pad={l:55,r:20,t:28,b:35};
  const H=220;
  const ph=H-pad.t-pad.b;
  const maxCV=Math.max(...cams.flatMap(c=>cols.map(col=>c[col]||0)))*1.05;
  function yp(v){ return pad.t+(1-v/maxCV)*ph; }
  function xGroup(i){ return pad.l+i*(groupW+gap)+gap; }

  let s=`<svg width="100%" viewBox="0 0 ${W} ${H}" style="display:block">`;
  // Y grid
  [0,0.5,1,1.5,2,2.5,3].forEach(v=>{
    if(v>maxCV) return;
    const y=yp(v).toFixed(1);
    s+=`<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="currentColor" stroke-width="0.5" opacity="0.1"/>`;
    s+=`<text x="${pad.l-4}" y="${parseFloat(y)+3}" text-anchor="end" font-size="9" fill="currentColor" opacity="0.45">${v.toFixed(1)}%</text>`;
  });
  s+=`<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H-pad.b}" stroke="currentColor" stroke-width="0.5" opacity="0.15"/>`;
  // Bars
  cams.forEach((cam,gi)=>{
    const gx=xGroup(gi);
    const cx=gx+groupW/2;
    cols.forEach((col,ci)=>{
      const val=cam[col]||0;
      const bx=gx+ci*(barW+1.5);
      const by=yp(val);
      const bh=H-pad.b-by;
      s+=`<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(1,bh).toFixed(1)}" `+
         `fill="${color(col)}" rx="1.5" opacity="0.85"/>`;
    });
    s+=`<text x="${cx.toFixed(1)}" y="${H-pad.b+13}" text-anchor="middle" font-size="10" fill="${CAM_C[cam.cam]}" font-weight="600">Cam${cam.cam}</text>`;
  });
  // Legend
  cols.forEach((col,i)=>{
    const lx=pad.l+i*120;
    s+=`<rect x="${lx}" y="6" width="10" height="5" rx="1.5" fill="${color(col)}" opacity="0.85"/>`;
    s+=`<text x="${lx+13}" y="13" font-size="9" fill="currentColor" opacity="0.7">${col}</text>`;
  });
  s+='</svg>';
  document.getElementById('cam-chart').innerHTML=s;
})();

// ── Camera table ───────────────────────────────────────────────────────────
(function(){
  const flowNames = D.approaches.filter(a=>a.startsWith('LOO+'));
  const rawName   = 'Raw (no correction)';
  const showCols  = [rawName, ...flowNames];
  const hdr='<thead><tr><th>Camera</th>'+showCols.map(c=>`<th class="r">${c}</th>`).join('')+
    '<th class="r">Best ↓ vs raw</th></tr></thead>';
  const body=D.cam.map(c=>{
    const raw=c[rawName];
    const flowVals=flowNames.map(n=>c[n]);
    const best=Math.min(...flowVals);
    const red=((raw-best)/raw*100).toFixed(1);
    return`<tr>
      <td><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${CAM_C[c.cam]};margin-right:5px;vertical-align:middle"></span>Cam${c.cam}</td>
      ${showCols.map(col=>{
        const v=c[col];
        const isBest=v===best && col!==rawName;
        return`<td class="r mono ${isBest?'best':''}">${v.toFixed(3)}%</td>`;
      }).join('')}
      <td class="r"><span class="pill pill-g">↓${red}%</span></td>
    </tr>`;
  }).join('');
  document.getElementById('cam-table').innerHTML=hdr+'<tbody>'+body+'</tbody>';
})();

// ── Tmag table ─────────────────────────────────────────────────────────────
(function(){
  const flowNames=D.approaches.filter(a=>a.startsWith('LOO+'));
  const raw='Raw (no correction)';
  const cols=[raw,...flowNames];
  const hdr='<thead><tr><th>Tmag</th><th class="r">N stars</th>'+
    cols.map(c=>`<th class="r">${c}</th>`).join('')+'</tr></thead>';
  const body=D.tmag.map(r=>{
    const rawV=r[raw];
    const flowVals=flowNames.map(n=>r[n]);
    const best=Math.min(...flowVals);
    return`<tr>
      <td>${r.bin}</td>
      <td class="r">${r.n.toLocaleString()}</td>
      ${cols.map(c=>{
        const v=r[c];
        const isBest=v===best&&c!==raw;
        return`<td class="r mono ${isBest?'best':''}">${v.toFixed(3)}%</td>`;
      }).join('')}
    </tr>`;
  }).join('');
  document.getElementById('tmag-table').innerHTML=hdr+'<tbody>'+body+'</tbody>';
})();

// ── N-sectors table ────────────────────────────────────────────────────────
(function(){
  const flowNames=D.approaches.filter(a=>a.startsWith('LOO+'));
  const raw='Raw (no correction)';
  const cols=[raw,...flowNames];
  const hdr='<thead><tr><th>Sectors observed</th><th class="r">N stars</th>'+
    cols.map(c=>`<th class="r">${c}</th>`).join('')+'</tr></thead>';
  const body=D.nsec.map(r=>{
    const flowVals=flowNames.map(n=>r[n]);
    const best=Math.min(...flowVals);
    return`<tr>
      <td>${r.bin}</td>
      <td class="r">${r.n.toLocaleString()}</td>
      ${cols.map(c=>{
        const v=r[c];
        const isBest=v===best&&c!==raw;
        return`<td class="r mono ${isBest?'best':''}">${v.toFixed(3)}%</td>`;
      }).join('')}
    </tr>`;
  }).join('');
  document.getElementById('nsec-table').innerHTML=hdr+'<tbody>'+body+'</tbody>';
})();

// ── Histogram ───────────────────────────────────────────────────────────────
(function(){
  const H=D.hist;
  const W=700,pH=180,pad={l:45,r:20,t:10,b:35};
  const pw=W-pad.l-pad.r,ph=pH-pad.t-pad.b;
  const maxN=Math.max(...H.map(b=>b.n));
  const xMin=H[0].x, xMax=H[H.length-1].x+H[H.length-1].w;
  function xp(v){return pad.l+(v-xMin)/(xMax-xMin)*pw;}
  function yp(v){return pad.t+(1-v/maxN)*ph;}

  let s=`<svg width="100%" viewBox="0 0 ${W} ${pH}" style="display:block">`;
  // X grid lines
  [-2,-1.5,-1,-0.5,0,0.5,1].forEach(v=>{
    if(v<xMin||v>xMax) return;
    const x=xp(v).toFixed(1);
    s+=`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${pH-pad.b}" stroke="currentColor" stroke-width="${v===0?1:0.5}" stroke-dasharray="${v===0?'':'3 3'}" opacity="${v===0?0.2:0.1}"/>`;
    s+=`<text x="${x}" y="${pH-pad.b+13}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.5">${v>=0?'+':''}${v.toFixed(1)}</text>`;
  });
  // Axis label
  s+=`<text x="${pad.l+pw/2}" y="${pH-pad.b+27}" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.4">Δ CV% (model – raw)  ·  negative = improved</text>`;
  // Bars
  H.forEach(b=>{
    if(b.n===0) return;
    const x=xp(b.x); const x2=xp(b.x+b.w);
    const bw=Math.max(1,x2-x-1);
    const by=yp(b.n); const bh=pH-pad.b-by;
    const isHurt=b.x>=0;
    s+=`<rect x="${x.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1,bh).toFixed(1)}" `+
       `fill="${isHurt?'#dc6803':'#059669'}" rx="1" opacity="0.8"/>`;
  });
  s+=`<line x1="${pad.l}" y1="${pH-pad.b}" x2="${W-pad.r}" y2="${pH-pad.b}" stroke="currentColor" stroke-width="0.5" opacity="0.15"/>`;
  s+='</svg>';
  document.getElementById('hist-chart').innerHTML=s;

  // Annotation
  const nHurt=H.filter(b=>b.x>=0).reduce((a,b)=>a+b.n,0);
  const nImp =H.filter(b=>b.x<0).reduce((a,b)=>a+b.n,0);
  const tot=nHurt+nImp;
  document.getElementById('hist-note').textContent=
    `Green = improved vs raw (${nImp.toLocaleString()} stars, ${(nImp/tot*100).toFixed(1)}%). `+
    `Orange = worsened (${nHurt.toLocaleString()}, ${(nHurt/tot*100).toFixed(1)}%). `+
    `Model: ${D.best_name}`;
})();
</script>
"""

HTML = HTML.replace("DATA_PLACEHOLDER", data_js)
with open(OUT, "w") as f:
    f.write(HTML)
print(f"\nWritten → {OUT}  ({len(HTML)/1024:.1f} KB)")
