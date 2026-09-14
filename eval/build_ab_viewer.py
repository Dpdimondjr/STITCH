"""
Build self-contained A/B model comparison viewer for STITCH.

Usage (from STITCH root):
  python3 eval/build_ab_viewer.py [model_a.pt] [model_b.pt] [parquet] [output.html]

Defaults:
  model_a  = stitch_nsf_gaia_feat.pt
  model_b  = stitch_nsf_knn.pt
  parquet  = training_data_topup_pdc.parquet
  output   = stitch_ab_viewer.html
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import zuko
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

# ── CLI args ──────────────────────────────────────────────────────────────────
pt_args  = [a for a in sys.argv[1:] if a.endswith(".pt")]
pq_args  = [a for a in sys.argv[1:] if a.endswith(".parquet")]
htm_args = [a for a in sys.argv[1:] if a.endswith(".html")]

MODEL_A = pt_args[0] if len(pt_args) > 0 else "stitch_nsf_gaia_feat.pt"
MODEL_B = pt_args[1] if len(pt_args) > 1 else "stitch_nsf_knn.pt"
PARQUET = pq_args[0] if pq_args else "training_data_topup_pdc.parquet"
OUT     = htm_args[0] if htm_args else "stitch_ab_viewer.html"

NAME_A = os.path.basename(MODEL_A).replace(".pt", "")
NAME_B = os.path.basename(MODEL_B).replace(".pt", "")

MIN_SECTORS = 4
TMAG_MAX    = 13.0
N_SAMPLES   = 200

print(f"Model A : {MODEL_A}  →  {NAME_A}")
print(f"Model B : {MODEL_B}  →  {NAME_B}")
print(f"Parquet : {PARQUET}")
print(f"Output  : {OUT}")
print()

# ── Load & clean ───────────────────────────────────────────────────────────────
print("Loading …")
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["col", "row", "flux_offset", "flux_offset_loo"])
df = df[(df["flux_offset"]     > 0.85) & (df["flux_offset"]     < 1.15)]
df = df[(df["flux_offset_loo"] > 0.85) & (df["flux_offset_loo"] < 1.15)]
df = df[df["n_sectors_total"] >= MIN_SECTORS]
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
_K = 20
_knn = np.full(len(df), np.nan, dtype=np.float32)
for (_s, _c, _q), _g in df.groupby(["sector", "cam", "ccd"]):
    _n = len(_g)
    if _n < 2:
        continue
    _xy = _g[["col", "row"]].values.astype(np.float32)
    _v  = _g["flux_offset_loo"].values
    _k  = min(_K, _n - 1)
    _, _nn = cKDTree(_xy).query(_xy, k=_k + 1)
    _knn[_g.index.values] = _v[_nn[:, 1:]].mean(axis=1)
df["spatial_knn_mean_loo"] = _knn
df["spatial_knn_mean_loo"] = df["spatial_knn_mean_loo"].fillna(df["sector_ccd_mean_loo"])

CONTINUOUS_BASE = ["col", "row", "delta_sub_col", "delta_sub_row",
                   "sector", "tmag", "crowdsap", "cdpp1_0", "pdcvar", "jitter_rms",
                   "pdc_noi", "pr_wght2"]
for col in CONTINUOUS_BASE:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# ── Reproduce exact train/val/test split ───────────────────────────────────────
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: int(x.mode()[0]))
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))

tr, tmp = train_test_split(star_cam["tic_id"], test_size=0.2,
                            stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp)]["dominant_cam"]
vl, te  = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)

test_df = df[df["tic_id"].isin(te)].copy().reset_index(drop=True)
print(f"  Test : {len(test_df):,} rows  ·  {test_df['tic_id'].nunique():,} stars\n")

# ── Inference ─────────────────────────────────────────────────────────────────
device = (torch.device("mps")  if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

def make_context(split_df, means, stds, cont_cols, cam_cols, ccd_cols):
    cont   = (split_df[cont_cols] - means) / stds
    cam_oh = (pd.get_dummies(split_df["cam"].astype(int), prefix="cam")
               .reindex(columns=cam_cols, fill_value=0))
    ccd_oh = (pd.get_dummies(split_df["ccd"].astype(int), prefix="ccd")
               .reindex(columns=ccd_cols, fill_value=0))
    return pd.concat([cont.reset_index(drop=True),
                      cam_oh.reset_index(drop=True),
                      ccd_oh.reset_index(drop=True)], axis=1).values.astype(np.float32)

def predict(path, eval_df):
    print(f"  Running {os.path.basename(path)} …", flush=True)
    ck   = torch.load(path, map_location="cpu", weights_only=False)
    flow = zuko.flows.NSF(**ck["flow_config"])
    flow.load_state_dict(ck["model_state"])
    flow = flow.to(device); flow.eval()
    C = make_context(eval_df, ck["means"], ck["stds"],
                     ck["continuous_cols"], ck["cam_cols"], ck["ccd_cols"])
    C_t = torch.tensor(C).to(device)
    with torch.no_grad():
        mu = flow(C_t).sample((N_SAMPLES,)).squeeze(-1).mean(0).cpu().numpy()
    return mu * ck["y_std"] + ck["y_mean"]

pa = predict(MODEL_A, test_df)
pb = predict(MODEL_B, test_df)

test_df["pred_a"] = pa
test_df["pred_b"] = pb
test_df["corr_raw"] = test_df["sector_median"]
test_df["corr_a"]   = test_df["sector_median"] / pa
test_df["corr_b"]   = test_df["sector_median"] / pb

# ── Within-star scatter CV ────────────────────────────────────────────────────
def cv_per_star(df, col):
    return df.groupby("tic_id")[col].apply(
        lambda x: float(x.std() / x.mean()) if len(x) >= 2 else np.nan)

cv_raw = cv_per_star(test_df, "corr_raw")
cv_a   = cv_per_star(test_df, "corr_a")
cv_b   = cv_per_star(test_df, "corr_b")

pct_b_wins = (cv_b < cv_a).mean() * 100
med_raw = float(cv_raw.median()) * 100
med_a   = float(cv_a.median())   * 100
med_b   = float(cv_b.median())   * 100
print(f"\n  CV raw={med_raw:.3f}%  A={med_a:.3f}%  B={med_b:.3f}%  "
      f"B wins: {pct_b_wins:.1f}% of stars")

# ── Aggregate & encode ────────────────────────────────────────────────────────
CHARS  = "0123456789abcdefghijklmnopqrstu"
SECT36 = "0123456789abcdefghijklmnopqrstuvwxyz"

def enc_v(v):
    return CHARS[max(0, min(30, round((v - 0.85) * 100)))]

def enc_vs(lst):
    return "".join(enc_v(v) for v in lst)

def enc_sects(ss):
    return "".join(SECT36[s // 36] + SECT36[s % 36] for s in sorted(int(s) for s in ss))

agg = test_df.groupby("tic_id").agg(
    tmag=("tmag",   "mean"),
    n   =("sector", "nunique"),
    cam =("cam",    lambda x: int(x.mode()[0])),
    ra  =("ra",     "first"),
    dec =("dec",    "first"),
).reset_index()

sorted_test = test_df.sort_values(["tic_id", "sector"])

rows = []
for _, r in agg.iterrows():
    tic = int(r["tic_id"])
    g   = sorted_test[sorted_test["tic_id"] == tic]
    cr  = float(cv_raw.get(tic, 0) or 0) * 100
    ca  = float(cv_a  .get(tic, 0) or 0) * 100
    cb  = float(cv_b  .get(tic, 0) or 0) * 100
    rows.append([
        tic,
        round(float(r["tmag"]), 1),
        int(r["n"]),
        int(r["cam"]),
        round(float(r["ra"]),  1),
        round(float(r["dec"]), 1),
        round(cr, 3),   # [6] cv_raw %
        round(ca, 3),   # [7] cv_a %
        round(cb, 3),   # [8] cv_b %
        enc_sects(g["sector"].tolist()),         # [9]
        enc_vs(g["flux_offset_loo"].tolist()),   # [10]
        enc_vs(g["pred_a"].tolist()),            # [11]
        enc_vs(g["pred_b"].tolist()),            # [12]
    ])

rows.sort(key=lambda x: x[7] - x[8], reverse=True)

data_js    = json.dumps(rows, separators=(",", ":"))
summary_js = json.dumps({
    "medRaw": round(med_raw, 3),
    "medA":   round(med_a,   3),
    "medB":   round(med_b,   3),
    "pctB":   round(pct_b_wins, 1),
    "n":      len(rows),
})
print(f"\n  Encoded {len(rows):,} stars  ·  {len(data_js)/1024:.0f} KB")

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """\
<title>STITCH · A/B Comparison</title>
<style>
:root{
  --bg:#f7f6f4;--bg2:#efede9;--bg3:#e6e4df;
  --ink:#17160f;--ink2:#4e4d47;--muted:#8c8b84;
  --border:#dddbd4;--accent:#2a78d6;--acc-dim:#d0e4f7;
  --hover:#eceae6;
  --c1:#e05c3a;--c2:#2a78d6;--c3:#1baf7a;--c4:#c47900;
  --font:system-ui,-apple-system,sans-serif;
  --mono:'SF Mono','Cascadia Code','Fira Code',ui-monospace,monospace;
  --ma:#7c3aed;--mb:#059669;
  --ma-lo:#ede9fe;--mb-lo:#d1fae5;
  --raw-c:#8c8b84;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --ink:#dddbd3;--ink2:#9a9890;--muted:#555750;
  --border:#2a2f38;--accent:#4d9de0;--acc-dim:#132236;
  --hover:#1c2128;
  --ma:#a78bfa;--mb:#34d399;
  --ma-lo:#2e1065;--mb-lo:#064e3b;
}}
:root[data-theme=dark]{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --ink:#dddbd3;--ink2:#9a9890;--muted:#555750;
  --border:#2a2f38;--accent:#4d9de0;--acc-dim:#132236;
  --hover:#1c2128;
  --ma:#a78bfa;--mb:#34d399;
  --ma-lo:#2e1065;--mb-lo:#064e3b;
}
:root[data-theme=light]{
  --bg:#f7f6f4;--bg2:#efede9;--bg3:#e6e4df;
  --ink:#17160f;--ink2:#4e4d47;--muted:#8c8b84;
  --border:#dddbd4;--accent:#2a78d6;--acc-dim:#d0e4f7;
  --hover:#eceae6;
  --ma:#7c3aed;--mb:#059669;
  --ma-lo:#ede9fe;--mb-lo:#d1fae5;
  --raw-c:#8c8b84;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:light dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font);font-size:13px;line-height:1.5;min-height:100vh}

.hdr{background:var(--bg);border-bottom:1px solid var(--border);
  padding:10px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.hdr-title{font-size:14px;font-weight:650;letter-spacing:-.02em}
.hdr-sub{font-size:11px;color:var(--muted);flex:1}
.model-tag{font-size:11px;padding:2px 9px;border-radius:12px;font-family:var(--mono);font-weight:500}
.tag-a{background:var(--ma-lo);color:var(--ma)}
.tag-b{background:var(--mb-lo);color:var(--mb)}

.flt{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:9px 18px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
.fg{display:flex;align-items:center;gap:6px}
.fl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);white-space:nowrap}
.tg{display:flex;gap:2px}
.tb{font-size:11px;padding:2px 7px;border:1px solid var(--border);border-radius:3px;
  background:transparent;color:var(--ink2);cursor:pointer;font-family:var(--font)}
.tb:hover{background:var(--bg3)}
.tb.on{background:var(--accent);border-color:var(--accent);color:#fff}
.srch{font-size:12px;font-family:var(--mono);padding:3px 8px;border:1px solid var(--border);
  border-radius:3px;background:var(--bg);color:var(--ink);width:120px;outline:none}
.srch:focus{border-color:var(--accent)}
select{font-size:11px;font-family:var(--font);padding:2px 5px;border:1px solid var(--border);
  border-radius:3px;background:var(--bg);color:var(--ink);outline:none;cursor:pointer}

.stats-strip{padding:6px 18px;font-size:11px;color:var(--muted);background:var(--bg2);
  border-bottom:1px solid var(--border);display:flex;gap:22px;flex-wrap:wrap;align-items:center}
.ss-item b{font-weight:600;font-variant-numeric:tabular-nums}
.ss-a{color:var(--ma)}
.ss-b{color:var(--mb)}

.rbar{position:sticky;top:0;z-index:40;padding:6px 18px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--border);background:var(--bg)}
.showing{font-size:11px;color:var(--muted);flex:1}
.srt-wr{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted)}
.pgn{display:flex;align-items:center;gap:5px}
.pb{width:24px;height:24px;display:flex;align-items:center;justify-content:center;
  border:1px solid var(--border);border-radius:3px;background:transparent;color:var(--ink2);cursor:pointer;font-size:12px}
.pb:disabled{opacity:.3;cursor:default}
.pi{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;min-width:60px;text-align:center}

.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--bg2);padding:6px 10px;text-align:left;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--border);
  white-space:nowrap;user-select:none;cursor:pointer;position:sticky;top:0;z-index:10}
thead th:hover{color:var(--ink)}
thead th.on{color:var(--accent)}
thead th.on::after{content:' ↓'}
thead th.asc::after{content:' ↑'}
thead th.nosort{cursor:default}
tbody tr{border-bottom:1px solid var(--border);cursor:pointer}
tbody tr:hover{background:var(--hover)}
tbody tr.sel{background:var(--acc-dim)}
td{padding:5px 10px;vertical-align:middle;white-space:nowrap}
.t-id{font-family:var(--mono);font-size:11.5px}
.t-id a{color:var(--accent);text-decoration:none}
.t-id a:hover{text-decoration:underline}
.t-num{font-variant-numeric:tabular-nums}
.t-coord{font-family:var(--mono);font-size:10.5px;color:var(--ink2)}
.chip-cam{display:inline-flex;align-items:center;justify-content:center;
  width:18px;height:18px;border-radius:3px;font-size:10px;font-weight:700;color:#fff}
.nbadge{display:inline-block;padding:1px 6px;border-radius:10px;
  font-size:10.5px;font-weight:600;font-variant-numeric:tabular-nums;
  background:var(--bg3);color:var(--ink2)}
.cv-cell{display:flex;align-items:center;gap:5px;min-width:100px}
.cv-bars{display:flex;flex-direction:column;gap:2px;flex:1;min-width:52px}
.cv-row{display:flex;align-items:center;gap:3px}
.cv-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.cv-bar-bg{flex:1;height:3px;background:var(--border);border-radius:2px;position:relative;overflow:hidden}
.cv-bar{height:100%;border-radius:2px;transition:width .15s}
.cv-num{font-size:10px;font-variant-numeric:tabular-nums;min-width:36px;text-align:right;
  font-family:var(--mono)}
.delta-cell{min-width:72px}
.delta-pill{display:inline-block;font-size:10.5px;font-weight:600;padding:1px 7px;
  border-radius:10px;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.dp-b{background:rgba(5,150,105,.15);color:#059669}
.dp-a{background:rgba(220,108,3,.15);color:#dc6803}
.dp-tie{background:var(--bg3);color:var(--muted)}
.no-r{padding:60px 18px;text-align:center;color:var(--muted);font-size:13px}

/* Detail panel */
.detail-row td{padding:0;background:var(--bg2);border-bottom:2px solid var(--accent)}
.detail-inner{padding:16px 18px;display:flex;gap:24px;flex-wrap:wrap}
.detail-chart{flex:1;min-width:300px}
.detail-stats{min-width:200px;display:flex;flex-direction:column;gap:10px}
.ds-head{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.ds-row{display:flex;align-items:center;gap:6px;font-size:12px}
.ds-label{color:var(--muted);font-size:11px;min-width:54px}
.ds-val{font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12px}
.ds-bar-bg{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden;min-width:80px}
.ds-bar{height:100%;border-radius:3px}
.ds-impr{font-size:10px;font-weight:600;padding:1px 5px;border-radius:8px}
.impr-pos{background:rgba(5,150,105,.15);color:#059669}
.impr-neg{background:rgba(220,108,3,.15);color:#dc6803}
</style>

<div class="hdr">
  <div class="hdr-title">STITCH &middot; A/B Viewer</div>
  <div class="hdr-sub">Within-star scatter reduction &middot; test set only</div>
  <span class="model-tag tag-a">A: NAME_A_PLACEHOLDER</span>
  <span>&rarr;</span>
  <span class="model-tag tag-b">B: NAME_B_PLACEHOLDER</span>
</div>

<div class="flt">
  <div class="fg">
    <span class="fl">TIC ID</span>
    <input class="srch" type="text" id="srch" placeholder="search&hellip;" autocomplete="off">
  </div>
  <div class="fg">
    <span class="fl">Camera</span>
    <div class="tg" id="cam-tg">
      <button class="tb on" data-v="0">All</button>
      <button class="tb" data-v="1" style="color:var(--c1)">1</button>
      <button class="tb" data-v="2" style="color:var(--c2)">2</button>
      <button class="tb" data-v="3" style="color:var(--c3)">3</button>
      <button class="tb" data-v="4" style="color:var(--c4)">4</button>
    </div>
  </div>
  <div class="fg">
    <span class="fl">Winner</span>
    <div class="tg" id="win-tg">
      <button class="tb on" data-v="all">All</button>
      <button class="tb" data-v="b" style="color:var(--mb)">B wins</button>
      <button class="tb" data-v="a" style="color:var(--ma)">A wins</button>
    </div>
  </div>
  <div class="fg">
    <span class="fl">Sort by</span>
    <div class="srt-wr">
      <select id="sort-sel">
        <option value="db-d">B improvement &darr; (most B-wins first)</option>
        <option value="db-a">A improvement &darr; (most A-wins first)</option>
        <option value="ra-d">Raw CV &darr;</option>
        <option value="tm-a">Tmag &uarr;</option>
        <option value="n-d">Sectors &darr;</option>
        <option value="cam-a">Camera</option>
      </select>
    </div>
  </div>
  <div class="fg"><button class="tb" id="rst">Reset</button></div>
</div>

<div class="stats-strip" id="stats-strip">
  <span class="ss-item">Stars: <b id="ss-n">&ndash;</b></span>
  <span class="ss-item">CV raw: <b id="ss-raw">&ndash;</b></span>
  <span class="ss-item">CV <span class="ss-a">Model A</span>: <b id="ss-a">&ndash;</b></span>
  <span class="ss-item">CV <span class="ss-b">Model B</span>: <b id="ss-b">&ndash;</b></span>
  <span class="ss-item ss-b">B wins: <b id="ss-bw">&ndash;</b></span>
</div>

<div class="rbar">
  <span class="showing" id="show-txt">&ndash;</span>
  <div class="pgn">
    <button class="pb" id="prv">&#8592;</button>
    <span class="pi" id="pg-inf">&ndash;</span>
    <button class="pb" id="nxt">&#8594;</button>
  </div>
</div>

<div class="tw">
  <table>
    <thead><tr>
      <th class="nosort">TIC ID</th>
      <th>Tmag</th>
      <th>N</th>
      <th class="nosort">Cam</th>
      <th class="nosort">RA &middot; Dec</th>
      <th class="nosort" title="Within-star scatter CV for raw / Model A / Model B">CV comparison</th>
      <th class="on" id="th-delta" title="CV improvement: positive = B better than A">&Delta; (A&rarr;B)</th>
      <th class="nosort">Offsets</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="no-r" class="no-r" style="display:none">No stars match these filters.</div>
</div>

<script>
// Data: [tic, tmag, n, cam, ra, dec, cv_raw%, cv_a%, cv_b%, sects, loo, pred_a, pred_b]
//        0    1     2  3    4   5    6         7      8      9      10   11      12
const DATA    = DATA_PLACEHOLDER;
const SUMMARY = SUMMARY_PLACEHOLDER;
const NTOT    = DATA.length;
const CAM_C   = ['','#e05c3a','#2a78d6','#1baf7a','#c47900'];
const CHARS   = '0123456789abcdefghijklmnopqrstu';
const SECT36  = '0123456789abcdefghijklmnopqrstuvwxyz';
const PER     = 80;
let fil = DATA.slice(), page = 0, selTic = null;
let st = {q:'', cam:0, win:'all', sort:'db-d'};

// Decode helpers
function decOff(s){
  const r=[];
  for(let i=0;i<s.length;i++) r.push(0.85+CHARS.indexOf(s[i])*0.01);
  return r;
}
function decSects(s){
  const r=[];
  for(let i=0;i<s.length;i+=2) r.push(SECT36.indexOf(s[i])*36+SECT36.indexOf(s[i+1]));
  return r;
}

// Max raw CV for bar scaling (95th percentile)
const _sorted = DATA.map(d=>d[6]).sort((a,b)=>a-b);
const MAX_CV  = _sorted[Math.floor(_sorted.length*0.95)] || 3.0;

function cvBar(val, color){
  const pct = Math.min(100, val/MAX_CV*100).toFixed(1);
  return `<div class="cv-bar" style="width:${pct}%;background:${color}"></div>`;
}

function cvCell(d){
  const cr=d[6], ca=d[7], cb=d[8];
  return `<div class="cv-cell">
    <div class="cv-bars">
      <div class="cv-row">
        <div class="cv-dot" style="background:var(--raw-c)"></div>
        <div class="cv-bar-bg">${cvBar(cr,'var(--raw-c)')}</div>
        <span class="cv-num" style="color:var(--muted)">${cr.toFixed(3)}%</span>
      </div>
      <div class="cv-row">
        <div class="cv-dot" style="background:var(--ma)"></div>
        <div class="cv-bar-bg">${cvBar(ca,'var(--ma)')}</div>
        <span class="cv-num" style="color:var(--ma)">${ca.toFixed(3)}%</span>
      </div>
      <div class="cv-row">
        <div class="cv-dot" style="background:var(--mb)"></div>
        <div class="cv-bar-bg">${cvBar(cb,'var(--mb)')}</div>
        <span class="cv-num" style="color:var(--mb)">${cb.toFixed(3)}%</span>
      </div>
    </div>
  </div>`;
}

function deltaPill(d){
  const delta = d[7] - d[8];  // cv_a - cv_b: positive = B better
  if(Math.abs(delta) < 0.001){
    return `<span class="delta-pill dp-tie">&asymp; 0</span>`;
  }
  const sign  = delta > 0 ? '+' : '';
  const cls   = delta > 0 ? 'dp-b' : 'dp-a';
  return `<span class="delta-pill ${cls}">${sign}${delta.toFixed(3)}%</span>`;
}

// Offset sparkline (same compact style as dataset viewer)
function spark(s){
  if(!s) return '';
  const offs=decOff(s);
  const W=100,H=22,cy=11,S=11/0.12;
  const ref5=(0.05*S).toFixed(2);
  const dw=W/offs.length, bw=Math.max(2,dw-1.5);
  let r=`<svg width="${W}" height="${H}" style="display:block;overflow:visible">`
    +`<line x1="0" y1="${cy}" x2="${W}" y2="${cy}" stroke="currentColor" stroke-width="0.6" opacity="0.12"/>`
    +`<line x1="0" y1="${cy-ref5}" x2="${W}" y2="${cy-ref5}" stroke="currentColor" stroke-dasharray="2 2" stroke-width="0.5" opacity="0.2"/>`
    +`<line x1="0" y1="${cy+ref5}" x2="${W}" y2="${cy+ref5}" stroke="currentColor" stroke-dasharray="2 2" stroke-width="0.5" opacity="0.2"/>`;
  offs.forEach((o,i)=>{
    const dev=Math.max(-11,Math.min(11,(o-1)*S));
    if(Math.abs(dev)<0.15)return;
    const x=(i*dw+(dw-bw)/2).toFixed(1);
    const y=(dev>0?cy-dev:cy).toFixed(1);
    const h=Math.max(0.6,Math.abs(dev)).toFixed(1);
    r+=`<rect x="${x}" y="${y}" width="${bw.toFixed(1)}" height="${h}" fill="${o>1?'#e05c3a':'#2a78d6'}" opacity=".8"/>`;
  });
  return r+'</svg>';
}

// Full comparison chart for detail row
function buildChart(d){
  const sects = decSects(d[9]);
  const loo   = decOff(d[10]);
  const pa    = decOff(d[11]);
  const pb    = decOff(d[12]);
  const n = sects.length;
  if(n < 2) return '<em style="color:var(--muted);font-size:11px">Too few sectors</em>';

  const W=580, H=190, ml=42, mr=16, mt=22, mb=36;
  const pw=W-ml-mr, ph=H-mt-mb;

  const allV = [...loo,...pa,...pb];
  const dMin = Math.min(...allV), dMax = Math.max(...allV);
  const pad  = Math.max(0.005, (dMax-dMin)*0.15);
  const yMin = Math.max(0.88, dMin-pad);
  const yMax = Math.min(1.12, dMax+pad);

  function xp(i){ return ml + (n===1?pw/2:i/(n-1)*pw); }
  function yp(v){ return mt + (1-(v-yMin)/(yMax-yMin))*ph; }

  // Y grid values
  const yGridBase = [0.90,0.92,0.94,0.96,0.98,1.00,1.02,1.04,1.06,1.08,1.10];
  const yGrid = yGridBase.filter(v=>v>=yMin-0.005&&v<=yMax+0.005);

  let g = `<svg width="100%" viewBox="0 0 ${W} ${H}" style="display:block;overflow:visible">`;

  // Grid
  yGrid.forEach(yv=>{
    const y=yp(yv).toFixed(1);
    const is1=(Math.abs(yv-1.0)<0.001);
    g+=`<line x1="${ml}" y1="${y}" x2="${W-mr}" y2="${y}" stroke="currentColor" `
      +`stroke-width="${is1?0.9:0.45}" stroke-dasharray="${is1?'':'3 3'}" opacity="${is1?0.18:0.09}"/>`;
    const label=((yv-1)*100).toFixed(0);
    g+=`<text x="${ml-4}" y="${y}" text-anchor="end" dominant-baseline="middle" `
      +`font-size="9" fill="currentColor" opacity="0.45">${label>0?'+'+label:label}%</text>`;
  });

  // Model A path
  const pathA=sects.map((_,i)=>`${i?'L':'M'}${xp(i).toFixed(1)},${yp(pa[i]).toFixed(1)}`).join('');
  g+=`<path d="${pathA}" fill="none" stroke="var(--ma)" stroke-width="1.8" stroke-linejoin="round" opacity="0.9"/>`;

  // Model B path
  const pathB=sects.map((_,i)=>`${i?'L':'M'}${xp(i).toFixed(1)},${yp(pb[i]).toFixed(1)}`).join('');
  g+=`<path d="${pathB}" fill="none" stroke="var(--mb)" stroke-width="1.8" stroke-linejoin="round" opacity="0.9"/>`;

  // LOO dots with native tooltips
  sects.forEach((s,i)=>{
    const cx=xp(i).toFixed(1), cy=yp(loo[i]).toFixed(1);
    const pct=((loo[i]-1)*100).toFixed(1);
    g+=`<circle cx="${cx}" cy="${cy}" r="3.5" fill="currentColor" opacity="0.55" `
      +`stroke="var(--bg)" stroke-width="1.2"><title>S${s}: LOO ${pct>0?'+':''}${pct}%</title></circle>`;
  });

  // Model A dots (smaller)
  sects.forEach((s,i)=>{
    const cx=xp(i).toFixed(1), cy=yp(pa[i]).toFixed(1);
    const pct=((pa[i]-1)*100).toFixed(1);
    g+=`<circle cx="${cx}" cy="${cy}" r="2" fill="var(--ma)" opacity="0.7">`
      +`<title>S${s}: Model A ${pct>0?'+':''}${pct}%</title></circle>`;
  });

  // Model B dots (smaller)
  sects.forEach((s,i)=>{
    const cx=xp(i).toFixed(1), cy=yp(pb[i]).toFixed(1);
    const pct=((pb[i]-1)*100).toFixed(1);
    g+=`<circle cx="${cx}" cy="${cy}" r="2" fill="var(--mb)" opacity="0.7">`
      +`<title>S${s}: Model B ${pct>0?'+':''}${pct}%</title></circle>`;
  });

  // X axis
  g+=`<line x1="${ml}" y1="${H-mb}" x2="${W-mr}" y2="${H-mb}" stroke="currentColor" stroke-width="0.5" opacity="0.15"/>`;
  const step=Math.max(1,Math.floor(n/10));
  sects.forEach((s,i)=>{
    if(i%step===0||i===n-1){
      g+=`<text x="${xp(i).toFixed(1)}" y="${H-mb+12}" text-anchor="middle" `
        +`font-size="9" fill="currentColor" opacity="0.55">${s}</text>`;
    }
  });
  g+=`<text x="${(ml+W-mr)/2}" y="${H-mb+25}" text-anchor="middle" `
    +`font-size="9" fill="currentColor" opacity="0.35">Sector</text>`;

  // Legend
  const ly=9, lx=ml+2;
  g+=`<circle cx="${lx}" cy="${ly}" r="3.5" fill="currentColor" opacity="0.5"/>`;
  g+=`<text x="${lx+8}" y="${ly+3.5}" font-size="9" fill="currentColor" opacity="0.55">LOO label</text>`;
  g+=`<line x1="${lx+65}" y1="${ly}" x2="${lx+80}" y2="${ly}" stroke="var(--ma)" stroke-width="2"/>`;
  g+=`<text x="${lx+84}" y="${ly+3.5}" font-size="9" fill="var(--ma)">Model A</text>`;
  g+=`<line x1="${lx+138}" y1="${ly}" x2="${lx+153}" y2="${ly}" stroke="var(--mb)" stroke-width="2"/>`;
  g+=`<text x="${lx+157}" y="${ly+3.5}" font-size="9" fill="var(--mb)">Model B</text>`;

  g+='</svg>';
  return g;
}

function detailHTML(d){
  const cr=d[6],ca=d[7],cb=d[8];
  const improvA=((cr-ca)/cr*100).toFixed(1);
  const improvB=((cr-cb)/cr*100).toFixed(1);
  const improvAB=((ca-cb)/ca*100).toFixed(1);
  function impCls(v){return parseFloat(v)>0?'impr-pos':'impr-neg';}
  function impSign(v){return parseFloat(v)>0?'+':'';}
  return `<div class="detail-inner">
    <div class="detail-chart">${buildChart(d)}</div>
    <div class="detail-stats">
      <div class="ds-head">Within-star CV</div>
      <div class="ds-row">
        <span class="ds-label" style="color:var(--muted)">Raw</span>
        <div class="ds-bar-bg"><div class="ds-bar" style="width:100%;background:var(--raw-c)"></div></div>
        <span class="ds-val" style="color:var(--muted)">${cr.toFixed(3)}%</span>
      </div>
      <div class="ds-row">
        <span class="ds-label" style="color:var(--ma)">Model A</span>
        <div class="ds-bar-bg"><div class="ds-bar" style="width:${Math.min(100,ca/cr*100).toFixed(1)}%;background:var(--ma)"></div></div>
        <span class="ds-val" style="color:var(--ma)">${ca.toFixed(3)}%</span>
        <span class="ds-impr ${impCls(improvA)}">${impSign(improvA)}${improvA}%</span>
      </div>
      <div class="ds-row">
        <span class="ds-label" style="color:var(--mb)">Model B</span>
        <div class="ds-bar-bg"><div class="ds-bar" style="width:${Math.min(100,cb/cr*100).toFixed(1)}%;background:var(--mb)"></div></div>
        <span class="ds-val" style="color:var(--mb)">${cb.toFixed(3)}%</span>
        <span class="ds-impr ${impCls(improvB)}">${impSign(improvB)}${improvB}%</span>
      </div>
      <div style="height:1px;background:var(--border);margin:4px 0"></div>
      <div style="font-size:11px;color:var(--muted)">B vs A:
        <span class="ds-impr ${impCls(improvAB)}" style="margin-left:4px">${impSign(improvAB)}${improvAB}%</span>
      </div>
      <div style="font-size:10px;color:var(--muted);margin-top:6px">TIC ${d[0]}</div>
      <div style="font-size:10px;color:var(--muted)">Tmag ${d[1].toFixed(1)} &middot; Cam${d[3]} &middot; ${d[2]} sectors</div>
      <div style="font-size:10px;color:var(--muted)">${d[4].toFixed(2)}&deg; ${d[5]>=0?'+':''}${d[5].toFixed(2)}&deg;</div>
    </div>
  </div>`;
}

function refilter(){
  const q = st.q;
  fil = DATA.filter(d=>{
    if(q && !String(d[0]).startsWith(q)) return false;
    if(st.cam && d[3]!==st.cam) return false;
    if(st.win==='b' && d[8]>=d[7]) return false;
    if(st.win==='a' && d[7]>=d[8]) return false;
    return true;
  });
  resort();
  // Update stats strip for filtered set
  updateStats();
}

function updateStats(){
  const n = fil.length;
  document.getElementById('ss-n').textContent = n.toLocaleString();
  if(!n){ return; }
  const medOf = (arr,i)=>{ const s=[...arr.map(d=>d[i])].sort((a,b)=>a-b); return s[Math.floor(s.length*.5)]; };
  document.getElementById('ss-raw').textContent = medOf(fil,6).toFixed(3)+'%';
  document.getElementById('ss-a').textContent   = medOf(fil,7).toFixed(3)+'%';
  document.getElementById('ss-b').textContent   = medOf(fil,8).toFixed(3)+'%';
  const bw = (fil.filter(d=>d[8]<d[7]).length/n*100).toFixed(1);
  document.getElementById('ss-bw').textContent  = bw+'%';
}

function resort(){
  const k = st.sort;
  fil.sort((a,b)=>
    k==='db-d' ? (b[7]-b[8])-(a[7]-a[8]) :
    k==='db-a' ? (a[7]-a[8])-(b[7]-b[8]) :
    k==='ra-d' ? b[6]-a[6] :
    k==='tm-a' ? a[1]-b[1] :
    k==='n-d'  ? b[2]-a[2] :
    k==='cam-a'? a[3]-b[3] : 0
  );
  selTic = null;
  page = 0;
  render();
}

function render(){
  const tot=fil.length, tp=Math.max(1,Math.ceil(tot/PER));
  page=Math.min(page,tp-1);
  const s=page*PER, slice=fil.slice(s,s+PER);
  document.getElementById('show-txt').textContent=tot===0?'No results':
    `Showing ${(s+1).toLocaleString()}–${Math.min(s+PER,tot).toLocaleString()} of ${tot.toLocaleString()} stars`;
  document.getElementById('pg-inf').textContent=`${page+1} / ${tp}`;
  document.getElementById('prv').disabled=page===0;
  document.getElementById('nxt').disabled=page>=tp-1;
  const tb=document.getElementById('rows');
  if(!slice.length){
    tb.innerHTML='';
    document.getElementById('no-r').style.display='';
    return;
  }
  document.getElementById('no-r').style.display='none';
  tb.innerHTML=slice.map(d=>{
    const [tic,tm,n,cam,ra,dec]=d;
    const dc=dec>=0?'+'+dec.toFixed(1):dec.toFixed(1);
    const isSel = tic===selTic;
    const rows = `<tr data-tic="${tic}" class="${isSel?'sel':''}">
      <td class="t-id"><a href="https://exofop.ipac.caltech.edu/tess/target.php?id=${tic}" target="_blank" rel="noreferrer" onclick="event.stopPropagation()">${tic}</a></td>
      <td class="t-num">${tm.toFixed(1)}</td>
      <td><span class="nbadge">${n}</span></td>
      <td><span class="chip-cam" style="background:${CAM_C[cam]}">${cam}</span></td>
      <td class="t-coord">${ra.toFixed(1)}&nbsp;&middot;&nbsp;${dc}</td>
      <td>${cvCell(d)}</td>
      <td class="delta-cell">${deltaPill(d)}</td>
      <td>${spark(d[10])}</td>
    </tr>`;
    if(isSel){
      return rows + `<tr class="detail-row"><td colspan="8">${detailHTML(d)}</td></tr>`;
    }
    return rows;
  }).join('');
}

// Row click → expand/collapse detail
document.getElementById('rows').addEventListener('click', e=>{
  const tr = e.target.closest('tr[data-tic]');
  if(!tr) return;
  const tic = parseInt(tr.dataset.tic, 10);
  selTic = selTic===tic ? null : tic;
  render();
  if(selTic!==null){
    // Scroll to make the detail visible
    setTimeout(()=>{
      const sel = document.querySelector('tr.sel');
      if(sel) sel.scrollIntoView({block:'nearest',behavior:'smooth'});
    }, 20);
  }
});

let sT;
document.getElementById('srch').addEventListener('input',e=>{
  clearTimeout(sT); sT=setTimeout(()=>{st.q=e.target.value.trim(); refilter();}, 180);
});

function tog(gid, key){
  document.getElementById(gid).addEventListener('click',e=>{
    const b=e.target.closest('.tb'); if(!b) return;
    document.querySelectorAll(`#${gid} .tb`).forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    const v=b.dataset.v;
    st[key]=key==='cam'?parseInt(v,10):v;
    refilter();
  });
}
tog('cam-tg','cam'); tog('win-tg','win');
document.getElementById('sort-sel').addEventListener('change',e=>{st.sort=e.target.value; resort();});
document.getElementById('prv').addEventListener('click',()=>{page--; render();});
document.getElementById('nxt').addEventListener('click',()=>{page++; render();});
document.getElementById('rst').addEventListener('click',()=>{
  st={q:'',cam:0,win:'all',sort:'db-d'};
  document.getElementById('srch').value='';
  document.getElementById('sort-sel').value='db-d';
  ['cam-tg','win-tg'].forEach(id=>
    document.querySelectorAll(`#${id} .tb`).forEach((b,i)=>b.classList.toggle('on',i===0)));
  refilter();
});

// Fix sticky thead height
(function(){
  const rbar=document.querySelector('.rbar');
  function fix(){
    const h=rbar.getBoundingClientRect().height;
    document.querySelectorAll('thead th').forEach(th=>th.style.top=h+'px');
  }
  fix(); new ResizeObserver(fix).observe(rbar);
})();

// Init
updateStats();
refilter();
</script>
"""

HTML = (HTML
    .replace("DATA_PLACEHOLDER",    data_js)
    .replace("SUMMARY_PLACEHOLDER", summary_js)
    .replace("NAME_A_PLACEHOLDER",  NAME_A)
    .replace("NAME_B_PLACEHOLDER",  NAME_B))

with open(OUT, "w") as f:
    f.write(HTML)
print(f"\nWritten → {OUT}  ({len(HTML)/1024:.1f} KB)")
