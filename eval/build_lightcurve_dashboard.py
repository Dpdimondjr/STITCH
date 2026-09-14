"""
Build the interactive STITCH dashboard.

Two tabs:
  1. Light Curves  — before/after sparklines for test-set stars, rich filters
  2. Dataset Explorer — slice/dice metadata across train/val/test splits

Usage:
  python3 eval/build_lightcurve_dashboard.py [dashboard_data.json]
"""

import json, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PARQUET_IN = "training_data_topup.parquet"
JSON_IN    = next((a for a in sys.argv[1:] if a.endswith(".json")), "eval/dashboard_data.json")
TARS_CSV   = "tars_quiet_tics_v2.csv"
HTML_OUT   = "eval/lightcurve_dashboard.html"

# ── 1. Load light-curve data (test set) ───────────────────────────────────────
print(f"Loading {JSON_IN}...")
with open(JSON_IN) as f:
    lc_data = json.load(f)
stars, summary = lc_data["stars"], lc_data["summary"]
print(f"  {len(stars):,} test-set stars, model: {summary['model']}")

# Join TARS sys_score onto test stars
tars_map = {}
if os.path.exists(TARS_CSV):
    tars = pd.read_csv(TARS_CSV)
    tars_map = {int(r.tic_id): float(r.mean_sys_score) for r in tars.itertuples()}
    n_found = sum(1 for s in stars if s["tic_id"] in tars_map)
    for s in stars:
        ss = tars_map.get(s["tic_id"])
        s["sys_score"] = round(ss, 4) if ss is not None else None
    print(f"  TARS sys_score: {n_found}/{len(stars)} joined")
else:
    for s in stars:
        s["sys_score"] = None

# ── 2. Compute filter metadata ranges ─────────────────────────────────────────
tmags     = [s["tmag"] for s in stars if s.get("tmag") is not None]
ns        = [s["n"] for s in stars]
scs       = [s["sc_before"] for s in stars]
improvs   = [s["improv"] for s in stars if abs(s["improv"]) < 1000]

meta = {
    "tmag_min":   round(min(tmags), 1) if tmags else 0,
    "tmag_max":   round(max(tmags), 1) if tmags else 15,
    "n_min":      int(min(ns)),
    "n_max":      int(max(ns)),
    "sc_max":     round(float(np.percentile(scs, 99)), 1),
    "improv_min": round(float(np.percentile(improvs, 1)), 0) if improvs else -200,
}

# ── 3. Compute dataset-explorer metadata (all splits) ─────────────────────────
print(f"Loading {PARQUET_IN} for dataset explorer...")
df = pd.read_parquet(PARQUET_IN)
df = df.dropna(subset=["col", "row", "flux_offset", "sector_median"])
df = df[(df["flux_offset"] > 0.85) & (df["flux_offset"] < 1.15)]

# Reproduce the same split used in training
star_cam = (df.groupby("tic_id")["cam"]
              .agg(lambda x: x.mode()[0])
              .reset_index()
              .rename(columns={"cam": "dominant_cam"}))
tr_tics, tmp_tics = train_test_split(
    star_cam["tic_id"], test_size=0.2, stratify=star_cam["dominant_cam"], random_state=42)
tmp_cam = star_cam[star_cam["tic_id"].isin(tmp_tics)]["dominant_cam"]
val_tics, te_tics = train_test_split(
    tmp_tics, test_size=0.5, stratify=tmp_cam.values, random_state=42)

split_of = {t: 0 for t in tr_tics}
split_of.update({t: 1 for t in val_tics})
split_of.update({t: 2 for t in te_tics})
print(f"  train={len(tr_tics):,}  val={len(val_tics):,}  test={len(te_tics):,}")

# Per-star metadata for all splits
print("  Computing per-star metadata...")
star_groups = df.groupby("tic_id")
md_split, md_cam, md_ccd, md_n, md_tmag, md_sc, md_sys = [], [], [], [], [], [], []

for tic, g in star_groups:
    sp = split_of.get(tic)
    if sp is None:
        continue
    g = g.sort_values("sector")
    smed = g["sector_median"].values
    raw_n = smed / smed.mean() if smed.mean() > 0 else smed
    sc = float(raw_n.std() * 100) if len(raw_n) > 1 else 0.0
    tmag = float(g["tmag"].iloc[0]) if "tmag" in g.columns else None
    sys_s = tars_map.get(int(tic))

    md_split.append(sp)
    md_cam.append(int(g["cam"].mode()[0]))
    md_ccd.append(int(g["ccd"].mode()[0]))
    md_n.append(int(len(g)))
    md_tmag.append(round(tmag, 2) if tmag is not None else None)
    md_sc.append(round(sc, 3))
    md_sys.append(round(sys_s, 4) if sys_s is not None else None)

print(f"  {len(md_split):,} total stars across all splits")

# Pack compactly
md_payload = json.dumps({
    "split": md_split,
    "cam":   md_cam,
    "ccd":   md_ccd,
    "n":     md_n,
    "tmag":  md_tmag,
    "sc":    md_sc,
    "sys":   md_sys,
}, separators=(",", ":"))

# ── 4. Serialize light-curve payload ──────────────────────────────────────────
lc_data["meta"] = meta
lc_payload = json.dumps(lc_data, separators=(",", ":"))

print(f"  LC payload: {len(lc_payload)//1024} KB")
print(f"  MD payload: {len(md_payload)//1024} KB")

# ── 5. HTML substitution values ────────────────────────────────────────────────
SUBS = {
    "%%TOTAL%%":    f"{len(stars):,}",
    "%%MEDIMPROV%%": f"{summary['median_improv']}%",
    "%%HARMRATE%%": f"{summary['harm_rate_n5']}%",
    "%%NCVZ%%":     str(summary["n_cvz"]),
    "%%MODEL%%":    summary["model"],
    "%%N_MAX%%":    str(meta["n_max"]),
    "%%TMAG_MIN%%": str(meta["tmag_min"]),
    "%%TMAG_MAX%%": str(meta["tmag_max"]),
    "%%SC_MAX%%":   str(meta["sc_max"]),
    "%%IMP_MIN%%":  str(int(meta["improv_min"])),
    "%%LC_JSON%%":  lc_payload,
    "%%MD_JSON%%":  md_payload,
}

# ── 6. HTML template ───────────────────────────────────────────────────────────
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STITCH Dashboard</title>
<style>
/* ── Tokens ──────────────────────────────────────────────────────────── */
:root{
  --bg:#0e1118;--surface:#161b27;--card:#1c2130;--border:#252d40;
  --text:#d8dce8;--muted:#5e6580;--faint:#2a3148;
  --accent:#4d9ef7;--green:#3dbf82;--red:#e05c5c;--yellow:#d4901a;
  --cam1:#e07050;--cam2:#4d9ef7;--cam3:#2ec492;--cam4:#d4901a;
  --tr:#4d9ef7;--va:#d4901a;--te:#3dbf82;
}
@media(prefers-color-scheme:light){:root{
  --bg:#f0f2f7;--surface:#fff;--card:#fff;--border:#d4d8e8;
  --text:#1a1e2e;--muted:#7a8099;--faint:#e4e8f2;
  --accent:#2272d8;--green:#1e9e60;--red:#c53030;--yellow:#b07010;
  --cam1:#c05030;--cam2:#2272d8;--cam3:#168a5a;--cam4:#b07010;
  --tr:#2272d8;--va:#b07010;--te:#1e9e60;
}}
:root[data-theme="dark"]{
  --bg:#0e1118;--surface:#161b27;--card:#1c2130;--border:#252d40;
  --text:#d8dce8;--muted:#5e6580;--faint:#2a3148;
  --accent:#4d9ef7;--green:#3dbf82;--red:#e05c5c;--yellow:#d4901a;
  --cam1:#e07050;--cam2:#4d9ef7;--cam3:#2ec492;--cam4:#d4901a;
  --tr:#4d9ef7;--va:#d4901a;--te:#3dbf82;
}
:root[data-theme="light"]{
  --bg:#f0f2f7;--surface:#fff;--card:#fff;--border:#d4d8e8;
  --text:#1a1e2e;--muted:#7a8099;--faint:#e4e8f2;
  --accent:#2272d8;--green:#1e9e60;--red:#c53030;--yellow:#b07010;
  --cam1:#c05030;--cam2:#2272d8;--cam3:#168a5a;--cam4:#b07010;
  --tr:#2272d8;--va:#b07010;--te:#1e9e60;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;
  font-size:13px;display:flex;flex-direction:column;min-height:100vh;overflow:hidden}

/* ── Chrome ────────────────────────────────────────────────────────────── */
.hdr{height:48px;background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0}
.hdr-title{font-size:15px;font-weight:700;letter-spacing:-.02em}
.hdr-sub{font-size:11px;color:var(--muted);font-family:ui-monospace,monospace}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.legend{display:flex;align-items:center;gap:10px;font-size:11px;color:var(--muted)}
.leg-item{display:flex;align-items:center;gap:4px}
.leg-sw{width:18px;height:3px;border-radius:2px}
.theme-btn{background:var(--faint);border:1px solid var(--border);color:var(--muted);
  border-radius:6px;padding:3px 9px;font-size:11px;cursor:pointer}
.theme-btn:hover{color:var(--text)}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.tab-bar{
  display:flex;background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 16px;gap:2px;flex-shrink:0;
}
.tab-btn{
  padding:9px 16px;font-size:12px;font-weight:600;border:none;background:none;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  transition:color .12s;letter-spacing:.01em;
}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-panel{display:none;flex:1;min-height:0;overflow:hidden}
.tab-panel.active{display:flex;flex-direction:column}

/* ── Stats bar ────────────────────────────────────────────────────────── */
.stats-bar{display:flex;background:var(--surface);border-bottom:1px solid var(--border);
  gap:1px;flex-shrink:0}
.stat{flex:1;padding:8px 14px;background:var(--surface);display:flex;flex-direction:column;gap:1px}
.sv{font-size:17px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.sv-b{color:var(--accent)}.sv-g{color:var(--green)}.sv-r{color:var(--red)}
.sl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}

/* ── LC Tab Layout ────────────────────────────────────────────────────── */
.lc-layout{display:flex;flex:1;min-height:0;overflow:hidden}
.sidebar{width:218px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--border);
  overflow-y:auto;padding:11px;display:flex;flex-direction:column;gap:13px}
.main{flex:1;overflow-y:auto;padding:13px;display:flex;flex-direction:column;gap:10px;min-width:0}

/* ── Sidebar controls ─────────────────────────────────────────────────── */
.fgl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px}
.fgr{display:flex;flex-wrap:wrap;gap:3px}
.tog{padding:3px 8px;border-radius:5px;border:1px solid var(--border);background:var(--card);
  color:var(--muted);cursor:pointer;font-size:11px;font-weight:500;transition:background .1s,color .1s}
.tog:hover{color:var(--text)}
.tog.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog.c1.active{background:var(--cam1);border-color:var(--cam1)}
.tog.c2.active{background:var(--cam2);border-color:var(--cam2)}
.tog.c3.active{background:var(--cam3);border-color:var(--cam3)}
.tog.c4.active{background:var(--cam4);border-color:var(--cam4)}
.tog.tg.active{background:var(--green);border-color:var(--green)}
.tog.tr.active{background:var(--red);border-color:var(--red)}
.rr{display:flex;align-items:center;gap:4px;margin-top:3px}
.rl{font-size:11px;color:var(--muted);min-width:24px}
.ri{width:72px;background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:5px;padding:3px 6px;font-size:12px;font-variant-numeric:tabular-nums;outline:none}
.ri:focus{border-color:var(--accent)}
.rs{color:var(--muted);font-size:11px}
.sel{width:100%;background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:5px;padding:4px 7px;font-size:12px;outline:none;cursor:pointer}
.sel:focus{border-color:var(--accent)}
.tinp{width:100%;background:var(--card);border:1px solid var(--border);color:var(--text);
  border-radius:5px;padding:4px 7px;font-size:12px;outline:none}
.tinp::placeholder{color:var(--muted)}
.tinp:focus{border-color:var(--accent)}
.reset-btn{width:100%;padding:6px;background:var(--faint);border:1px solid var(--border);
  color:var(--muted);border-radius:6px;cursor:pointer;font-size:12px}
.reset-btn:hover{color:var(--text);background:var(--border)}

/* ── Card grid ────────────────────────────────────────────────────────── */
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.rc{font-size:12px;color:var(--muted)}.rc strong{color:var(--text)}
.vbtns{display:flex;gap:3px;margin-left:auto}
.vbtn{padding:3px 9px;border-radius:5px;border:1px solid var(--border);background:var(--card);
  color:var(--muted);cursor:pointer;font-size:11px}
.vbtn.active{background:var(--accent);border-color:var(--accent);color:#fff}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(254px,1fr));gap:8px}
.card-grid.wide{grid-template-columns:repeat(auto-fill,minmax(338px,1fr))}
.star-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:9px 11px 7px;cursor:pointer;transition:border-color .12s;
  display:flex;flex-direction:column;gap:6px}
.star-card:hover{border-color:var(--accent)}
.ct{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.tl{font-size:12px;font-weight:700;color:var(--text);flex:1;font-variant-numeric:tabular-nums}
.badge{padding:2px 6px;border-radius:4px;font-size:11px;font-weight:600;font-variant-numeric:tabular-nums}
.bc1{background:color-mix(in srgb,var(--cam1)18%,var(--faint));color:var(--cam1)}
.bc2{background:color-mix(in srgb,var(--cam2)18%,var(--faint));color:var(--cam2)}
.bc3{background:color-mix(in srgb,var(--cam3)18%,var(--faint));color:var(--cam3)}
.bc4{background:color-mix(in srgb,var(--cam4)18%,var(--faint));color:var(--cam4)}
.bg{background:color-mix(in srgb,var(--green)16%,var(--faint));color:var(--green)}
.br{background:color-mix(in srgb,var(--red)16%,var(--faint));color:var(--red)}
.bm{background:var(--faint);color:var(--muted)}
.bsh{background:color-mix(in srgb,var(--green)14%,var(--faint));color:var(--green)}
.bsm{background:color-mix(in srgb,var(--yellow)16%,var(--faint));color:var(--yellow)}
.bsl{background:color-mix(in srgb,var(--red)14%,var(--faint));color:var(--red)}
.sw{border-radius:4px;overflow:hidden;background:var(--bg)}
.sw svg{display:block}
.cb{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.st{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}
.st strong{color:var(--text)}
.load-more{text-align:center;padding:12px}
.load-btn{padding:6px 20px;background:var(--faint);border:1px solid var(--border);
  border-radius:8px;color:var(--text);cursor:pointer;font-size:12px}
.load-btn:hover{background:var(--border)}

/* ── Modal (centered) ────────────────────────────────────────────────── */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);
  z-index:200;align-items:center;justify-content:center;padding:24px}
.overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  width:100%;max-width:860px;max-height:80vh;overflow-y:auto;
  display:flex;flex-direction:column;gap:10px;padding:18px 22px 22px;
  box-shadow:0 24px 60px rgba(0,0,0,.45)}
.mhdr{display:flex;align-items:flex-start;gap:12px}
.mtitle{font-size:17px;font-weight:700;letter-spacing:-.02em}
.mmeta{font-size:12px;color:var(--muted);font-family:ui-monospace,monospace;line-height:1.7;margin-top:4px}
.close-btn{margin-left:auto;background:var(--faint);border:1px solid var(--border);
  color:var(--muted);border-radius:6px;padding:3px 9px;cursor:pointer;flex-shrink:0}
.close-btn:hover{color:var(--text)}
.canv-wrap{overflow-x:auto;border-radius:6px;background:var(--card)}
canvas{display:block}

/* ── Dataset Explorer Tab ─────────────────────────────────────────────── */
.exp-layout{display:flex;flex-direction:column;flex:1;overflow:hidden}
.exp-fbar{
  display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:8px 14px;flex-shrink:0;
}
.exp-fbar .fgl{margin-bottom:0}
.exp-body{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
.chart-card.full{grid-column:1/-1}
.cc-title{font-size:12px;font-weight:700;color:var(--text);margin-bottom:10px;letter-spacing:.01em}
.cc-sub{font-size:11px;color:var(--muted);margin-bottom:10px}
.chart-area{overflow-x:auto}
.chart-area svg{display:block}
.split-legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin-bottom:8px}
.split-leg-item{display:flex;align-items:center;gap:5px}
.sli{width:18px;height:3px;border-radius:2px}
.sli-tr{background:var(--tr)}
.sli-va{background:var(--va)}
.sli-te{background:var(--te)}
</style>
</head>
<body>

<!-- Header -->
<header class="hdr">
  <span class="hdr-title">STITCH</span>
  <span class="hdr-sub">%%MODEL%%</span>
  <div class="hdr-right">
    <div class="legend">
      <div class="leg-item"><div class="leg-sw" style="background:rgba(150,155,180,.5)"></div>Raw</div>
      <div class="leg-item"><div class="leg-sw" style="background:var(--accent)"></div>STITCH</div>
    </div>
    <button class="theme-btn" onclick="toggleTheme()">&#9680;</button>
  </div>
</header>

<!-- Tabs -->
<div class="tab-bar">
  <button class="tab-btn active" id="tab-lc-btn" onclick="switchTab('lc')">&#9673; Light Curves</button>
  <button class="tab-btn" id="tab-exp-btn" onclick="switchTab('exp')">&#9638; Dataset Explorer</button>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<!-- TAB 1: Light Curves                                                        -->
<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<div class="tab-panel active" id="tab-lc">

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat"><span class="sv sv-b" id="visCount">%%TOTAL%%</span><span class="sl">Showing</span></div>
    <div class="stat"><span class="sv">%%TOTAL%%</span><span class="sl">Test stars</span></div>
    <div class="stat"><span class="sv sv-g">%%MEDIMPROV%%</span><span class="sl">Median improvement</span></div>
    <div class="stat"><span class="sv sv-r">%%HARMRATE%%</span><span class="sl">Harmed (n&#8805;5)</span></div>
    <div class="stat"><span class="sv">%%NCVZ%%</span><span class="sl">CVZ (n&#8805;30)</span></div>
  </div>

  <div class="lc-layout">
    <!-- Sidebar -->
    <aside class="sidebar">
      <div><div class="fgl">Camera</div>
        <div class="fgr" id="camTogs">
          <button class="tog active" data-cam="0">All</button>
          <button class="tog c1" data-cam="1">Cam 1</button>
          <button class="tog c2" data-cam="2">Cam 2</button>
          <button class="tog c3" data-cam="3">Cam 3</button>
          <button class="tog c4" data-cam="4">Cam 4</button>
        </div>
      </div>
      <div><div class="fgl">CCD</div>
        <div class="fgr" id="ccdTogs">
          <button class="tog active" data-ccd="0">All</button>
          <button class="tog" data-ccd="1">CCD 1</button>
          <button class="tog" data-ccd="2">CCD 2</button>
          <button class="tog" data-ccd="3">CCD 3</button>
          <button class="tog" data-ccd="4">CCD 4</button>
        </div>
      </div>
      <div><div class="fgl">Status</div>
        <div class="fgr" id="statusTogs">
          <button class="tog active" data-st="all">All</button>
          <button class="tog tg" data-st="improved">Improved</button>
          <button class="tog tr" data-st="harmed">Harmed</button>
        </div>
      </div>
      <div><div class="fgl">N Sectors</div>
        <div class="rr"><span class="rl">Min</span>
          <input class="ri" type="number" id="nMin" value="1" min="1" max="%%N_MAX%%">
          <span class="rs">–</span>
          <input class="ri" type="number" id="nMax" value="%%N_MAX%%" min="1" max="%%N_MAX%%">
        </div>
      </div>
      <div><div class="fgl">Tmag</div>
        <div class="rr"><span class="rl">Min</span>
          <input class="ri" type="number" id="tmagMin" value="%%TMAG_MIN%%" step="0.1">
          <span class="rs">–</span>
          <input class="ri" type="number" id="tmagMax" value="%%TMAG_MAX%%" step="0.1">
        </div>
      </div>
      <div><div class="fgl">Scatter Before (%)</div>
        <div class="rr"><span class="rl">Min</span>
          <input class="ri" type="number" id="scMin" value="0" step="0.1">
          <span class="rs">–</span>
          <input class="ri" type="number" id="scMax" value="%%SC_MAX%%" step="0.1">
        </div>
      </div>
      <div><div class="fgl">TARS Sys Score &#8805;</div>
        <div class="rr"><input class="ri" style="width:100%" type="number" id="sysMin"
          value="0" min="0" max="1" step="0.001" placeholder="0=all"></div>
      </div>
      <div><div class="fgl">Improvement (%)</div>
        <div class="rr"><span class="rl">Min</span>
          <input class="ri" type="number" id="impMin" value="%%IMP_MIN%%" step="1">
          <span class="rs">–</span>
          <input class="ri" type="number" id="impMax" value="100" step="1">
        </div>
      </div>
      <div><div class="fgl">Sort By</div>
        <select class="sel" id="sortSel">
          <option value="improv_d">Improvement &#8595;</option>
          <option value="improv_a">Improvement &#8593;</option>
          <option value="scbef_d">Scatter Before &#8595;</option>
          <option value="scbef_a">Scatter Before &#8593;</option>
          <option value="scaft_a">Scatter After &#8593;</option>
          <option value="n_d">N Sectors &#8595;</option>
          <option value="n_a">N Sectors &#8593;</option>
          <option value="tmag_a">Tmag &#8593;</option>
          <option value="tmag_d">Tmag &#8595;</option>
          <option value="sys_d">Sys Score &#8595;</option>
        </select>
      </div>
      <div><div class="fgl">TIC Search</div>
        <input class="tinp" type="text" id="ticSearch" placeholder="e.g. 441612383">
      </div>
      <button class="reset-btn" id="resetBtn">Reset Filters</button>
    </aside>

    <!-- Main card area -->
    <main class="main">
      <div class="toolbar">
        <span class="rc" id="rcnt"><strong>—</strong> stars</span>
        <div class="vbtns">
          <button class="vbtn active" onclick="setView('small')">Compact</button>
          <button class="vbtn" onclick="setView('wide')">Wide</button>
        </div>
      </div>
      <div class="card-grid" id="cardGrid"></div>
      <div class="load-more" id="loadMore" style="display:none">
        <button class="load-btn" id="loadBtn">Show more</button>
      </div>
    </main>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<!-- TAB 2: Dataset Explorer                                                    -->
<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<div class="tab-panel" id="tab-exp">
  <div class="exp-layout">

    <!-- Filter bar -->
    <div class="exp-fbar">
      <div style="display:flex;align-items:center;gap:6px">
        <span class="fgl">Split</span>
        <div class="fgr" id="expSplitTogs">
          <button class="tog active" data-sp="0" style="background:color-mix(in srgb,var(--tr)20%,var(--faint));color:var(--tr);border-color:var(--tr)">Train</button>
          <button class="tog active" data-sp="1" style="background:color-mix(in srgb,var(--va)20%,var(--faint));color:var(--va);border-color:var(--va)">Val</button>
          <button class="tog active" data-sp="2" style="background:color-mix(in srgb,var(--te)20%,var(--faint));color:var(--te);border-color:var(--te)">Test</button>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span class="fgl">Cam</span>
        <div class="fgr" id="expCamTogs">
          <button class="tog active" data-ec="0">All</button>
          <button class="tog c1" data-ec="1">Cam 1</button>
          <button class="tog c2" data-ec="2">Cam 2</button>
          <button class="tog c3" data-ec="3">Cam 3</button>
          <button class="tog c4" data-ec="4">Cam 4</button>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span class="fgl">CCD</span>
        <div class="fgr" id="expCcdTogs">
          <button class="tog active" data-ecc="0">All</button>
          <button class="tog" data-ecc="1">CCD 1</button>
          <button class="tog" data-ecc="2">CCD 2</button>
          <button class="tog" data-ecc="3">CCD 3</button>
          <button class="tog" data-ecc="4">CCD 4</button>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <span class="fgl">N sectors</span>
        <input class="ri" type="number" id="expNMin" value="1" min="1" placeholder="min" style="width:58px">
        <span class="rs">–</span>
        <input class="ri" type="number" id="expNMax" value="%%N_MAX%%" placeholder="max" style="width:58px">
      </div>
      <span id="expCount" style="font-size:11px;color:var(--muted);margin-left:auto"></span>
    </div>

    <!-- Charts -->
    <div class="exp-body">
      <div class="split-legend">
        <div class="split-leg-item"><div class="sli sli-tr"></div>Train</div>
        <div class="split-leg-item"><div class="sli sli-va"></div>Val</div>
        <div class="split-leg-item"><div class="sli sli-te"></div>Test</div>
      </div>

      <!-- N Sectors by Camera (full width) -->
      <div class="chart-card full">
        <div class="cc-title">N Sectors Distribution by Camera</div>
        <div class="cc-sub">Box plots: median, IQR, whiskers (1.5&#215;IQR). Split = color.</div>
        <div class="chart-area" id="chartNsecCam"></div>
      </div>

      <div class="chart-row">
        <!-- N sectors histogram -->
        <div class="chart-card">
          <div class="cc-title">N Sectors Distribution</div>
          <div class="chart-area" id="chartNsecHist"></div>
        </div>
        <!-- Tmag histogram -->
        <div class="chart-card">
          <div class="cc-title">Tmag Distribution</div>
          <div class="chart-area" id="chartTmag"></div>
        </div>
        <!-- Sector scatter histogram -->
        <div class="chart-card">
          <div class="cc-title">Sector Scatter Distribution (%)</div>
          <div class="chart-area" id="chartSc"></div>
        </div>
        <!-- Cam × CCD heatmap -->
        <div class="chart-card">
          <div class="cc-title">Stars per Cam &#215; CCD</div>
          <div class="chart-area" id="chartCamCcd"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<!-- Detail Modal (centered)                                                    -->
<!-- ═══════════════════════════════════════════════════════════════════════════ -->
<div class="overlay" id="overlay">
  <div class="modal">
    <div class="mhdr">
      <div>
        <div class="mtitle" id="mTitle"></div>
        <div class="mmeta" id="mMeta"></div>
      </div>
      <button class="close-btn" id="closeBtn">&#x2715;</button>
    </div>
    <div class="canv-wrap"><canvas id="plot"></canvas></div>
  </div>
</div>

<script>
// ── Data ───────────────────────────────────────────────────────────────────────
const LC   = %%LC_JSON%%;
const MD   = %%MD_JSON%%;   // {split,cam,ccd,n,tmag,sc,sys} — all splits
const ALL  = LC.stars;
const M    = LC.meta;
const NCOL = {1:"var(--cam1)",2:"var(--cam2)",3:"var(--cam3)",4:"var(--cam4)"};
const SCOL = ["var(--tr)","var(--va)","var(--te)"];
const SNAME = ["Train","Val","Test"];
const PAGE = 60;

// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(id) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("tab-"+id).classList.add("active");
  document.getElementById("tab-"+id+"-btn").classList.add("active");
  if (id === "exp") renderExplorer();
}

// ════════════════════════════════════════════════════════════════════════════════
// LIGHT CURVES TAB
// ════════════════════════════════════════════════════════════════════════════════
let filtered=[], shown=PAGE, viewMode="small";
let fCam=0,fCcd=0,fSt="all",fNMin=1,fNMax=M.n_max;
let fTMin=M.tmag_min,fTMax=M.tmag_max,fScMin=0,fScMax=M.sc_max;
let fSys=0,fIMin=M.improv_min,fIMax=100,fTic="",sortKey="improv_d";

function gv(s,k){
  if(k.startsWith("improv")) return s.improv;
  if(k.startsWith("scbef")) return s.sc_before;
  if(k.startsWith("scaft")) return s.sc_after;
  if(k.startsWith("n_")) return s.n;
  if(k.startsWith("tmag")) return s.tmag??99;
  if(k.startsWith("sys")) return s.sys_score??0;
  return s.improv;
}
function runLC(){
  filtered = ALL.filter(s=>{
    if(fCam && s.cam!==fCam) return false;
    if(fCcd && s.ccd!==fCcd) return false;
    if(fSt==="improved"&&s.harmed) return false;
    if(fSt==="harmed"&&!s.harmed) return false;
    if(s.n<fNMin||s.n>fNMax) return false;
    if((s.tmag??99)<fTMin||(s.tmag??99)>fTMax) return false;
    if(s.sc_before<fScMin||s.sc_before>fScMax) return false;
    if(fSys>0&&(s.sys_score==null||s.sys_score<fSys)) return false;
    if(s.improv<fIMin||s.improv>fIMax) return false;
    if(fTic&&!String(s.tic_id).includes(fTic)) return false;
    return true;
  });
  const asc=sortKey.endsWith("_a"), d=asc?1:-1;
  filtered.sort((a,b)=>{const av=gv(a,sortKey),bv=gv(b,sortKey);return(av<bv?-1:av>bv?1:0)*d});
  shown=PAGE;
  document.getElementById("visCount").textContent=filtered.length.toLocaleString();
  document.getElementById("rcnt").innerHTML=`<strong>${filtered.length.toLocaleString()}</strong> stars`;
  renderLC();
}

function spark(raw,cor,col,w,h){
  const n=raw.length;
  if(!n) return `<svg width="${w}" height="${h}"></svg>`;
  const all=[...raw,...cor];
  let lo=Math.min(...all),hi=Math.max(...all);
  const pad=Math.max((hi-lo)*0.2,0.005); lo-=pad; hi+=pad;
  const rng=hi-lo;
  const PL=3,PR=3,PT=5,PB=5,pw=w-PL-PR,ph=h-PT-PB;
  const px=i=>PL+(n===1?pw/2:i/(n-1)*pw);
  const py=v=>PT+(1-(v-lo)/rng)*ph;
  const y1=py(1.0);
  const rp=raw.map((v,i)=>`${px(i).toFixed(1)},${py(v).toFixed(1)}`);
  const cp=cor.map((v,i)=>`${px(i).toFixed(1)},${py(v).toFixed(1)}`);
  const ra=`M${rp[0]}L${rp.join("L")}L${(PL+pw).toFixed(1)},${y1.toFixed(1)}L${PL},${y1.toFixed(1)}Z`;
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <path d="${ra}" fill="rgba(150,155,180,.12)" stroke="none"/>
    <path d="M${rp[0]}L${rp.join("L")}" fill="none" stroke="rgba(150,155,180,.4)" stroke-width="1"/>
    <line x1="${PL}" y1="${y1.toFixed(1)}" x2="${(PL+pw).toFixed(1)}" y2="${y1.toFixed(1)}"
          stroke="rgba(255,255,255,.06)" stroke-width=".5" stroke-dasharray="2,2"/>
    <path d="M${cp[0]}L${cp.join("L")}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}

function mkCard(s,spW){
  const col=NCOL[s.cam]||"var(--muted)";
  const iCls=s.harmed?"br":"bg";
  const iStr=(s.harmed?"▼":"▲")+Math.abs(s.improv).toFixed(1)+"%";
  let sb="";
  if(s.sys_score!=null){
    const sc=s.sys_score>=0.999?"bsh":s.sys_score>=0.995?"bsm":"bsl";
    sb=`<span class="badge ${sc}">sys ${s.sys_score.toFixed(4)}</span>`;
  }
  const sv=spark(s.raw_meds,s.cor_meds,col,spW,60);
  const tm=s.tmag!=null?`<span class="badge bm">T=${s.tmag.toFixed(1)}</span>`:"";
  return `<div class="star-card" onclick="detail(${s.tic_id})">
    <div class="ct"><span class="tl">TIC ${s.tic_id}</span>
      <span class="badge bc${s.cam}">Cam${s.cam}&#xB7;CCD${s.ccd}</span>
      <span class="badge ${iCls}">${iStr}</span></div>
    <div class="sw">${sv}</div>
    <div class="cb">
      <span class="st"><strong>${s.sc_before.toFixed(2)}%</strong>&#8594;<strong>${s.sc_after.toFixed(2)}%</strong></span>
      <span class="badge bm">n=${s.n}</span>${tm}${sb}</div>
  </div>`;
}

function renderLC(){
  const grid=document.getElementById("cardGrid");
  const spW=viewMode==="wide"?314:230;
  grid.innerHTML=filtered.slice(0,shown).map(s=>mkCard(s,spW)).join("");
  const more=document.getElementById("loadMore");
  const rem=filtered.length-shown;
  if(rem>0){more.style.display="block";
    document.getElementById("loadBtn").textContent=`Show more (${rem} remaining)`;
  } else more.style.display="none";
}

function setView(m){
  viewMode=m;
  document.getElementById("cardGrid").classList.toggle("wide",m==="wide");
  document.querySelectorAll(".vbtn").forEach((b,i)=>b.classList.toggle("active",i===(m==="wide"?1:0)));
  renderLC();
}

// Modal (centered)
function detail(tic){
  const s=ALL.find(x=>x.tic_id===tic);
  if(!s) return;
  document.getElementById("overlay").classList.add("open");
  document.getElementById("mTitle").textContent=`TIC ${s.tic_id}`;
  const sys=s.sys_score!=null?` · sys ${s.sys_score.toFixed(4)}`:"";
  document.getElementById("mMeta").innerHTML=
    `Cam ${s.cam} / CCD ${s.ccd} · Tmag ${s.tmag!=null?s.tmag.toFixed(2):"—"} · ${s.n} sectors${sys}<br>`+
    `&#963; before = ${s.sc_before.toFixed(3)}%  &#8594;  after = ${s.sc_after.toFixed(3)}%  `+
    `(${s.harmed?"▼":"▲"}${Math.abs(s.improv).toFixed(1)}% ${s.harmed?"worse":"improvement"})`;
  drawPlot(s);
}

function drawPlot(s){
  const canvas=document.getElementById("plot");
  const wrap=canvas.parentElement;
  const n=s.sectors.length;
  const BAR=13,GAP=4,GRP=BAR*2+GAP+7;
  const W=Math.max(wrap.clientWidth||600,n*GRP+80),H=130;
  canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext("2d");
  const dark=document.documentElement.dataset.theme!=="light"&&
    (document.documentElement.dataset.theme==="dark"||window.matchMedia("(prefers-color-scheme:dark)").matches);
  const sfC=dark?"#1c2130":"#fff", grC=dark?"#252d40":"#dde0ee";
  const muC=dark?"#5e6580":"#7a8099", rC=dark?"#3a4060":"#b0b5cc";
  const cc={1:dark?"#e07050":"#c05030",2:dark?"#4d9ef7":"#2272d8",3:dark?"#2ec492":"#168a5a",4:dark?"#d4901a":"#b07010"};
  const camC=cc[s.cam]||"#888";
  const PL=48,PR=10,PT=12,PB=22,pH=H-PT-PB,pW=W-PL-PR;
  const all=[...s.raw_meds,...s.cor_meds];
  let lo=Math.min(...all),hi=Math.max(...all);
  const vp=Math.max((hi-lo)*0.2,0.005); lo-=vp; hi+=vp;
  const yS=v=>PT+pH*(1-(v-lo)/(hi-lo));
  ctx.fillStyle=sfC; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle=grC; ctx.lineWidth=1;
  ctx.fillStyle=muC; ctx.font="10px ui-monospace,monospace"; ctx.textAlign="right";
  for(let i=0;i<=4;i++){
    const v=lo+(hi-lo)*i/4,yy=yS(v);
    ctx.beginPath(); ctx.moveTo(PL,yy); ctx.lineTo(W-PR,yy); ctx.stroke();
    ctx.fillText(v.toFixed(4),PL-4,yy+3);
  }
  const y1=yS(1.0);
  ctx.strokeStyle=dark?"rgba(77,158,247,.35)":"rgba(34,114,216,.35)";
  ctx.lineWidth=.8; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(PL,y1); ctx.lineTo(W-PR,y1); ctx.stroke(); ctx.setLineDash([]);
  s.sectors.forEach((sec,i)=>{
    const x=PL+i*GRP+3,yB=yS(1),yR=yS(s.raw_meds[i]),yC=yS(s.cor_meds[i]);
    ctx.fillStyle=rC;
    ctx.fillRect(x,Math.min(yR,yB),BAR,Math.max(Math.abs(yR-yB),1));
    ctx.fillStyle=s.harmed?"#e05c5c":camC; ctx.globalAlpha=.88;
    ctx.fillRect(x+BAR+GAP,Math.min(yC,yB),BAR,Math.max(Math.abs(yC-yB),1));
    ctx.globalAlpha=1;
    ctx.fillStyle=muC; ctx.font="9px system-ui"; ctx.textAlign="center";
    ctx.fillText(`S${sec}`,x+BAR+GAP/2,H-5);
  });
  ctx.fillStyle=rC; ctx.fillRect(PL,3,12,7);
  ctx.fillStyle=muC; ctx.font="10px system-ui"; ctx.textAlign="left";
  ctx.fillText("Raw",PL+15,10);
  ctx.fillStyle=camC; ctx.fillRect(PL+58,3,12,7);
  ctx.fillText("STITCH",PL+73,10);
}

// LC controls
let dt;
const deb=fn=>{clearTimeout(dt);dt=setTimeout(fn,200)};

document.querySelectorAll("#camTogs .tog").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#camTogs .tog").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); fCam=+b.dataset.cam; runLC();
}));
document.querySelectorAll("#ccdTogs .tog").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#ccdTogs .tog").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); fCcd=+b.dataset.ccd; runLC();
}));
document.querySelectorAll("#statusTogs .tog").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#statusTogs .tog").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); fSt=b.dataset.st; runLC();
}));
["nMin","nMax","tmagMin","tmagMax","scMin","scMax","sysMin","impMin","impMax"].forEach(id=>{
  document.getElementById(id).addEventListener("input",()=>deb(()=>{
    fNMin=+document.getElementById("nMin").value||1;
    fNMax=+document.getElementById("nMax").value||M.n_max;
    fTMin=+document.getElementById("tmagMin").value||M.tmag_min;
    fTMax=+document.getElementById("tmagMax").value||M.tmag_max;
    fScMin=+document.getElementById("scMin").value||0;
    fScMax=+document.getElementById("scMax").value||M.sc_max;
    fSys=+document.getElementById("sysMin").value||0;
    fIMin=+document.getElementById("impMin").value;
    fIMax=+document.getElementById("impMax").value;
    if(isNaN(fIMin))fIMin=M.improv_min;
    if(isNaN(fIMax))fIMax=100;
    runLC();
  }));
});
document.getElementById("sortSel").addEventListener("change",e=>{sortKey=e.target.value;runLC()});
document.getElementById("ticSearch").addEventListener("input",e=>{fTic=e.target.value.trim();runLC()});
document.getElementById("loadBtn").addEventListener("click",()=>{shown+=PAGE;renderLC()});
document.getElementById("closeBtn").addEventListener("click",()=>
  document.getElementById("overlay").classList.remove("open"));
document.getElementById("overlay").addEventListener("click",e=>{
  if(e.target===document.getElementById("overlay"))
    document.getElementById("overlay").classList.remove("open");
});
document.getElementById("resetBtn").addEventListener("click",()=>{
  fCam=0;fCcd=0;fSt="all";fNMin=1;fNMax=M.n_max;
  fTMin=M.tmag_min;fTMax=M.tmag_max;fScMin=0;fScMax=M.sc_max;
  fSys=0;fIMin=M.improv_min;fIMax=100;fTic="";sortKey="improv_d";
  document.querySelectorAll("#camTogs .tog,#ccdTogs .tog,#statusTogs .tog")
    .forEach(b=>b.classList.remove("active"));
  document.querySelector("[data-cam='0']").classList.add("active");
  document.querySelector("[data-ccd='0']").classList.add("active");
  document.querySelector("[data-st='all']").classList.add("active");
  document.getElementById("nMin").value=1;
  document.getElementById("nMax").value=M.n_max;
  document.getElementById("tmagMin").value=M.tmag_min;
  document.getElementById("tmagMax").value=M.tmag_max;
  document.getElementById("scMin").value=0;
  document.getElementById("scMax").value=M.sc_max;
  document.getElementById("sysMin").value=0;
  document.getElementById("impMin").value=M.improv_min;
  document.getElementById("impMax").value=100;
  document.getElementById("ticSearch").value="";
  document.getElementById("sortSel").value="improv_d";
  runLC();
});

// ════════════════════════════════════════════════════════════════════════════════
// DATASET EXPLORER TAB
// ════════════════════════════════════════════════════════════════════════════════
let expSplits=new Set([0,1,2]), expCam=0, expCcd=0, expNMin=1, expNMax=M.n_max;
let expDirty=true;

function getExpSubset(){
  const {split,cam,ccd,n,tmag,sc,sys}=MD;
  const idx=[];
  for(let i=0;i<split.length;i++){
    if(!expSplits.has(split[i])) continue;
    if(expCam && cam[i]!==expCam) continue;
    if(expCcd && ccd[i]!==expCcd) continue;
    if(n[i]<expNMin || n[i]>expNMax) continue;
    idx.push(i);
  }
  return idx;
}

// Box-plot statistics
function boxStats(vals){
  if(!vals.length) return null;
  const s=[...vals].sort((a,b)=>a-b);
  const n=s.length;
  const p=q=>s[Math.max(0,Math.min(n-1,Math.floor(q*(n-1))))];
  const q1=p(.25),q2=p(.5),q3=p(.75);
  const iqr=q3-q1;
  const wlo=Math.max(s[0],q1-1.5*iqr);
  const whi=Math.min(s[n-1],q3+1.5*iqr);
  return {q1,q2,q3,wlo,whi,n,mean:s.reduce((a,b)=>a+b)/n};
}

// Histogram helper
function histogram(vals, lo, hi, bins){
  const step=(hi-lo)/bins;
  const counts=new Array(bins).fill(0);
  for(const v of vals){
    if(v<lo||v>hi) continue;
    const b=Math.min(bins-1,Math.floor((v-lo)/step));
    counts[b]++;
  }
  return counts;
}

// ── Chart: N Sectors by Camera (box plots) ────────────────────────────────────
function renderNsecCam(idx){
  const W=Math.min(document.getElementById("chartNsecCam").clientWidth||720,900);
  const H=220;
  const PL=44,PR=16,PT=14,PB=32;
  const pw=W-PL-PR,ph=H-PT-PB;

  // collect n-values per (cam, split)
  const groups={};
  for(let c=1;c<=4;c++) for(let sp=0;sp<3;sp++) groups[`${c}_${sp}`]=[];
  for(const i of idx){
    const key=`${MD.cam[i]}_${MD.split[i]}`;
    if(groups[key]) groups[key].push(MD.n[i]);
  }

  const allStats=[];
  for(let c=1;c<=4;c++)
    for(let sp=0;sp<3;sp++)
      allStats.push({c,sp,stats:boxStats(groups[`${c}_${sp}`])});

  // Y range: 0 to max-whisker
  const allWhi=[...allStats.map(x=>x.stats?.whi||0)];
  const yMax=Math.max(...allWhi,10);
  const yS=v=>PT+ph*(1-v/yMax);

  // X layout: 4 cam groups, 3 boxes per cam
  const camW=pw/4, bW=Math.min(20, camW/5);
  const camX=c=>PL+(c-1)*camW+camW/2;
  const spOffsets=[-1,0,1];

  let svg=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="overflow:visible">`;

  // Grid lines
  const nGridLines=5;
  svg+=`<style>.gline{stroke:var(--border);stroke-width:.5}.axlbl{font-size:10px;fill:var(--muted);font-family:ui-monospace,monospace}</style>`;
  for(let i=0;i<=nGridLines;i++){
    const v=yMax*i/nGridLines;
    const yy=yS(v);
    svg+=`<line class="gline" x1="${PL}" y1="${yy.toFixed(1)}" x2="${W-PR}" y2="${yy.toFixed(1)}"/>`;
    svg+=`<text class="axlbl" x="${PL-4}" y="${(yy+3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`;
  }

  // Box plots
  for(const {c,sp,stats} of allStats){
    if(!stats || !expSplits.has(sp)) continue;
    const cx=camX(c)+spOffsets[sp]*(bW+4);
    const col=SCOL[sp];
    const q1y=yS(stats.q1),q3y=yS(stats.q3),q2y=yS(stats.q2);
    const wloy=yS(stats.wlo),whiy=yS(stats.whi);
    const boxH=Math.abs(q3y-q1y);
    const boxTop=Math.min(q1y,q3y);

    // whisker lines
    svg+=`<line x1="${cx}" y1="${whiy.toFixed(1)}" x2="${cx}" y2="${wloy.toFixed(1)}" stroke="${col}" stroke-width="1" opacity=".6"/>`;
    // whisker caps
    svg+=`<line x1="${(cx-bW/2).toFixed(1)}" y1="${whiy.toFixed(1)}" x2="${(cx+bW/2).toFixed(1)}" y2="${whiy.toFixed(1)}" stroke="${col}" stroke-width="1.5"/>`;
    svg+=`<line x1="${(cx-bW/2).toFixed(1)}" y1="${wloy.toFixed(1)}" x2="${(cx+bW/2).toFixed(1)}" y2="${wloy.toFixed(1)}" stroke="${col}" stroke-width="1.5"/>`;
    // IQR box
    svg+=`<rect x="${(cx-bW/2).toFixed(1)}" y="${boxTop.toFixed(1)}" width="${bW}" height="${Math.max(boxH,1).toFixed(1)}"
      fill="${col}" opacity=".25" rx="2"/>`;
    svg+=`<rect x="${(cx-bW/2).toFixed(1)}" y="${boxTop.toFixed(1)}" width="${bW}" height="${Math.max(boxH,1).toFixed(1)}"
      fill="none" stroke="${col}" stroke-width="1.5" rx="2"/>`;
    // Median line
    svg+=`<line x1="${(cx-bW/2).toFixed(1)}" y1="${q2y.toFixed(1)}" x2="${(cx+bW/2).toFixed(1)}" y2="${q2y.toFixed(1)}"
      stroke="${col}" stroke-width="2"/>`;
    // Tooltip title (median n)
    svg+=`<title>${SNAME[sp]}: n=${stats.n}, med=${stats.q2.toFixed(0)}, IQR=${stats.q1.toFixed(0)}–${stats.q3.toFixed(0)}</title>`;
  }

  // Cam labels
  for(let c=1;c<=4;c++){
    svg+=`<text class="axlbl" x="${camX(c).toFixed(1)}" y="${H-8}" text-anchor="middle" style="font-size:11px;fill:var(--text);font-weight:600">Cam ${c}</text>`;
  }

  // Y axis label
  svg+=`<text class="axlbl" transform="rotate(-90) translate(-${(PT+ph/2).toFixed(0)},${(PL-30).toFixed(0)})" text-anchor="middle">N Sectors</text>`;

  svg+="</svg>";
  document.getElementById("chartNsecCam").innerHTML=svg;
}

// ── Chart: Histogram helper ────────────────────────────────────────────────────
function renderHist(containerId, title, vals_by_split, lo, hi, nBins, xLabel, clip){
  const cEl=document.getElementById(containerId);
  const W=cEl.clientWidth||340;
  const H=160;
  const PL=40,PR=10,PT=10,PB=30,pw=W-PL-PR,ph=H-PT-PB;
  const step=(hi-lo)/nBins;
  const bins=Array.from({length:nBins},(_,i)=>lo+i*step+step/2);

  // Compute counts per split
  const counts=vals_by_split.map(vals=>{
    const c=histogram(vals,lo,hi,nBins);
    const tot=c.reduce((a,b)=>a+b,0)||1;
    return c.map(x=>x/tot*100);  // % of split
  });

  // Y max
  const yMax=Math.max(...counts.flat(),0.1);
  const yS=v=>PT+ph*(1-v/yMax);
  const xS=i=>PL+i*pw/nBins;

  let svg=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  svg+=`<style>.gline{stroke:var(--border);stroke-width:.5}.axlbl{font-size:9px;fill:var(--muted);font-family:ui-monospace,monospace}</style>`;

  // Grid
  for(let i=0;i<=4;i++){
    const v=yMax*i/4,yy=yS(v);
    svg+=`<line class="gline" x1="${PL}" y1="${yy.toFixed(1)}" x2="${W-PR}" y2="${yy.toFixed(1)}"/>`;
    svg+=`<text class="axlbl" x="${PL-3}" y="${(yy+3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}%</text>`;
  }

  // Area fills per split (back to front: train, val, test)
  counts.forEach((cnt, sp)=>{
    if(!expSplits.has(sp)) return;
    const col=SCOL[sp];
    const pts=cnt.map((v,i)=>`${(xS(i)+pw/nBins/2).toFixed(1)},${yS(v).toFixed(1)}`);
    const base=yS(0).toFixed(1);
    const areaPath=`M${(xS(0)+pw/nBins/2).toFixed(1)},${base}L${pts.join("L")}L${(xS(nBins-1)+pw/nBins/2).toFixed(1)},${base}Z`;
    svg+=`<path d="${areaPath}" fill="${col}" opacity=".15"/>`;
    svg+=`<path d="M${pts.join("L")}" fill="none" stroke="${col}" stroke-width="1.5"/>`;
  });

  // X axis labels (sparse)
  const nLabels=5;
  for(let i=0;i<nBins;i+=Math.max(1,Math.floor(nBins/nLabels))){
    const xp=(xS(i)+pw/nBins/2).toFixed(1);
    const lbl=bins[i].toFixed(bins[i]<10?1:0);
    svg+=`<text class="axlbl" x="${xp}" y="${H-4}" text-anchor="middle">${lbl}${clip&&i===nBins-1?"+":" "}</text>`;
  }
  svg+=`<text class="axlbl" x="${(PL+pw/2).toFixed(1)}" y="${H}" text-anchor="middle" style="font-size:9px">${xLabel}</text>`;
  svg+="</svg>";
  cEl.innerHTML=svg;
}

// ── Chart: Cam × CCD heatmap ──────────────────────────────────────────────────
function renderCamCcd(idx){
  const cEl=document.getElementById("chartCamCcd");
  const W=cEl.clientWidth||340;
  const H=160;
  const PL=36,PR=16,PT=14,PB=24;
  const cellW=(W-PL-PR)/4, cellH=(H-PT-PB)/4;

  // count per cam × ccd × split
  const grid={}; // "cam_ccd" -> {0:n, 1:n, 2:n}
  for(let c=1;c<=4;c++) for(let cc=1;cc<=4;cc++) grid[`${c}_${cc}`]={0:0,1:0,2:0};
  for(const i of idx) grid[`${MD.cam[i]}_${MD.ccd[i]}`][MD.split[i]]++;

  // total per cell
  const totals={};
  for(const k in grid) totals[k]=Object.values(grid[k]).reduce((a,b)=>a+b,0);
  const maxT=Math.max(...Object.values(totals),1);

  let svg=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  svg+=`<style>.axlbl{font-size:9px;fill:var(--muted)}.cellv{font-size:9px;font-family:ui-monospace,monospace;dominant-baseline:middle;text-anchor:middle}</style>`;

  // Column headers (cam)
  for(let c=1;c<=4;c++){
    const cx=PL+(c-1)*cellW+cellW/2;
    svg+=`<text class="axlbl" x="${cx.toFixed(1)}" y="${PT-3}" text-anchor="middle">Cam${c}</text>`;
  }
  // Row headers (ccd)
  for(let cc=1;cc<=4;cc++){
    const cy=PT+(cc-1)*cellH+cellH/2;
    svg+=`<text class="axlbl" x="${PL-4}" y="${cy.toFixed(1)}" text-anchor="end" dominant-baseline="middle">CCD${cc}</text>`;
  }

  // Cells
  for(let c=1;c<=4;c++){
    for(let cc=1;cc<=4;cc++){
      const key=`${c}_${cc}`;
      const tot=totals[key];
      const t=tot/maxT;
      const x=PL+(c-1)*cellW, y=PT+(cc-1)*cellH;
      const alpha=0.08+t*0.55;
      svg+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(cellW-2).toFixed(1)}" height="${(cellH-2).toFixed(1)}" rx="3" fill="var(--accent)" opacity="${alpha.toFixed(2)}"/>`;
      if(tot>0) svg+=`<text class="cellv" x="${(x+cellW/2).toFixed(1)}" y="${(y+cellH/2).toFixed(1)}" fill="var(--text)" opacity=".85">${tot>=1000?(tot/1000).toFixed(1)+"k":tot}</text>`;
    }
  }
  svg+="</svg>";
  cEl.innerHTML=svg;
}

function renderExplorer(){
  const idx=getExpSubset();
  const counts=[0,1,2].map(sp=>idx.filter(i=>MD.split[i]===sp).length);
  document.getElementById("expCount").textContent=
    `Train ${counts[0].toLocaleString()} · Val ${counts[1].toLocaleString()} · Test ${counts[2].toLocaleString()}`;

  renderNsecCam(idx);

  // N sectors histogram (clip at 40)
  const nVals=[...[0,1,2].map(sp=>idx.filter(i=>MD.split[i]===sp).map(i=>Math.min(MD.n[i],40)))];
  renderHist("chartNsecHist","",nVals,0,40,20,"N Sectors",true);

  // Tmag histogram
  const tVals=[...[0,1,2].map(sp=>idx.filter(i=>MD.split[i]===sp).map(i=>MD.tmag[i]).filter(v=>v!=null))];
  renderHist("chartTmag","",tVals,4,14,20,"Tmag",false);

  // Scatter histogram (clip at 8%)
  const scVals=[...[0,1,2].map(sp=>idx.filter(i=>MD.split[i]===sp).map(i=>Math.min(MD.sc[i],8)))];
  renderHist("chartSc","",scVals,0,8,24,"Sector Scatter %",true);

  renderCamCcd(idx);
}

// Explorer controls
document.querySelectorAll("#expSplitTogs .tog").forEach(b=>{
  b.addEventListener("click",()=>{
    const sp=+b.dataset.sp;
    if(expSplits.has(sp)) expSplits.delete(sp);
    else expSplits.add(sp);
    b.classList.toggle("active", expSplits.has(sp));
    renderExplorer();
  });
});
document.querySelectorAll("#expCamTogs .tog").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#expCamTogs .tog").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); expCam=+b.dataset.ec; renderExplorer();
}));
document.querySelectorAll("#expCcdTogs .tog").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll("#expCcdTogs .tog").forEach(x=>x.classList.remove("active"));
  b.classList.add("active"); expCcd=+b.dataset.ecc; renderExplorer();
}));
["expNMin","expNMax"].forEach(id=>document.getElementById(id).addEventListener("input",()=>deb(()=>{
  expNMin=+document.getElementById("expNMin").value||1;
  expNMax=+document.getElementById("expNMax").value||M.n_max;
  renderExplorer();
})));

// ── Theme & init ───────────────────────────────────────────────────────────────
function toggleTheme(){
  const r=document.documentElement;
  const cur=r.dataset.theme||(window.matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.dataset.theme=cur==="dark"?"light":"dark";
  // Redraw explorer charts if visible
  if(document.getElementById("tab-exp").classList.contains("active")) renderExplorer();
}

runLC();
</script>
</body>
</html>"""

# ── Apply substitutions ────────────────────────────────────────────────────────
html = TEMPLATE
for k, v in SUBS.items():
    html = html.replace(k, v)

os.makedirs("eval", exist_ok=True)
with open(HTML_OUT, "w") as f:
    f.write(html)
size_mb = os.path.getsize(HTML_OUT) / 1024 / 1024
print(f"\nSaved → {HTML_OUT}  ({size_mb:.1f} MB)")
