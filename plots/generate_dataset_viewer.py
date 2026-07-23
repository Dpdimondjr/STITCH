"""
Generate self-contained HTML dataset viewer for STITCH training data.
Run from the STITCH root directory:
    python3 plots/generate_dataset_viewer.py
"""
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

OUT = "stitch_dataset_viewer.html"

print("Loading...")
df = pd.read_parquet("training_data.parquet")

# sys_score merged from all available TARS catalog versions
sys_score_map = {}
for _cat in ["tars_quiet_tics.csv", "tars_quiet_tics_v2.csv", "tars_quiet_tics_v2_5k_cap.csv"]:
    try:
        _t = pd.read_csv(_cat)[["tic_id", "mean_sys_score"]]
        for _, _r in _t.iterrows():
            _tic = int(_r["tic_id"])
            if _tic not in sys_score_map:
                sys_score_map[_tic] = _r["mean_sys_score"]
    except FileNotFoundError:
        pass

sc = (df.groupby("tic_id")["cam"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index().rename(columns={"cam": "dom"}))
tr, tmp = train_test_split(sc["tic_id"], test_size=0.2, stratify=sc["dom"], random_state=42)
tmp_cam = sc[sc["tic_id"].isin(tmp)]["dom"]
vl, te = train_test_split(tmp, test_size=0.5, stratify=tmp_cam.values, random_state=42)
splits = {int(t): 0 for t in tr}
splits.update({int(t): 1 for t in vl})
splits.update({int(t): 2 for t in te})

print("Aggregating...")
agg = df.groupby("tic_id").agg(
    tmag=("tmag", "mean"),
    n=("sector", "nunique"),
    cam=("cam", lambda x: int(x.mode()[0])),
    ra=("ra", "first"),
    dec=("dec", "first"),
    sc=("flux_offset", lambda x: float(np.std(x)) * 100),
).reset_index()

# Compact encoding helpers (shared CHARS for offsets and sector list)
CHARS = "0123456789abcdefghijklmnopqrstu"

def encode_offs(lst):
    out = []
    for v in lst:
        idx = max(0, min(30, round((v - 0.85) * 100)))
        out.append(CHARS[idx])
    return "".join(out)

SECT36 = "0123456789abcdefghijklmnopqrstuvwxyz"  # full 36-char base-36

def encode_sects(sectors):
    # Each sector encoded as 2-char base-36 (sectors 1–129 fit in 2 chars)
    out = []
    for s in sorted(int(x) for x in sectors):
        out.append(SECT36[s // 36] + SECT36[s % 36])
    return "".join(out)

print("Building offset strings...")
sorted_df = df.sort_values(["tic_id", "sector"])
offs_raw   = sorted_df.groupby("tic_id")["flux_offset"].apply(list)
sects_raw  = sorted_df.groupby("tic_id")["sector"].apply(lambda x: sorted(x.unique().tolist()))

print("Serialising rows...")
rows = []
for _, r in agg.iterrows():
    tic  = int(r["tic_id"])
    tmag = float(r["tmag"])
    ra   = float(r["ra"])
    dec  = float(r["dec"])
    sc_v = float(r["sc"])
    offs_str  = encode_offs(offs_raw.get(tic, []))
    sects_str = encode_sects(sects_raw.get(tic, []))
    sys_s = sys_score_map.get(tic, None)
    rows.append([
        tic,
        round(tmag, 1) if not np.isnan(tmag) else 0,
        int(r["n"]),
        int(r["cam"]),
        round(ra, 1)   if not np.isnan(ra)   else 0,
        round(dec, 1)  if not np.isnan(dec)  else 0,
        round(sc_v, 2) if not np.isnan(sc_v) else 0,
        splits.get(tic, 0),
        offs_str,
        sects_str,
        round(float(sys_s), 3) if sys_s is not None and not np.isnan(float(sys_s)) else None,
    ])

rows.sort(key=lambda x: -x[6])
data_js = json.dumps(rows, separators=(",", ":"))
print(f"  {len(rows):,} TICs  ·  {len(data_js)/1024:.0f} KB")

HTML = """\
<!doctype html>
<title>STITCH · Dataset Viewer</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#f7f6f4;--bg2:#efede9;--bg3:#e6e4df;
  --ink:#17160f;--ink2:#4e4d47;--muted:#8c8b84;
  --border:#dddbd4;--accent:#2a78d6;--acc-dim:#d0e4f7;
  --hover:#eceae6;
  --c1:#e05c3a;--c2:#2a78d6;--c3:#1baf7a;--c4:#c47900;
  --tr:#2a78d6;--vl:#c47900;--te:#e05c3a;
  --tr-bg:#d0e4f7;--vl-bg:#fcefd3;--te-bg:#fad4cc;
  --font:system-ui,-apple-system,sans-serif;
  --mono:'SF Mono','Cascadia Code','Fira Code',ui-monospace,monospace;
}
@media(prefers-color-scheme:dark){:root{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --ink:#dddbd3;--ink2:#9a9890;--muted:#555750;
  --border:#2a2f38;--accent:#4d9de0;--acc-dim:#132236;
  --hover:#1c2128;
  --tr-bg:#132236;--vl-bg:#2d2000;--te-bg:#2d0e08;
}}
:root[data-theme=dark]{
  --bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;
  --ink:#dddbd3;--ink2:#9a9890;--muted:#555750;
  --border:#2a2f38;--accent:#4d9de0;--acc-dim:#132236;
  --hover:#1c2128;
  --tr-bg:#132236;--vl-bg:#2d2000;--te-bg:#2d0e08;
}
:root[data-theme=light]{
  --bg:#f7f6f4;--bg2:#efede9;--bg3:#e6e4df;
  --ink:#17160f;--ink2:#4e4d47;--muted:#8c8b84;
  --border:#dddbd4;--accent:#2a78d6;--acc-dim:#d0e4f7;
  --hover:#eceae6;
  --tr-bg:#d0e4f7;--vl-bg:#fcefd3;--te-bg:#fad4cc;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{color-scheme:light dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font);font-size:13px;line-height:1.5;min-height:100vh}

.hdr{background:var(--bg);border-bottom:1px solid var(--border);
  padding:10px 18px;display:flex;align-items:center;gap:12px}
.hdr-title{font-size:14px;font-weight:650;letter-spacing:-.02em}
.hdr-title span{color:var(--muted);font-weight:400}
.hdr-sub{font-size:11px;color:var(--muted);flex:1}
.hdr-badge{font-size:11px;font-variant-numeric:tabular-nums;background:var(--bg2);
  border:1px solid var(--border);border-radius:20px;padding:2px 10px;color:var(--ink2)}

.flt{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:9px 18px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
.fg{display:flex;align-items:center;gap:6px}
.fl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);white-space:nowrap}
.rw{display:flex;align-items:center;gap:4px}
.rv{font-size:11px;font-variant-numeric:tabular-nums;color:var(--ink2);min-width:26px;text-align:center}
input[type=range]{-webkit-appearance:none;width:76px;height:3px;background:var(--border);border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:var(--accent);cursor:pointer}
.tg{display:flex;gap:2px}
.tb{font-size:11px;padding:2px 7px;border:1px solid var(--border);border-radius:3px;
  background:transparent;color:var(--ink2);cursor:pointer;font-family:var(--font)}
.tb:hover{background:var(--bg3)}
.tb.on{background:var(--accent);border-color:var(--accent);color:#fff}
.srch{font-size:12px;font-family:var(--mono);padding:3px 8px;border:1px solid var(--border);
  border-radius:3px;background:var(--bg);color:var(--ink);width:120px;outline:none}
.srch:focus{border-color:var(--accent)}

.rbar{position:sticky;top:0;z-index:40;padding:6px 18px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--border);background:var(--bg)}
.showing{font-size:11px;color:var(--muted);flex:1}
.srt-wr{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted)}
select{font-size:11px;font-family:var(--font);padding:2px 5px;border:1px solid var(--border);
  border-radius:3px;background:var(--bg);color:var(--ink);outline:none;cursor:pointer}
.pgn{display:flex;align-items:center;gap:5px}
.pb{width:24px;height:24px;display:flex;align-items:center;justify-content:center;
  border:1px solid var(--border);border-radius:3px;background:transparent;color:var(--ink2);cursor:pointer;font-size:12px}
.pb:disabled{opacity:.3;cursor:default}
.pi{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;min-width:64px;text-align:center}
.dl-btn{font-size:11px;padding:2px 8px;border:1px solid var(--border);border-radius:3px;
  background:transparent;color:var(--ink2);cursor:pointer;font-family:var(--font)}
.dl-btn:hover{background:var(--bg2)}

.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--bg2);padding:6px 10px;text-align:left;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--border);
  white-space:nowrap;user-select:none;cursor:pointer;position:sticky;top:0;z-index:10}
thead th:hover{color:var(--ink)}
thead th[data-sort].on{color:var(--accent)}
thead th[data-sort].on::after{content:' ↓'}
thead th[data-sort].on.asc::after{content:' ↑'}
tbody tr{border-bottom:1px solid var(--border)}
tbody tr:hover{background:var(--hover)}
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
.scell{display:flex;align-items:center;gap:6px;min-width:130px}
.sbar-bg{flex:1;height:4px;background:var(--border);border-radius:2px;min-width:52px;position:relative}
.sbar{height:100%;border-radius:2px}
.sbar-med{position:absolute;top:-2px;bottom:-2px;width:1.5px;background:var(--ink2);opacity:.35;border-radius:1px}
.sval{font-size:11px;font-variant-numeric:tabular-nums;color:var(--ink2);min-width:38px;text-align:right}
.pct-pill{font-size:10px;font-weight:600;padding:1px 5px;border-radius:8px;flex-shrink:0}
.spbadge{font-size:10px;font-weight:500;padding:1px 6px;border-radius:10px}
.spark-wrap{display:flex;align-items:center;gap:6px}
.off-range{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.stats-strip{padding:5px 18px;font-size:11px;color:var(--muted);background:var(--bg2);
  border-bottom:1px solid var(--border);display:flex;gap:18px;flex-wrap:wrap}
.ss-item b{color:var(--ink2);font-weight:600;font-variant-numeric:tabular-nums}
.sect-list{display:flex;flex-wrap:wrap;gap:2px;max-width:200px}
.sect-chip{font-size:9.5px;padding:0 3px;height:15px;line-height:15px;border-radius:2px;
  background:var(--bg3);color:var(--ink2);font-variant-numeric:tabular-nums;
  font-family:var(--mono);flex-shrink:0}
.score-val{font-size:11px;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.no-r{padding:60px 18px;text-align:center;color:var(--muted);font-size:13px}
</style>

<div class="hdr">
  <div class="hdr-title">STITCH <span>&middot;</span> Dataset Viewer</div>
  <div class="hdr-sub">SPOC PDCSAP 2-min &middot; Sectors 1&ndash;100 &middot; leave-one-out flux offsets</div>
  <div class="hdr-badge" id="tot-badge">&ndash;</div>
</div>

<div class="flt">
  <div class="fg">
    <span class="fl">TIC ID</span>
    <input class="srch" type="text" id="srch" placeholder="search&hellip;" autocomplete="off">
  </div>
  <div class="fg">
    <span class="fl">Sectors</span>
    <div class="rw">
      <span class="rv" id="sl-v">1</span>
      <input type="range" id="sl" min="1" max="12" value="1" step="1">
      &ndash;
      <input type="range" id="sh" min="1" max="12" value="12" step="1">
      <span class="rv" id="sh-v">12+</span>
    </div>
  </div>
  <div class="fg">
    <span class="fl">Tmag</span>
    <div class="rw">
      <span class="rv" id="tl-v">5</span>
      <input type="range" id="tl" min="50" max="130" value="50" step="1">
      &ndash;
      <input type="range" id="th" min="50" max="130" value="130" step="1">
      <span class="rv" id="th-v">13</span>
    </div>
  </div>
  <div class="fg">
    <span class="fl">Scatter &le;</span>
    <div class="rw">
      <input type="range" id="sm" min="0" max="20" value="20" step="0.5">
      <span class="rv" id="sm-v">any</span>
    </div>
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
    <span class="fl">Split</span>
    <div class="tg" id="sp-tg">
      <button class="tb on" data-v="-1">All</button>
      <button class="tb" data-v="0" style="color:var(--tr)">Train</button>
      <button class="tb" data-v="1" style="color:var(--vl)">Val</button>
      <button class="tb" data-v="2" style="color:var(--te)">Test</button>
    </div>
  </div>
  <div class="fg"><button class="tb" id="rst">Reset</button></div>
</div>

<div class="stats-strip">
  <span class="ss-item">Dataset: <b id="ss-ntot">&ndash;</b></span>
  <span class="ss-item">Median scatter: <b id="ss-med">&ndash;</b></span>
  <span class="ss-item">p75: <b id="ss-p75">&ndash;</b></span>
  <span class="ss-item">p90: <b id="ss-p90">&ndash;</b></span>
  <span class="ss-item" style="opacity:.6" title="Dashed lines in sparklines mark ±5% offsets; ±12% fills full bar height">Sparkline scale: &plusmn;12% = full height &middot; dashed = &plusmn;5%</span>
</div>

<div class="rbar">
  <span class="showing" id="show-txt">&ndash;</span>
  <div class="srt-wr">Sort
    <select id="sort-sel">
      <option value="sc-d">Scatter &darr;</option>
      <option value="sc-a">Scatter &uarr;</option>
      <option value="n-d">Sectors &darr;</option>
      <option value="n-a">Sectors &uarr;</option>
      <option value="tm-a">Tmag &uarr;</option>
      <option value="tm-d">Tmag &darr;</option>
      <option value="ss-d">Score &darr;</option>
      <option value="ss-a">Score &uarr;</option>
    </select>
  </div>
  <button class="dl-btn" id="dl-btn">&darr; CSV</button>
  <div class="pgn">
    <button class="pb" id="prv">&#8592;</button>
    <span class="pi" id="pg-inf">&ndash;</span>
    <button class="pb" id="nxt">&#8594;</button>
  </div>
</div>

<div class="tw">
  <table>
    <thead><tr>
      <th data-sort="id">TIC ID</th>
      <th data-sort="tm">Tmag</th>
      <th data-sort="n">Sectors</th>
      <th style="cursor:default" title="Which TESS sectors this star was observed in">Sector List</th>
      <th data-sort="cam">Cam</th>
      <th>RA &middot; Dec</th>
      <th data-sort="sc" class="on">Scatter</th>
      <th>Split</th>
      <th data-sort="ss" title="TARS mean systematic score (higher = quieter star)">Score</th>
      <th style="cursor:default">Offsets</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="no-r" class="no-r" style="display:none">No stars match these filters.</div>
</div>

<script>
// Data: [tic_id, tmag, n_sectors, cam, ra, dec, scatter_pct, split_idx, offs_str, sects_str, sys_score]
// offs_str: base-36 chars, each = (offset-0.85)*100 rounded, clamped 0-30
// sects_str: 2-char base-36 per sector number (e.g. sector 12 → '0c')
const DATA = DATA_PLACEHOLDER;
const NTOT = DATA.length;

// Pre-compute scatter percentile lookup (sorted copy for binary search)
const _scS=DATA.map(d=>d[6]).sort((a,b)=>a-b);
const _scN=_scS.length;
const _p50=_scS[Math.floor(_scN*0.50)];
const _p75=_scS[Math.floor(_scN*0.75)];
const _p90=_scS[Math.floor(_scN*0.90)];
function pctRank(v){
  let lo=0,hi=_scN;
  while(lo<hi){const m=(lo+hi)>>1;_scS[m]<v?lo=m+1:hi=m;}
  return Math.min(99,Math.round(lo/_scN*100));
}
function pctStyle(p){
  if(p>=90)return'background:rgba(192,57,43,.18);color:#c0392b';
  if(p>=75)return'background:rgba(230,126,34,.18);color:#e67e22';
  if(p>=50)return'background:rgba(181,130,0,.18);color:#b58200';
  return'background:rgba(26,127,79,.18);color:#1a7f4f';
}

const CAM_C=['','#e05c3a','#2a78d6','#1baf7a','#c47900'];
const SP_L=['Train','Val','Test'];
const SP_C=['var(--tr)','var(--vl)','var(--te)'];
const SP_BG=['var(--tr-bg)','var(--vl-bg)','var(--te-bg)'];
const CHARS='0123456789abcdefghijklmnopqrstu';   // 31 chars for offset encoding
const SECT36='0123456789abcdefghijklmnopqrstuvwxyz'; // 36 chars for sector encoding
const PER=100;

let fil=DATA.slice(), page=0;
let st={q:'',sl:1,sh:12,tl:5,th:13,sm:20,cam:0,sp:-1,sort:'sc-d',ssort:'ss-d'};
let sortAsc=false;

function scColor(p){
  const t=Math.min(Math.max(p,0)/15,1);
  return`rgb(${Math.round(27+(224-27)*t)},${Math.round(175+(92-175)*t)},${Math.round(122+(58-122)*t)})`;
}
function nBg(n){
  if(n>=10)return'background:#1a4d2e;color:#6ee7a0'; // always dark-safe green
  if(n>=7) return'background:#3d2e00;color:#fcd34d';
  if(n>=4) return'background:var(--acc-dim);color:var(--accent)';
  return'background:var(--bg3);color:var(--muted)';
}

// Decode offset string to array of floats
function decOff(s){
  const out=[];
  for(let i=0;i<s.length;i++) out.push(0.85+CHARS.indexOf(s[i])*0.01);
  return out;
}

// Decode sector string to array of ints (2-char base-36 per sector, using SECT36)
function decSects(s){
  const out=[];
  for(let i=0;i<s.length;i+=2) out.push(SECT36.indexOf(s[i])*36+SECT36.indexOf(s[i+1]));
  return out;
}

// Sector list cell
function sectorCell(s){
  if(!s) return '<td></td>';
  const sects=decSects(s);
  const chips=sects.map(n=>`<span class="sect-chip">${n}</span>`).join('');
  return`<td><div class="sect-list" title="Sectors: ${sects.join(', ')}">${chips}</div></td>`;
}

// Sparkline SVG — larger for readability
function spark(s){
  if(!s) return '';
  const offs=decOff(s);
  const W=110,H=24,cy=12,S=12/0.12; // ±12% = ±12px = full half-height
  const ref5=(0.05*S).toFixed(2);    // 5% = 5px from center
  const dw=W/offs.length,bw=Math.max(2.5,dw-1.5);
  let r=`<svg width="${W}" height="${H}" style="display:block;overflow:visible">`
    +`<line x1="0" y1="${cy}" x2="${W}" y2="${cy}" stroke="currentColor" stroke-width="0.6" opacity="0.15"/>`
    +`<line x1="0" y1="${cy-ref5}" x2="${W}" y2="${cy-ref5}" stroke="currentColor" stroke-dasharray="2 2" stroke-width="0.6" opacity="0.25"/>`
    +`<line x1="0" y1="${cy+ref5}" x2="${W}" y2="${cy+ref5}" stroke="currentColor" stroke-dasharray="2 2" stroke-width="0.6" opacity="0.25"/>`;
  offs.forEach((o,i)=>{
    const dev=Math.max(-12,Math.min(12,(o-1)*S));
    if(Math.abs(dev)<0.2)return;
    const x=(i*dw+(dw-bw)/2).toFixed(1);
    const y=(dev>0?cy-dev:cy).toFixed(1);
    const h=Math.max(0.7,Math.abs(dev)).toFixed(1);
    r+=`<rect x="${x}" y="${y}" width="${bw.toFixed(1)}" height="${h}" fill="${o>1?'#e05c3a':'#2a78d6'}" opacity=".85"/>`;
  });
  return r+'</svg>';
}

// Offset range text: min/max decoded from the offset string
function offRange(s){
  if(!s||s.length<2)return '';
  let mn=99,mx=-99;
  for(let i=0;i<s.length;i++){
    const v=0.85+CHARS.indexOf(s[i])*0.01;
    if(v<mn)mn=v;if(v>mx)mx=v;
  }
  const lo=((mn-1)*100),hi=((mx-1)*100);
  const fmt=x=>(x>0?'+':'')+x.toFixed(0)+'%';
  return `<span class="off-range" title="Offset range across sectors: ${fmt(lo)} to ${fmt(hi)}">${fmt(lo)}&thinsp;…&thinsp;${fmt(hi)}</span>`;
}

// sys_score display
function scoreCell(v){
  if(v===null||v===undefined) return`<td><span style="color:var(--muted);font-size:10px">&ndash;</span></td>`;
  const col=v>=0.99?'#1a7f4f':v>=0.97?'#2a78d6':'#8c8b84';
  return`<td><span class="score-val" style="color:${col}" title="TARS mean systematic score: ${v.toFixed(3)}">${v.toFixed(3)}</span></td>`;
}

function ticUrl(t){return`https://exofop.ipac.caltech.edu/tess/target.php?id=${t}`}

function refilter(){
  const q=st.q;
  fil=DATA.filter(d=>{
    if(q&&!String(d[0]).startsWith(q))return false;
    if(d[2]<st.sl)return false;
    if(st.sh<12&&d[2]>st.sh)return false;
    if(d[1]<st.tl||d[1]>st.th)return false;
    if(st.sm<20&&d[6]>st.sm)return false;
    if(st.cam&&d[3]!==st.cam)return false;
    if(st.sp!==-1&&d[7]!==st.sp)return false;
    return true;
  });
  resort();
}
function resort(){
  const k=st.sort;
  fil.sort((a,b)=>
    k==='sc-d'?b[6]-a[6]:k==='sc-a'?a[6]-b[6]:
    k==='n-d'?b[2]-a[2]:k==='n-a'?a[2]-b[2]:
    k==='ss-d'?b[10]-a[10]:k==='ss-a'?a[10]-b[10]:
    k==='tm-a'?a[1]-b[1]:b[1]-a[1]
  );
  // Update header active state
  document.querySelectorAll('thead th[data-sort]').forEach(th=>{
    th.classList.remove('on','asc');
    const map={id:'id',tm:'tm',n:'n',cam:'cam',sc:'sc'};
    const col=th.dataset.sort;
    if((k.startsWith(col+'-'))||(k==='sc-d'&&col==='sc')||(k==='sc-a'&&col==='sc')||
       (k==='n-d'&&col==='n')||(k==='n-a'&&col==='n')||
       (k==='tm-a'&&col==='tm')||(k==='tm-d'&&col==='tm')){
      th.classList.add('on');
      if(k.endsWith('-a'))th.classList.add('asc');
    }
  });
  page=0;render();
}
function render(){
  const tot=fil.length,tp=Math.max(1,Math.ceil(tot/PER));
  page=Math.min(page,tp-1);
  const s=page*PER,slice=fil.slice(s,s+PER);
  document.getElementById('show-txt').textContent=tot===0?'No results':
    `Showing ${(s+1).toLocaleString()}–${Math.min(s+PER,tot).toLocaleString()} of ${tot.toLocaleString()} stars`;
  document.getElementById('pg-inf').textContent=`${page+1} / ${tp}`;
  document.getElementById('prv').disabled=page===0;
  document.getElementById('nxt').disabled=page>=tp-1;
  const tb=document.getElementById('rows');
  if(!slice.length){tb.innerHTML='';document.getElementById('no-r').style.display='';return;}
  document.getElementById('no-r').style.display='none';
  tb.innerHTML=slice.map(d=>{
    const[tic,tm,n,cam,ra,dec,sc,sp,offs,sects,syssc]=d;
    const dcS=dec>=0?'+'+dec.toFixed(1):dec.toFixed(1);
    const pct=pctRank(sc);
    const medLeft=Math.min(100,_p50/15*100).toFixed(1);
    return`<tr>
      <td class="t-id"><a href="${ticUrl(tic)}" target="_blank" rel="noreferrer">${tic}</a></td>
      <td class="t-num">${tm.toFixed(1)}</td>
      <td><span class="nbadge" style="${nBg(n)}">${n}</span></td>
      ${sectorCell(sects)}
      <td><span class="chip-cam" style="background:${CAM_C[cam]}">${cam}</span></td>
      <td class="t-coord">${ra.toFixed(1)}&nbsp;&middot;&nbsp;${dcS}</td>
      <td title="Cross-sector scatter · p${pct} of all ${NTOT.toLocaleString()} stars · median=${_p50.toFixed(2)}% · p75=${_p75.toFixed(2)}% · p90=${_p90.toFixed(2)}%">
        <div class="scell">
          <div class="sbar-bg">
            <div class="sbar" style="width:${Math.min(100,sc/15*100).toFixed(1)}%;background:${scColor(sc)}"></div>
            <div class="sbar-med" style="left:${medLeft}%"></div>
          </div>
          <span class="sval">${sc.toFixed(2)}%</span>
          <span class="pct-pill" style="${pctStyle(pct)}">p${pct}</span>
        </div>
      </td>
      <td><span class="spbadge" style="background:${SP_BG[sp]};color:${SP_C[sp]}">${SP_L[sp]}</span></td>
      ${scoreCell(syssc)}
      <td class="spark-cell"><div class="spark-wrap">${spark(offs)}${offRange(offs)}</div></td>
    </tr>`;
  }).join('');
}

let sT;
document.getElementById('srch').addEventListener('input',e=>{
  clearTimeout(sT);sT=setTimeout(()=>{st.q=e.target.value.trim();refilter();},180);
});

function rng(id,vid,key,scale,fmt){
  const el=document.getElementById(id),vEl=document.getElementById(vid);
  el.addEventListener('input',()=>{
    st[key]=parseFloat(el.value)*scale;
    vEl.textContent=fmt(st[key]);
    clearTimeout(el._t);el._t=setTimeout(refilter,120);
  });
}
rng('sl','sl-v','sl',1,v=>v);
rng('sh','sh-v','sh',1,v=>v>=12?'12+':v);
rng('tl','tl-v','tl',0.1,v=>v.toFixed(1));
rng('th','th-v','th',0.1,v=>v.toFixed(1));
rng('sm','sm-v','sm',1,v=>v>=20?'any':v.toFixed(1)+'%');

function tog(gid,key){
  document.getElementById(gid).addEventListener('click',e=>{
    const b=e.target.closest('.tb');if(!b)return;
    document.querySelectorAll(`#${gid} .tb`).forEach(x=>x.classList.remove('on'));
    b.classList.add('on');st[key]=parseInt(b.dataset.v,10);refilter();
  });
}
tog('cam-tg','cam');tog('sp-tg','sp');

document.getElementById('sort-sel').addEventListener('change',e=>{st.sort=e.target.value;resort();});
document.getElementById('prv').addEventListener('click',()=>{page--;render();});
document.getElementById('nxt').addEventListener('click',()=>{page++;render();});

document.getElementById('rst').addEventListener('click',()=>{
  st={q:'',sl:1,sh:12,tl:5,th:13,sm:20,cam:0,sp:-1,sort:'sc-d'};
  ['sl','sh','tl','th','sm'].forEach(id=>document.getElementById(id).value=
    {sl:1,sh:12,tl:50,th:130,sm:20}[id]);
  document.getElementById('sl-v').textContent='1';
  document.getElementById('sh-v').textContent='12+';
  document.getElementById('tl-v').textContent='5.0';
  document.getElementById('th-v').textContent='13.0';
  document.getElementById('sm-v').textContent='any';
  document.getElementById('srch').value='';
  document.getElementById('sort-sel').value='sc-d';
  ['cam-tg','sp-tg'].forEach(id=>
    document.querySelectorAll(`#${id} .tb`).forEach((b,i)=>b.classList.toggle('on',i===0)));
  refilter();
});

document.getElementById('dl-btn').addEventListener('click',()=>{
  const hdr='tic_id,tmag,n_sectors,cam,ra,dec,scatter_pct,split';
  const rows=fil.map(d=>`${d[0]},${d[1]},${d[2]},${d[3]},${d[4]},${d[5]},${d[6]},${SP_L[d[7]].toLowerCase()}`);
  const blob=new Blob([hdr+'\\n'+rows.join('\\n')],{type:'text/csv'});
  const a=Object.assign(document.createElement('a'),{href:URL.createObjectURL(blob),download:'stitch_filtered.csv'});
  a.click();URL.revokeObjectURL(a.href);
});

// Column header sort clicks
document.querySelectorAll('thead th[data-sort]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.sort;
    const cur=st.sort;
    const base={id:'id',tm:'tm',n:'n',cam:'cam',sc:'sc',ss:'ss'}[col];
    if(!base)return;
    if(cur===base+'-d')st.sort=base+'-a';
    else st.sort=base+'-d';
    document.getElementById('sort-sel').value=st.sort;
    resort();
  });
});

// Fix sticky thead to sit exactly below the results bar (dynamic height)
(function(){
  const rbar=document.querySelector('.rbar');
  function fix(){
    const h=rbar.getBoundingClientRect().height;
    document.querySelectorAll('thead th').forEach(th=>th.style.top=h+'px');
  }
  fix();
  new ResizeObserver(fix).observe(rbar);
})();

document.getElementById('tot-badge').textContent=NTOT.toLocaleString()+' stars';
document.getElementById('ss-ntot').textContent=NTOT.toLocaleString()+' stars';
document.getElementById('ss-med').textContent=_p50.toFixed(2)+'%';
document.getElementById('ss-p75').textContent=_p75.toFixed(2)+'%';
document.getElementById('ss-p90').textContent=_p90.toFixed(2)+'%';
refilter();
</script>
"""

HTML = HTML.replace("DATA_PLACEHOLDER", data_js)
with open(OUT, "w") as f:
    f.write(HTML)
print(f"Written → {OUT}  ({len(HTML)/1024:.0f} KB)")
