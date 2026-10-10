from pathlib import Path
import csv
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
src = ROOT / "pairwise_global_robust_results" / "per_star_scatter.csv"
rows = list(csv.DictReader(src.open()))

def f(row, key):
    return float(row[key])

raw = [f(r, "raw")*100 for r in rows]
rob = [f(r, "pairwise_global_robust")*100 for r in rows]
order = sorted(range(len(rows)), key=lambda i: raw[i], reverse=True)

W, H = 1800, 1120
BG = "#0b1220"; PANEL = "#111c2e"; TEXT = "#e8eef7"; MUTED = "#9aaac0"
BLUE = "#62a8ff"; GREEN = "#43d17a"; ORANGE = "#ffb35c"; RED = "#ff6f7d"; GRID = "#2a3a52"
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
except Exception:
    font = small = title = ImageFont.load_default()

def text(x, y, s, fill=TEXT, ft=font): d.text((x, y), s, fill=fill, font=ft)
def panel(x, y, w, h):
    d.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=PANEL, outline=GRID, width=2)

text(55, 28, "STITCH before / after diagnostics", ft=title)
text(58, 70, "Per-star cross-sector scatter and absolute-error views on the robust-global test split", fill=MUTED, ft=small)

# Scatter panel
x0, y0, pw, ph = 55, 125, 810, 570
panel(x0, y0, pw, ph); text(x0+25, y0+18, "Per-star scatter: raw vs robust global", ft=font)
left, top, right, bottom = x0+80, y0+70, x0+pw-35, y0+ph-65
lim = max(2.5, min(5.0, max(max(raw), max(rob))*1.03))
for k in range(6):
    val = lim*k/5
    xx = left + (right-left)*k/5; yy = bottom - (bottom-top)*k/5
    d.line((xx, top, xx, bottom), fill=GRID, width=1); d.line((left, yy, right, yy), fill=GRID, width=1)
    text(xx-12, bottom+10, f"{val:.1f}", fill=MUTED, ft=small)
    text(left-50, yy-9, f"{val:.1f}", fill=MUTED, ft=small)
d.line((left, bottom, right, top), fill=MUTED, width=2)
for a,b in zip(raw, rob):
    xx = left + min(a,lim)/lim*(right-left); yy = bottom - min(b,lim)/lim*(bottom-top)
    d.ellipse((xx-3, yy-3, xx+3, yy+3), fill=GREEN if b<a else RED)
text((left+right)//2-85, bottom+39, "Raw scatter (%)", fill=MUTED, ft=small)
text(x0+13, (top+bottom)//2-60, "Corrected (%)", fill=MUTED, ft=small)
text(left+15, top+12, f"median: {sorted(raw)[len(raw)//2]:.3f}%  →  {sorted(rob)[len(rob)//2]:.3f}%", fill=GREEN, ft=small)
text(left+15, top+38, f"{sum(b<a for a,b in zip(raw,rob))/len(raw)*100:.1f}% of stars improve", fill=MUTED, ft=small)

# Sorted paired panel
x0, y0, pw, ph = 925, 125, 820, 570
panel(x0, y0, pw, ph); text(x0+25, y0+18, "Paired ranking: every test star", ft=font)
left, top, right, bottom = x0+70, y0+70, x0+pw-35, y0+ph-65
lim2 = min(4.0, max(max(raw), max(rob))*1.02)
for k in range(5):
    yy = bottom - (bottom-top)*k/4; val = lim2*k/4
    d.line((left, yy, right, yy), fill=GRID, width=1); text(left-48, yy-9, f"{val:.1f}", fill=MUTED, ft=small)
for j,i in enumerate(order):
    xx = left + (right-left)*j/max(1,len(order)-1)
    yr = bottom - min(raw[i],lim2)/lim2*(bottom-top); yc = bottom - min(rob[i],lim2)/lim2*(bottom-top)
    d.line((xx, yr, xx, yc), fill=GREEN if rob[i]<raw[i] else RED, width=1)
    d.ellipse((xx-2,yr-2,xx+2,yr+2), fill=BLUE); d.ellipse((xx-2,yc-2,xx+2,yc+2), fill=GREEN)
text((left+right)//2-145, bottom+39, "Stars sorted by raw scatter", fill=MUTED, ft=small)
text(left+15, top+12, "blue = raw", fill=BLUE, ft=small); text(left+120, top+12, "green = robust global", fill=GREEN, ft=small)
text(left+15, top+38, "vertical line shows the change for each star", fill=MUTED, ft=small)

# MAE bars
x0, y0, pw, ph = 55, 735, 1690, 315
panel(x0, y0, pw, ph); text(x0+25, y0+18, "MAE diagnostics (lower is better)", ft=font)
groups = [
    ("Centered log-flux MAE\n(calibration-aligned)", [("Raw",.01312096,BLUE),("Median KNN",.01040838,ORANGE),("Residual HGB",.00979047,ORANGE),("Robust global",.00896558,GREEN)]),
    ("MAE against noisy LOO label\n(diagnostic only)", [("Median KNN",.01574144,ORANGE),("Residual HGB",.01474178,GREEN),("Robust global",.01602749,RED)]),
]
for gi,(label,bars) in enumerate(groups):
    gx = x0 + 80 + gi*800; base = y0+ph-60; chart_h = 170; maxv=.018
    text(gx, y0+58, label, fill=MUTED, ft=small)
    d.line((gx, base, gx+690, base), fill=MUTED, width=2)
    for bi,(name,val,col) in enumerate(bars):
        bx = gx+35+bi*170; bh = val/maxv*chart_h
        d.rectangle((bx, base-bh, bx+90, base), fill=col)
        text(bx+45-len(f"{val:.4f}")*4, base-bh-25, f"{val:.4f}", fill=TEXT, ft=small)
        text(bx+45-len(name)*3.5, base+10, name, fill=MUTED, ft=small)
text(x0+30, 1070, "Primary ranking uses median per-star scatter. LOO-label MAE can rank methods differently because the label itself is noisy.", fill=MUTED, ft=small)

out = ROOT / "stitch_before_after.png"
im.save(out)
print(out)
