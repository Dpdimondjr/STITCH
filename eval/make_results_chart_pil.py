from PIL import Image, ImageDraw, ImageFont

W, H = 1800, 720
im = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(im)
font = ImageFont.load_default()
blue, green = (68, 114, 196), (46, 139, 87)
d.text((55, 25), 'STITCH experiment results — lower scatter is better', fill=(20,20,20), font=font)
d.text((55, 48), 'Green marks the best result within each evaluation panel.', fill=(90,90,90), font=font)
panels = [
 ('Main diagnostic split', ['Raw','Median-KNN','Residual HGB','Robust global'], [1.504,1.217,1.147,.980], 1.75, '37,372 training stars; robust untruncated rerun', 3),
 ('Future-sector simulation', ['Raw','Median-KNN','Residual HGB','Robust global'], [1.478,1.036,.970,.820], 1.75, 'Train Sectors 1–59; test Sectors 68–85; 955 science stars', 3),
 ('Feature / model checks', ['Spatial surface','FLFRCSAP*','Latent modes','Pairwise NSF'], [1.109,.877,.982,1.045], 1.25, '*Complete-case FLFRCSAP is not directly comparable', 0),
]
pw, gap, left, top, chartw = 555, 32, 55, 115, 435
for pi, (title, labels, vals, xmax, note, best) in enumerate(panels):
    x0 = left + pi*(pw+gap)
    d.text((x0, top-35), title, fill=(20,20,20), font=font)
    for j, (lab, val) in enumerate(zip(labels, vals)):
        y = top + j*90
        d.text((x0, y+8), lab, fill=(25,25,25), font=font)
        bw = int(chartw*val/xmax)
        d.rectangle((x0+145, y, x0+145+bw, y+32), fill=green if j==best else blue)
        d.text((x0+155+bw, y+8), f'{val:.3f}%', fill=(20,20,20), font=font)
    for tick in [0,.5,1,1.5,2]:
        if tick <= xmax:
            xx = x0+145+int(chartw*tick/xmax)
            d.line((xx, top-8, xx, top+350), fill=(220,220,220), width=1)
            d.text((xx-10, top+365), f'{tick:.1f}%', fill=(90,90,90), font=font)
    d.text((x0, top+405), note, fill=(90,90,90), font=font)
d.text((55, 675), 'Best overall: robust global calibration — 0.980% main split and 0.820% future-sector simulation.', fill=(20,20,20), font=font)
im.save('eval/stitch_results_chart.png')
print('saved eval/stitch_results_chart.png')
