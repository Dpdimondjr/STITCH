from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pairwise_global_robust import fit_robust_global, REF_LOWER, REF_UPPER
from residual_multiscale_experiment import split_stars, spatial_features
from pairwise_global_calibration import BASE_COLS, FEATURE_COLS

ROOT = Path(__file__).resolve().parent
df = pd.read_parquet(ROOT.parent/'training_data_v3.parquet')
finite = (np.isfinite(df.sector_median) & df.sector_median.gt(0) & np.isfinite(df.flux_offset_loo) & df.n_sectors_total.ge(4) & df.tmag.le(13) & df.col.notna() & df.row.notna())
df = df[finite].copy().reset_index(drop=True)
train_ids, val_ids, test_ids = split_stars(df)
masks = {'train':df.tic_id.isin(train_ids), 'val':df.tic_id.isin(val_ids), 'test':df.tic_id.isin(test_ids)}
ref = df[masks['train']].copy(); ref['flux_offset_loo'] = ref.flux_offset_loo.clip(REF_LOWER, REF_UPPER)
frames, X, base = {}, {}, {}
for n,m in masks.items():
    fr = df[m].reset_index(drop=True); ft = spatial_features(ref, df[m]); good = np.isfinite(ft.knn20_median.to_numpy())
    frames[n] = fr.loc[good].reset_index(drop=True); ft = ft.loc[good].reset_index(drop=True)
    X[n] = pd.concat([frames[n][BASE_COLS], ft], axis=1)[FEATURE_COLS]; base[n] = ft.knn20_median.to_numpy()
model,_ = fit_robust_global(X['train'], np.log(frames['train'].sector_median.to_numpy()), np.log(base['train']), frames['train'].tic_id.to_numpy())
def scatter(fr, off):
    e = np.log(fr.sector_median.to_numpy()) - np.log(off)
    return pd.Series(e).groupby(fr.tic_id).transform(lambda z: np.sqrt(np.mean((z-z.mean())**2))).to_numpy()
rv = model.predict(X['val']); best_beta, best_score = 0., 1e9
for beta in np.linspace(-.25,1.5,36):
    s = np.median(scatter(frames['val'], np.exp(np.log(base['val'])+beta*rv)))
    if s < best_score: best_score, best_beta = s, beta
test = frames['test']; pred = np.exp(np.log(base['test']) + best_beta*model.predict(X['test']))
raw_s, new_s = scatter(test, np.ones(len(test))), scatter(test, pred)
tab = pd.DataFrame({'tic_id':test.tic_id.to_numpy(),'raw':raw_s,'new':new_s}).groupby('tic_id').first(); tab['gain'] = tab.raw-tab.new
counts = test.groupby('tic_id').size(); tab = tab[counts>=5].sort_values('gain', ascending=False); chosen = list(tab.head(3).index)

W,H = 1800,1050; im = Image.new('RGB',(W,H),'#0b1220'); d = ImageDraw.Draw(im)
try:
    ft = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',20); title = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',30); sm = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16)
except Exception: ft = title = sm = ImageFont.load_default()
def txt(x,y,s,c='#e8eef7',f=ft): d.text((x,y),s,fill=c,font=f)
txt(50,28,'Individual TIC examples: before and after STITCH',f=title); txt(52,68,'Sector medians normalized to each star median; green is corrected using robust global',c='#9aaac0',f=sm)
for j,tic in enumerate(chosen):
    x = 50+(j%2)*875; y = 120+(j//2)*400; pw,ph = 820,340
    d.rounded_rectangle((x,y,x+pw,y+ph),radius=12,fill='#111c2e',outline='#2a3a52',width=2)
    fr = test[test.tic_id==tic].copy(); inds = np.where(test.tic_id.to_numpy()==tic)[0]; off = pred[inds]
    fr['raw_norm'] = fr.sector_median/np.median(fr.sector_median); fr['corr_norm'] = (fr.sector_median.to_numpy()/off)/np.median(fr.sector_median.to_numpy()/off); fr = fr.sort_values('sector')
    xx,yy,ww,hh = x+70,y+65,pw-105,ph-115; lo = min(fr.raw_norm.min(),fr.corr_norm.min())-.03; hi = max(fr.raw_norm.max(),fr.corr_norm.max())+.03
    for k in range(5):
        gy = yy+hh*k/4; d.line((xx,gy,xx+ww,gy),fill='#2a3a52',width=1); txt(x+12,gy-8,f'{hi-(hi-lo)*k/4:.2f}',c='#9aaac0',f=sm)
    xs = np.linspace(xx,xx+ww,len(fr))
    for i in range(len(fr)-1):
        a,b = fr.iloc[i],fr.iloc[i+1]
        d.line((xs[i],yy+hh*(hi-a.raw_norm)/(hi-lo),xs[i+1],yy+hh*(hi-b.raw_norm)/(hi-lo)),fill='#62a8ff',width=3)
        d.line((xs[i],yy+hh*(hi-a.corr_norm)/(hi-lo),xs[i+1],yy+hh*(hi-b.corr_norm)/(hi-lo)),fill='#43d17a',width=3)
    for i,s in enumerate(fr.sector): txt(xs[i]-10,yy+hh+10,str(int(s)),c='#9aaac0',f=sm)
    q = tab.loc[tic]; txt(x+20,y+18,f'TIC {int(float(tic))}   scatter {q.raw*100:.2f}% → {q.new*100:.2f}% ({(q.gain/q.raw)*100:.0f}% lower)',c='#43d17a')
    txt(xx,y+ph-28,'blue raw',c='#62a8ff',f=sm); txt(xx+100,y+ph-28,'green corrected',c='#43d17a',f=sm)
out = ROOT/'stitch_tic_examples.png'; im.save(out); print(out); print('chosen',chosen)
for tic in chosen:
    ii = np.where(test.tic_id.to_numpy()==tic)[0]
    print('DETAIL', int(float(tic)), test.iloc[ii][['sector','sector_median','flux_offset_loo','row','col','cam','ccd','crowdsap']].assign(predicted_offset=pred[ii]).to_string(index=False))
