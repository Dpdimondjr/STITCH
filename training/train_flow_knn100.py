"""
STITCH normalizing flow — KNN-100 base decomposition.

Key design principle: use the same log-residual decomposition as the HGB.
  Target: y = log(flux_offset_loo) - log(knn_100)
  Prediction: knn_100 * exp(flow_median(context))

Why this beats the earlier flow:
  - Prior flow used KNN-20 as a scalar context feature, leaving dominant spatial
    variance in the target, making the flow's task harder.
  - HGB's 40% improvement comes from KNN-100 as a multiplicative base that
    removes the dominant spatial component BEFORE regression.
  - This flow mirrors that decomposition: the flow only needs to model the
    residual from KNN-100, which is smaller and more normally distributed.
  - The flow adds UNCERTAINTY quantification that HGB cannot provide.

Context: all kitchen-sink features (BASE_COLS + TIC_COLS + THERMAL + PDC + SEC_AGG)
Splits: uses split.csv for direct comparability with HGB ablations.
Evaluation: reports both NLL and the STITCH CV metric (for direct comparison).
"""

import sys as _flush_sys
_flush_sys.stdout.reconfigure(line_buffering=True)

import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import zuko
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import cKDTree
import sys as _sys

t0 = time.time()

_parquet  = next((s for s in _sys.argv[1:] if s.endswith(".parquet")),
                 "/Users/daviddimond/Documents/STITCH/training_data_v3.parquet")
_out_pt   = next((s for s in _sys.argv[1:] if s.endswith(".pt")),
                 "/private/tmp/claude-501/-Users-daviddimond-Documents-STITCH/5c4c4482-2695-4c29-a4c0-9c4e279cad18/scratchpad/stitch_flow_knn100.pt")
_alpha    = float(next((s.split("=")[1] for s in _sys.argv[1:] if s.startswith("--alpha=")), "0.3"))
_n_mc     = int(next((s.split("=")[1] for s in _sys.argv[1:] if s.startswith("--n-mc=")), "8"))
print(f"Parquet: {_parquet}  →  {_out_pt}  (alpha={_alpha}, n_mc={_n_mc})")

# ── Constants ────────────────────────────────────────────────────────────────
SEED=20260915; MIN_SECTORS=5; MAX_TMAG=13; REF_CLIP=(0.5,1.5)
KNN_K = 100
BASE_COLS=['col','row','delta_sub_col','delta_sub_row','sector','cam','ccd',
    'tmag','crowdsap','cdpp1_0','pdcvar','jitter_rms','pdc_noi','pr_wght2','gaiarp']
KEYS=['sector','cam','ccd']
TIC_CACHE='/private/tmp/claude-501/-Users-daviddimond-Documents-STITCH/5c4c4482-2695-4c29-a4c0-9c4e279cad18/scratchpad/tic_features_cache.parquet'

# ── 1. Load & filter data ─────────────────────────────────────────────────────
print("Loading data...", flush=True)
df = pd.read_parquet(_parquet, engine='pyarrow')
df['tic_id'] = df['tic_id'].astype('int64'); df['sector'] = df['sector'].astype('int64')

valid = ((df.sector.between(1,100)) & (df.cam.between(1,4)) & (df.ccd.between(1,4)) &
         (df.tmag <= MAX_TMAG) & (df.gaiarp.notna()) & (df.sector_median > 0) &
         (df.col.between(0,2200)) & (df.row.between(0,2200)))
df = df[valid]
sc = df.groupby('tic_id')['sector'].nunique()
df = df[df.tic_id.isin(sc[sc >= MIN_SECTORS].index)].copy()

# ── 2. TIC catalog features ───────────────────────────────────────────────────
tc = pd.read_parquet(TIC_CACHE)
df = df.merge(tc[['tic_id','contratio','numcont','gaiabp']], on='tic_id', how='left', suffixes=('','_tic'))
for col in ['contratio','numcont','gaiabp']:
    df[col] = df[col].astype('float64')
df['bp_rp']         = df['gaiabp'] - df['gaiarp']
df['log_contratio'] = np.log1p(df['contratio'])
df['numcont_x_crowdsap']  = df['numcont']  * df['crowdsap']
df['numcont_x_row']       = df['numcont']  * (df['row'] / 2200.0)
df['bp_rp_x_crowdsap']    = df['bp_rp']    * df['crowdsap']
df['contratio_x_row']     = df['log_contratio'] * (df['row'] / 2200.0)
df['numcont_x_cam']       = df['numcont']  * df['cam']

# Thermal features
df['focal_plane_temp']     = df['focal_plane_temp'].astype('float64')
df['focal_plane_temp_std'] = df['focal_plane_temp_std'].astype('float64')
df['temp_x_col']      = df['focal_plane_temp'] * (df['col'] / 2200.0)
df['temp_x_row']      = df['focal_plane_temp'] * (df['row'] / 2200.0)
df['temp_x_crowdsap'] = df['focal_plane_temp'] * df['crowdsap']

# PDC features
for c in ['pdc_corp','pdc_totp','teff','bkg_rms','scatter_flag_frac']:
    if c in df.columns: df[c] = df[c].astype('float64')
    else: df[c] = np.nan
df['pdctotp_x_crowdsap']  = df['pdc_totp'] * df['crowdsap']
df['pdctotp_x_contratio'] = df['pdc_totp'] * df['log_contratio']

df['n_sectors_obs'] = df.groupby('tic_id')['sector'].transform('nunique').astype(float)
df = df.reset_index(drop=True)
print(f"  {len(df):,} obs, {df.tic_id.nunique():,} TICs  T={time.time()-t0:.0f}s")

# ── 3. Load split ─────────────────────────────────────────────────────────────
split = pd.read_csv('/Users/daviddimond/Documents/STITCH/eval/value_benchmark_results/split.csv')
def get_p(n): return set(split[split.partition==n].tic_id.astype('int64').values)
test_tics = get_p('test'); val_tics = get_p('validation')

ref_full  = df[~df.tic_id.isin(test_tics) & df.flux_offset_loo.between(*REF_CLIP)].copy()
train_all = df[~df.tic_id.isin(test_tics) & ~df.tic_id.isin(val_tics)].copy()
train_all = train_all[train_all.flux_offset_loo.between(*REF_CLIP)].reset_index(drop=True)
val_r     = df[df.tic_id.isin(val_tics)].reset_index(drop=True)
test_r    = df[df.tic_id.isin(test_tics)].reset_index(drop=True)
print(f"  Splits: train={len(train_all)}, val={len(val_r)}, test={len(test_r)}  T={time.time()-t0:.0f}s")
print(f"  Val extreme (outside REF_CLIP): {(~val_r.flux_offset_loo.between(*REF_CLIP)).sum()} rows (excluded from NLL scoring)")

# ── 4. Sector aggregates (from reference, LOO-safe) ───────────────────────────
sec_stats = ref_full.groupby(['sector','cam','ccd'])['flux_offset_loo'].agg(
    sector_mean_loo='mean',
    sector_std_loo='std',
    sector_frac_extreme=lambda x: (np.abs(x-1) > 0.02).mean()
).reset_index()
for df_ in [train_all, val_r, test_r]:
    df_.drop(columns=['sector_mean_loo','sector_std_loo','sector_frac_extreme'], errors='ignore', inplace=True)
    tmp = df_.merge(sec_stats, on=['sector','cam','ccd'], how='left')
    df_['sector_mean_loo']       = tmp['sector_mean_loo'].values
    df_['sector_std_loo']        = tmp['sector_std_loo'].values
    df_['sector_frac_extreme']   = tmp['sector_frac_extreme'].values
    df_['secmean_x_contratio']   = df_['sector_mean_loo'] * df_['log_contratio']

# ── 5. KNN-100 base (LOO-aware) ───────────────────────────────────────────────
def knn_median_base(ref, query, k=100):
    """LOO-aware KNN-100 median. Returns array of knn corrections."""
    result = np.ones(len(query))
    query  = query.reset_index(drop=True)
    for key, qg in query.groupby(['sector','cam','ccd']):
        r = ref[(ref.sector==key[0]) & (ref.cam==key[1]) & (ref.ccd==key[2])]
        if len(r) < 2: continue
        coords  = r[['col','row']].to_numpy(float)
        qcoords = qg[['col','row']].to_numpy(float)
        qids    = qg.tic_id.to_numpy(); rids = r.tic_id.to_numpy()
        take    = min(k + int(r.tic_id.value_counts().max()), len(r))
        dists, idxs = cKDTree(coords).query(qcoords, k=take)
        if take == 1: dists, idxs = dists[:,None], idxs[:,None]
        loo = r.flux_offset_loo.to_numpy()
        for i, (q_id, nn) in enumerate(zip(qids, idxs)):
            keep = rids[nn] != q_id; vals = loo[nn[keep][:k]]
            if len(vals) >= 1: result[qg.index[i]] = np.median(vals)
    return result

print(f"Computing KNN-{KNN_K} base for train...", flush=True)
knn_tr = knn_median_base(ref_full, train_all, k=KNN_K)
print(f"Computing KNN-{KNN_K} base for val...", flush=True)
knn_vl = knn_median_base(ref_full, val_r,     k=KNN_K)
print(f"Computing KNN-{KNN_K} base for test...", flush=True)
knn_te = knn_median_base(ref_full, test_r,    k=KNN_K)

# Log-residual target: what the flow needs to predict
train_all['y_resid'] = np.log(train_all.flux_offset_loo.values) - np.log(knn_tr)
val_r['y_resid']     = np.log(val_r.flux_offset_loo.values)     - np.log(knn_vl)
test_r['y_resid']    = np.log(test_r.flux_offset_loo.values)    - np.log(knn_te)

r_knn_tr = np.corrcoef(np.log(train_all.flux_offset_loo.values), np.log(knn_tr))[0,1]
print(f"KNN-100 correlation with log(loo): r={r_knn_tr:.3f}  (residual std={train_all.y_resid.std():.5f})")
print(f"  T={time.time()-t0:.0f}s")

# ── 6. Feature matrix ─────────────────────────────────────────────────────────
TIC_COLS = ['log_contratio','bp_rp','numcont']
TIC_IX   = ['numcont_x_crowdsap','numcont_x_row','bp_rp_x_crowdsap','contratio_x_row','numcont_x_cam']
THERMAL  = ['focal_plane_temp','focal_plane_temp_std','temp_x_col','temp_x_row','temp_x_crowdsap']
PDC_F    = ['pdc_corp','pdc_totp','pdctotp_x_crowdsap','pdctotp_x_contratio']
SEC_AGG  = ['sector_mean_loo','sector_std_loo','sector_frac_extreme','secmean_x_contratio']
MISC     = ['teff','bkg_rms','scatter_flag_frac','n_sectors_obs']

ALL_COLS = BASE_COLS + TIC_COLS + TIC_IX + THERMAL + PDC_F + SEC_AGG + MISC
# Include log(knn_100) as a context feature: tells the flow the spatial level
ALL_COLS_WITH_KNN = ALL_COLS  # knn added separately below

cam_dummies = pd.get_dummies(df['cam'].astype(int), prefix='cam')
ccd_dummies = pd.get_dummies(df['ccd'].astype(int), prefix='ccd')

# Filter to existing cols
feat_cols = [c for c in ALL_COLS if c in train_all.columns]
print(f"Features: {len(feat_cols)} + cam/ccd OHE + log_knn100 = {len(feat_cols)+8+1}D context")

# Compute means/stds from training data (fill NaN with median for normalization)
tr_means = train_all[feat_cols].mean()
tr_stds  = train_all[feat_cols].std().replace(0, 1)

def make_context(df_, knn_arr):
    cont = ((df_[feat_cols].fillna(tr_means) - tr_means) / tr_stds).reset_index(drop=True).values
    cam_oh = pd.get_dummies(df_['cam'].astype(int), prefix='cam').reindex(
                 columns=cam_dummies.columns, fill_value=0).reset_index(drop=True).values
    ccd_oh = pd.get_dummies(df_['ccd'].astype(int), prefix='ccd').reindex(
                 columns=ccd_dummies.columns, fill_value=0).reset_index(drop=True).values
    log_knn = ((np.log(knn_arr) - float(np.log(knn_tr).mean())) /
               max(float(np.log(knn_tr).std()), 1e-8))[:,None]
    return np.hstack([cont, cam_oh, ccd_oh, log_knn]).astype(np.float32)

C_train = make_context(train_all, knn_tr)
C_val   = make_context(val_r,     knn_vl)
C_test  = make_context(test_r,    knn_te)
context_dim = C_train.shape[1]

# Standardize the target residual
y_mean = float(train_all['y_resid'].mean())
y_std  = float(train_all['y_resid'].std())
y_train = ((train_all['y_resid'].values - y_mean) / y_std).astype(np.float32)
y_val   = ((val_r['y_resid'].values     - y_mean) / y_std).astype(np.float32)
y_test  = ((test_r['y_resid'].values    - y_mean) / y_std).astype(np.float32)

# Sample weights: sharper reward for well-observed stars
_n_obs  = train_all['n_sectors_obs'].clip(lower=2)
_raw_w  = (_n_obs - 1).astype(np.float32)
w_train = (_raw_w / _raw_w.max()).astype(np.float32)

print(f"\nContext dimension: {context_dim}")
print(f"Target y_resid_mean={y_mean:.5f}  y_resid_std={y_std:.5f}")

# ── 7. NSF model ──────────────────────────────────────────────────────────────
TRANSFORMS = 4
HIDDEN     = [128, 128]
BINS       = 8

device = (torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))
print(f"Device: {device}")

flow = zuko.flows.NSF(
    features=1, context=context_dim,
    transforms=TRANSFORMS, hidden_features=HIDDEN, bins=BINS,
).to(device)
n_params = sum(p.numel() for p in flow.parameters())
print(f"NSF parameters: {n_params:,}  ({TRANSFORMS} transforms, {BINS} bins, hidden={HIDDEN})")

# ── 8. Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 512; LR = 1e-4; MAX_EPOCHS = 300; PATIENCE = 30

opt   = torch.optim.Adam(flow.parameters(), lr=LR, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

train_loader = DataLoader(
    TensorDataset(torch.tensor(C_train),
                  torch.tensor(y_train).unsqueeze(-1),
                  torch.tensor(w_train)),
    batch_size=BATCH_SIZE, shuffle=True,
)
# Clipped val subset for NLL scoring: exclude extreme LOO rows (outside REF_CLIP)
# so early stopping signal is not dominated by ~200 outlier rows each contributing NLL~130
_val_clip = val_r.flux_offset_loo.between(*REF_CLIP).values
val_r_nll    = val_r[_val_clip].reset_index(drop=True)
knn_vl_nll   = knn_vl[_val_clip]
C_val_nll    = make_context(val_r_nll, knn_vl_nll)
y_val_nll    = ((val_r_nll['y_resid'].values - y_mean) / y_std).astype(np.float32)

C_val_t      = torch.tensor(C_val).to(device)
y_val_t      = torch.tensor(y_val).unsqueeze(-1).to(device)
C_val_nll_t  = torch.tensor(C_val_nll).to(device)
y_val_nll_t  = torch.tensor(y_val_nll).unsqueeze(-1).to(device)
C_test_t     = torch.tensor(C_test).to(device)

best_val_nll   = float('inf')
best_state     = None
patience_count = 0

_obj = f"NLL + {_alpha}*MAE(mc={_n_mc})" if _alpha > 0 else "NLL only"
print(f"\nTraining NSF  [{_obj}]...\n  epoch  train_nll  train_mae   val_nll    lr")
for epoch in range(1, MAX_EPOCHS + 1):
    flow.train()
    train_nlls, train_maes = [], []
    for cb, yb, wb in train_loader:
        cb, yb, wb = cb.to(device), yb.to(device), wb.to(device)
        dist = flow(cb)
        log_probs = dist.log_prob(yb)
        nll = -(log_probs * wb).sum() / wb.sum()

        if _alpha > 0 and _n_mc > 0:
            samples_r = dist.rsample((_n_mc,)).squeeze(-1)
            cond_mean = samples_r.mean(0)
            mae_loss  = ((cond_mean - yb.squeeze(-1)).abs() * wb).sum() / wb.sum()
            loss = nll + _alpha * mae_loss
            train_maes.append(mae_loss.item())
        else:
            loss = nll

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        train_nlls.append(nll.item())

    flow.eval()
    with torch.no_grad():
        val_nll = -flow(C_val_nll_t).log_prob(y_val_nll_t).mean().item()

    sched.step(val_nll)
    train_nll = np.mean(train_nlls)
    train_mae = np.mean(train_maes) if train_maes else float('nan')
    if epoch % 10 == 0:
        lr_now = opt.param_groups[0]['lr']
        print(f"  {epoch:5d}  {train_nll:.4f}     {train_mae:.4f}   {val_nll:.4f}    {lr_now:.2e}")

    if val_nll < best_val_nll:
        best_val_nll = val_nll
        best_state   = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

flow.load_state_dict(best_state)
print(f"\nBest val NLL: {best_val_nll:.4f}")

# ── 9. Evaluation ─────────────────────────────────────────────────────────────
flow.eval()
with torch.no_grad():
    samples  = flow(C_test_t).sample((500,)).squeeze(-1)   # (500, N_test)
    mu_resid = samples.median(0).values.cpu().numpy()      # standardized residual median

# Convert back: flow predicts log(loo/knn_100), standardized
log_resid_pred = mu_resid * y_std + y_mean                  # log-scale residual
pred_loo       = knn_te * np.exp(log_resid_pred)            # predicted flux_offset_loo
true_loo       = test_r['flux_offset_loo'].values

residuals    = true_loo - pred_loo
baseline_mae = np.abs(true_loo - true_loo.mean()).mean()
model_mae    = np.abs(residuals).mean()

print(f"\n=== Test Set Evaluation ===")
print(f"  Mean absolute error:  {model_mae:.4f}")
print(f"  Residual std:         {residuals.std():.4f}")
print(f"  Residual mean:        {residuals.mean():.4f}")
print(f"  Baseline (global mean): MAE = {baseline_mae:.4f}")
print(f"  Improvement over baseline:  {(1 - model_mae/baseline_mae)*100:.1f}%")

# STITCH CV metric: std(sector_median/prediction per star), comparable to HGB
def cv_per_star(part, pred_arr):
    rows = []
    for tic, g in part.groupby('tic_id'):
        if g.sector.nunique() < MIN_SECTORS: continue
        c = g.sector_median.values / pred_arr[g.index]
        rows.append({'tic_id': tic, 'cv': c.std() / c.mean()})
    return pd.DataFrame(rows).set_index('tic_id').cv * 100

cv_te  = cv_per_star(test_r, pred_loo).median()
cv_raw = cv_per_star(test_r, np.ones(len(test_r))).median()
cv_knn = cv_per_star(test_r, knn_te).median()
print(f"\n  STITCH CV metric (test, median per-star):")
print(f"    Raw (no correction):  {cv_raw:.4f}%")
print(f"    KNN-100 only:         {cv_knn:.4f}%")
print(f"    Flow (knn100 + flow): {cv_te:.4f}%")
print(f"    HGB reference:        0.9040%")

print(f"\n  Per-camera breakdown:")
print(f"  {'Cam':<6} {'n':>5} {'MAE':>8} {'CV%':>8} {'improvement':>12}")
for cam, cg in test_r.reset_index(drop=True).groupby('cam'):
    pos  = cg.index.tolist()
    y_c  = true_loo[pos]; p_c = pred_loo[pos]
    mae  = np.abs(y_c - p_c).mean()
    base = np.abs(y_c - y_c.mean()).mean()
    cv_c = cv_per_star(cg.reset_index(drop=True), p_c).median()
    print(f"  Cam{int(cam):<3} {len(y_c):>5}  {mae:.4f}  {cv_c:.4f}%  {(1-mae/base)*100:>10.1f}%")

# ── 10. Save ───────────────────────────────────────────────────────────────────
import time as _time
torch.save({
    'model_state':   best_state,
    'tr_means':      tr_means,
    'tr_stds':       tr_stds,
    'y_mean':        y_mean,
    'y_std':         y_std,
    'feat_cols':     feat_cols,
    'cam_cols':      list(cam_dummies.columns),
    'ccd_cols':      list(ccd_dummies.columns),
    'knn_log_mean':  float(np.log(knn_tr).mean()),
    'knn_log_std':   float(max(np.log(knn_tr).std(), 1e-8)),
    'flow_config': {
        'features': 1, 'context': context_dim,
        'transforms': TRANSFORMS, 'hidden_features': HIDDEN, 'bins': BINS,
    },
    'provenance': {
        'parquet': _parquet, 'knn_k': KNN_K,
        'target': 'log(flux_offset_loo/knn_100)',
        'alpha': _alpha, 'n_mc': _n_mc,
        'val_nll': best_val_nll, 'test_cv': float(cv_te),
        'date': _time.strftime('%Y-%m-%d'),
    },
}, _out_pt)
print(f"\nSaved → {_out_pt}")
print(f"Done  T={time.time()-t0:.0f}s")
