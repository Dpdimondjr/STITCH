"""Disjoint quiet-star calibration banks with retained Gaia RP anchors.

Every label-derived reference and color transfer is rebuilt within its bank.
The old validation population is split by TIC into discovery and confirmation;
neither subset contributes training labels. Test rows in the source parquet are
excluded before any reference construction, fitting, or outcome evaluation.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/stitch_mpl')
import argparse
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

from hgb_reference_repair import reference_features, SECTOR, SPATIAL, STRIPE, KEYS
from hgb_reference_repair import group_median, per_star, sha256, write_json, log
from gaia_anchor_experiment import color_design

OUT = Path('eval/calibration_bank_study_20261011')
SOURCE = Path('eval/hgb_reference_repair_20261010/clean_features.parquet')
SCORES = Path('eval/gaia_selection_20261011/tars_scores.parquet')
PARAMS = dict(loss='absolute_error', learning_rate=.06, max_iter=200,
              max_leaf_nodes=255, min_samples_leaf=40, l2_regularization=.1,
              early_stopping=False)
ROUNDS = 3
BASE = json.loads(Path('eval/hgb_reference_repair_20261010/prepared.json').read_text())['features_deviation']
PROFILE = [f'{axis}_sector_{width}_{kind}' for axis in ['col', 'row']
           for width in [32, 128, 512] for kind in ['response', 'count']]
META = ['detector', 'log_background_rp', 'log_background_rms_rp',
        'log_background_p90_rp', 'capture_log', 'crowding_log', 'log_perr_area',
        'log_background_rp_dev', 'capture_log_dev', 'crowding_log_dev',
        'background_x_capture', 'background_x_crowding', 'color_x_capture',
        'color_x_crowding', 'col_fraction', 'row_fraction']


def add_metadata(f):
    f = f.copy()
    f['pdctotp_x_crowdsap'] = f.pdc_totp * f.crowdsap
    f['detector'] = (f.cam - 1) * 4 + f.ccd
    rpflux = np.power(10., -.4 * (f.gaiarp.fillna(f.tmag) - 10))
    for source, dest in [('median_sap_bkg', 'log_background_rp'),
                         ('bkg_rms', 'log_background_rms_rp'),
                         ('p90_sap_bkg', 'log_background_p90_rp')]:
        f[dest] = np.log1p(f[source].clip(lower=0) / rpflux)
    f['capture_log'] = np.log(f.flfrcsap.clip(.05, 1.5))
    f['crowding_log'] = np.log(f.crowdsap.clip(.05, 1.5))
    f['log_perr_area'] = np.log1p((f.perr_col * f.perr_row).clip(lower=0))
    for col in ['log_background_rp', 'capture_log', 'crowding_log']:
        f[col + '_dev'] = f[col] - f.groupby('tic_id')[col].transform('median')
    f['background_x_capture'] = f.log_background_rp * f.capture_log
    f['background_x_crowding'] = f.log_background_rp * f.crowding_log
    f['color_x_capture'] = f.bp_rp * f.capture_log
    f['color_x_crowding'] = f.bp_rp * f.crowding_log
    f['col_fraction'] = f.col % 1
    f['row_fraction'] = f.row % 1
    return f


def prepare_manifest():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'freeze.json').exists():
        return
    f = pd.read_parquet(SOURCE)
    scores = pd.read_parquet(SCORES)
    f = f.merge(scores, on=['tic_id', 'sector'], how='left', validate='one_to_one')
    train = f[(f.role == 'train') & (f.systematic_score > .95)].copy()
    train = train[train.groupby('tic_id').sector.transform('nunique') >= 2]
    stars = train.groupby('tic_id').agg(cam=('cam', lambda s: s.mode().iloc[0]))
    records = []
    for rep in [0, 1]:
        rng = np.random.default_rng(202610110 + rep)
        for cam, g in stars.groupby('cam'):
            ids = rng.permutation(g.index.to_numpy())
            for i, tic in enumerate(ids):
                records.append(dict(tic_id=int(tic), dominant_cam=int(cam), replicate=rep,
                                    bank='A' if i % 2 == 0 else 'B',
                                    rank=i // 2, bank_cam_size=(len(ids) + (i % 2 == 0)) // 2))
    manifest = pd.DataFrame(records)
    manifest['fraction_rank'] = (manifest['rank'] + .5) / manifest.bank_cam_size
    manifest.to_parquet(OUT / 'training_banks.parquet', index=False)
    val = f[f.role == 'validation'].groupby('tic_id').agg(cam=('cam', lambda s: s.mode().iloc[0]))
    rng = np.random.default_rng(202610119)
    val['evaluation_group'] = ''
    for _, g in val.groupby('cam'):
        ids = rng.permutation(g.index.to_numpy())
        val.loc[ids, 'evaluation_group'] = np.where(np.arange(len(ids)) % 2, 'confirmation', 'discovery')
    val.reset_index().to_parquet(OUT / 'evaluation_manifest.parquet', index=False)
    freeze = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        source_sha256=sha256(SOURCE), scores_sha256=sha256(SCORES),
        script_sha256=sha256(__file__), training_banks_sha256=sha256(OUT/'training_banks.parquet'),
        evaluation_manifest_sha256=sha256(OUT/'evaluation_manifest.parquet'),
        params=PARAMS, rounds=ROUNDS, reference_folds=5, bank_fractions=[.125, .25, .5],
        replicates=[0, 1], fixed_prediction_strength=1.,
        primary_feature_family='base', candidates=['profiles', 'metadata', 'combined'],
        eligible_training_stars=int(train.tic_id.nunique()), eligible_training_rows=len(train),
        validation_stars=len(val), validation_rows=int((f.role=='validation').sum()),
        missing_anchor_policy='Explicit relative fallback for missing RP or color, retained in metrics',
        independence='Whole-TIC disjoint banks; every reference and color transfer bank-local; identical model seed',
        selection='Primary learning curves fixed; select follow-up variants only on discovery TICs',
        limitations=['Validation stars are historically examined development stars, not pristine.',
                     'Three rounds and moderate tree capacity are a controlled research family, not the full benchmark.',
                     'Bank agreement cannot expose shared misspecification; no sqrt(2) floor conversion.',
                     'Historical outcome eligibility cuts remain inherited.'])
    write_json(OUT / 'freeze.json', freeze)
    (OUT / 'PROTOCOL.md').write_text('''# Disjoint calibration-bank experiment

Assume the individually TARS-qualified calibration observations come from mostly
constant sources. Split whole training TICs into two disjoint banks; use nested
1/8, 1/4, and 1/2 population sizes in each of two deterministic partitions.
Rebuild all spatial and stripe references from the current bank, excluding each
query fold. Reference LOO ratios use only qualified observations within each
training TIC. Refit the Gaia RP color transfer on other bank folds. Never center
the anchored learning target. Missing RP/color retains an explicit relative
fallback, with attrition and anchor coverage reported.

The primary learning curve uses a fixed three-round HGB family and strength one.
Compare raw bank disagreements, the ensemble of disjoint banks, and held-out CV;
do not convert agreement to an irreducible floor. Select new detector/illumination
features on the frozen discovery half of validation only, then open confirmation.
Both halves have historical baseline evaluations and are development samples.
No original test outcomes enter this study. Report all retained sectors and
individually qualified sectors separately, using project CV with ddof=0.

New response information: sector/camera/CCD column and row profiles at 32, 128,
512 pixel bandwidths, with reference counts; source brightness scaled background,
capture, crowding, color interactions and deviations from a star's metadata
baseline. Profile response shrinks toward its sector/detector median with ten
pseudo-observations. These are transferable training-star measurements; no query
flux is used to construct references. Exact aperture strap overlap is not
available for the full training population and is not assumed from centroids.

Also test known-response recovery on the actual geometry and conditional cadence
noise, reference-poor/withheld geometry, and preservation of injected signals.
Any attainable range is conditional on its stated response family; residuals
and learning-curve asymptotes are not physical noise floors.
''')
    log(f'Manifests frozen: {len(stars):,} training TICs; {len(val):,} validation TICs')


def profile_features(ref, query):
    """Anisotropic sector-local information using only other-bank-fold labels."""
    assert not set(ref.tic_id) & set(query.tic_id)
    result = pd.DataFrame(0., index=np.arange(len(query)), columns=PROFILE)
    q = query.reset_index(drop=True)
    groups = dict(tuple(ref.groupby(KEYS)))
    for key, g in q.groupby(KEYS):
        r = groups.get(key)
        if r is None:
            continue
        label = np.log(r.flux_offset_loo.to_numpy()).clip(-.1, .1)
        center = np.median(label)
        for axis in ['col', 'row']:
            order = np.argsort(r[axis].to_numpy())
            pos = r[axis].to_numpy()[order]
            cumulative = np.r_[0., np.cumsum(label[order] - center)]
            for width in [32, 128, 512]:
                lo = np.searchsorted(pos, g[axis].to_numpy() - width / 2, side='left')
                hi = np.searchsorted(pos, g[axis].to_numpy() + width / 2, side='right')
                n = hi - lo
                value = (cumulative[hi] - cumulative[lo]) / (n + 10.)
                result.loc[g.index, f'{axis}_sector_{width}_response'] = value
                result.loc[g.index, f'{axis}_sector_{width}_count'] = n
    return result


def prepare_bank(name, replicate=0, fraction=.5, bank='A', exclude=None):
    run = OUT / name
    run.mkdir(exist_ok=True)
    if (run / 'prepared.json').exists():
        return
    f = pd.read_parquet(SOURCE)
    f = f[f.role != 'test'].copy()
    f = f.merge(pd.read_parquet(SCORES), on=['tic_id', 'sector'], how='left', validate='one_to_one')
    m = pd.read_parquet(OUT/'training_banks.parquet')
    if bank == 'full':
        ids = m[m.replicate == 0].tic_id.to_numpy()
    else:
        ids = m[(m.replicate == replicate) & (m.bank == bank) & (m.fraction_rank < fraction * 2)].tic_id.to_numpy()
    training = (f.role == 'train') & f.tic_id.isin(ids) & (f.systematic_score > .95)
    if exclude == 'late_sectors':
        training &= f.sector < 70
    elif exclude == 'central_column':
        training &= ~f.col.between(900, 1150)
    f = f[training | (f.role == 'validation')].copy().reset_index(drop=True)
    # Rows removed by geometry exclusion can leave singleton calibration sources.
    count = f[f.role == 'train'].groupby('tic_id').sector.nunique()
    f = f[(f.role != 'train') | f.tic_id.map(count).ge(2)].copy().reset_index(drop=True)
    f = add_metadata(f)
    f['bank_fold'] = f.reference_fold % 5
    tr = f.role == 'train'
    z = f[tr].groupby('tic_id').sector_median
    other = (z.transform('sum') - f.loc[tr, 'sector_median']) / (z.transform('size') - 1)
    f.loc[tr, 'flux_offset_loo'] = f.loc[tr, 'sector_median'] / other
    cols = SECTOR + SPATIAL + STRIPE + ['pred_knn'] + PROFILE
    features = pd.DataFrame(np.nan, index=f.index, columns=cols)
    audit = []
    for fold in range(5):
        ref = f[tr & (f.bank_fold != fold)]
        query = f[f.bank_fold == fold]
        block = reference_features(ref, query)
        profiles = profile_features(ref, query)
        features.loc[query.index, block.columns] = block.to_numpy()
        features.loc[query.index, profiles.columns] = profiles.to_numpy()
        audit.append(dict(fold=fold, reference_stars=int(ref.tic_id.nunique()), reference_rows=len(ref),
                          query_rows=len(query), overlap=0, nontraining_references=0))
        log(f'{name}: reference fold {fold+1}/5')
    f = f.drop(columns=cols, errors='ignore').join(features)
    stars = f.groupby('tic_id').agg(role=('role', 'first'), fold=('bank_fold', 'first'),
        rp=('gaiarp', 'first'), color=('bp_rp', 'first'), crowd=('crowdsap', 'median'))
    lr = np.log(f.sector_median / f.pred_knn)
    stars['log_flux_over_knn'] = lr.groupby(f.tic_id).median()
    stars['log_rp_flux'] = -.4 * np.log(10) * (stars.rp - 10)
    stars['response'] = stars.log_flux_over_knn - stars.log_rp_flux
    good = stars.rp.notna() & stars.color.notna()
    cal = good & (stars.role == 'train') & (stars.crowd > .95) & stars.rp.between(7,13.2) & stars.color.between(0,3.3)
    stars['color_prediction'] = np.nan
    color_models = {}
    for fold in range(5):
        fit = cal & (stars.fold != fold)
        query = good & (stars.fold == fold)
        model = HuberRegressor(epsilon=1.35, alpha=.0001, max_iter=500, tol=1e-8)
        model.fit(color_design(stars.loc[fit, 'color']), stars.loc[fit, 'response'])
        stars.loc[query, 'color_prediction'] = model.predict(color_design(stars.loc[query, 'color']))
        color_models[fold] = model
    stars['anchor'] = stars.log_rp_flux + stars.color_prediction
    f['gaia_log_anchor'] = f.tic_id.map(stars.anchor)
    f['has_gaia_anchor'] = f.gaia_log_anchor.notna()
    fallback = lr.groupby(f.tic_id).transform('median')
    f['gaia_log_anchor'] = f.gaia_log_anchor.fillna(fallback)
    evaluation = pd.read_parquet(OUT/'evaluation_manifest.parquet').set_index('tic_id').evaluation_group
    f['evaluation_group'] = f.tic_id.map(evaluation).fillna('train')
    f.to_parquet(run/'features.parquet', index=False)
    stars.to_parquet(run/'anchors.parquet')
    joblib.dump(color_models, run/'color_models.pkl')
    write_json(run/'prepared.json', dict(name=name, replicate=replicate, fraction=fraction, bank=bank,
        exclude=exclude, script_sha256=sha256(__file__), audit=audit,
        training_stars=int(f.loc[tr,'tic_id'].nunique()), training_rows=int(tr.sum()),
        color_training_stars=int(cal.sum()),
        missing_anchors={r:dict(rows=int((~g.has_gaia_anchor).sum()),
                              stars=int(g.loc[~g.has_gaia_anchor,'tic_id'].nunique())) for r,g in f.groupby('role')},
        features_sha256=sha256(run/'features.parquet')))
    log(f'{name}: prepared {tr.sum():,} training rows')


def training_residual(frame, total):
    residual = np.log(frame.sector_median.to_numpy()/frame.pred_knn.to_numpy()) - frame.gaia_log_anchor.to_numpy() - total
    missing = ~frame.has_gaia_anchor.to_numpy()
    if missing.any():
        residual[missing] -= group_median(residual[missing], frame.tic_id.to_numpy()[missing])
    return residual


def feature_names(family):
    return BASE + (PROFILE if family in ['profiles','combined'] else []) + (META if family in ['metadata','combined'] else [])


def fit_bank(name, family='base', rounds=ROUNDS, leaves=255, iterations=200):
    root = OUT/name
    run = root / f'{family}_r{rounds}_l{leaves}_i{iterations}'
    run.mkdir(exist_ok=True)
    if (run/'complete.json').exists():
        return
    f = pd.read_parquet(root/'features.parquet')
    train = f[f.role == 'train'].reset_index(drop=True)
    query = f[f.role == 'validation'].reset_index(drop=True)
    feats = feature_names(family)
    med = train[feats].replace([np.inf,-np.inf],np.nan).median().fillna(0)
    xtrain = train[feats].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(np.float32)
    xquery = query[feats].replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(np.float32)
    params = {**PARAMS, 'max_leaf_nodes':leaves, 'max_iter':iterations}
    config = dict(params=params, rounds=rounds, family=family, features=feats, seed=20261011,
                  prepared_sha256=sha256(root/'prepared.json'), script_sha256=sha256(__file__))
    if (run/'config.json').exists():
        assert json.loads((run/'config.json').read_text()) == config
    else:
        write_json(run/'config.json', config)
    joblib.dump(med, run/'imputation.pkl')
    total = np.zeros(len(train))
    pred = np.zeros(len(query))
    output = query[['source_row','tic_id','sector','cam','ccd','col','row','sector_median',
                    'systematic_score','has_gaia_anchor','gaia_log_anchor','evaluation_group','pred_knn']].copy()
    history=[]
    for rnd in range(rounds):
        started=time.monotonic()
        target = training_residual(train,total)
        modelpath=run/f'round_{rnd}.pkl'
        if modelpath.exists():
            model=joblib.load(modelpath)
        else:
            model=HistGradientBoostingRegressor(random_state=20261011+rnd,**params)
            log(f'{name}/{family}: fitting round {rnd+1}/{rounds}')
            model.fit(xtrain,target)
            joblib.dump(model,modelpath)
        total += model.predict(xtrain)
        pred += model.predict(xquery)
        output[f'log_correction_r{rnd+1}'] = np.log(query.pred_knn.to_numpy()) + pred
        output.to_parquet(run/'predictions.parquet',index=False)
        # Only discovery metrics are opened during model iteration.
        hist={'round':rnd+1,'seconds':time.monotonic()-started}
        for qualified in [False,True]:
            keep=query.evaluation_group.eq('discovery') & ((query.systematic_score>.95) if qualified else True)
            hist['qualified_cv' if qualified else 'all_cv']=float(per_star(query[keep],np.exp(output.loc[keep,f'log_correction_r{rnd+1}'])).median())
        history.append(hist)
        write_json(run/'progress.json',dict(history=history,status='training'))
        log(f'{name}/{family}: round {rnd+1}, discovery all {hist["all_cv"]:.4f}%, qualified {hist["qualified_cv"]:.4f}%')
    write_json(run/'complete.json',dict(history=history,prediction_sha256=sha256(run/'predictions.parquet')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['manifest','prepare','fit','run'])
    p.add_argument('--name',default='r0_half_A');p.add_argument('--replicate',type=int,default=0)
    p.add_argument('--fraction',type=float,default=.5);p.add_argument('--bank',choices=['A','B','full'],default='A')
    p.add_argument('--family',choices=['base','profiles','metadata','combined'],default='base')
    p.add_argument('--exclude',choices=['late_sectors','central_column'])
    p.add_argument('--rounds',type=int,default=ROUNDS);p.add_argument('--leaves',type=int,default=255);p.add_argument('--iterations',type=int,default=200)
    a=p.parse_args()
    if a.stage=='manifest':prepare_manifest()
    if a.stage in ['prepare','run']:prepare_bank(a.name,a.replicate,a.fraction,a.bank,a.exclude)
    if a.stage in ['fit','run']:fit_bank(a.name,a.family,a.rounds,a.leaves,a.iterations)


if __name__=='__main__':main()
