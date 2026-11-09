"""
Checks of respi_edm.run_analysis(method="direct"), the weekly run from round 2026-10-21.
Inputs: the three hub files saved on 12 October 2026 in gpedm_py_validation/inputs/, the
ones round 2026-10-14 was submitted from.  Run from ~/git/Respi.

1. The replay code after the move to gpedm_direct gives the forecasts stored before it.
2. run_analysis with its defaults writes a valid hub file for the series of the submitted
   file plus those it skipped although their last observation is at most MAX_WEEKS_MISSING
   weeks old, and its numbers are those of the replayed variant.
3. method="vt" still reproduces the file of 12 October.
"""
import contextlib
import io
import os
import sys
import warnings

sys.path[:0] = ['.', 'gpedm_method_tests', 'gpedm_py_validation', '../edm_playground/src/eDNAcode/']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import gpedm_direct as gd
import replay_data as rd
import replay_scoring as sc
import respi_edm as res
import run_replay as rr
import variants as va
from hub_inputs import load_inputs

OUT = 'gpedm_method_tests/weekly_trial'
os.makedirs(OUT, exist_ok=True)
NAME = 'direct_log_diff_season_off10'
KEY = ['target', 'location', 'horizon']
ok = True

# ---- 1. regression of the replay code ----
for od in ('2026-10-14', '2025-02-12'):
    old = pd.read_csv(f'gpedm_method_tests/fc_stage3/{od}.0of1.csv')
    old = old[old.variant == NAME].set_index(KEY)
    new = []
    peers = rr.load_cases(od)
    for c in peers:
        mean, sd, tr, E, info = va.run_variant(NAME, c, peers)
        for h in range(4):
            new.append(dict(target=c['target'], location=c['location'], horizon=h + 1, mean=mean[h], sd=sd[h], transform=tr))
    new = pd.DataFrame(new).set_index(KEY).loc[old.index]
    dm, ds = np.abs(new['mean'] - old['mean']).max(), np.abs(new['sd'] - old['sd']).max()
    same_tr = (new['transform'] == old['transform']).all()
    print(f'1. {od}: {len(old)} forecasts, max |mean diff| {dm:.1e}, max |sd diff| {ds:.1e}, same offsets: {same_tr}')
    ok &= dm < 1e-9 and ds < 1e-9 and same_tr

# ---- 2. the weekly call, exactly as the run_final notebook makes it ----
dfv, data = load_inputs()
origin = dfv.index[2]
trial = f'{OUT}/{origin}-IceLab-EDM.csv'
if os.path.exists(trial):
    os.remove(trial)
tables = {}
with contextlib.redirect_stdout(io.StringIO()):
    for key in ('ARI', 'ILI'):
        df = data[key]
        tables[key] = res.run_analysis(trial, df, np.unique(df['location']), dfv, origin_date=origin,
                                       target_incidence=df['target'].to_numpy()[0])
new = pd.read_csv(trial)
sub = pd.read_csv(f'{origin}-IceLab-EDM.csv')
print(f'2. trial file {trial}: {new.shape[0]} rows; submitted file: {sub.shape[0]} rows')
cols_same = list(new.columns) == list(sub.columns)
kcols = ['origin_date', 'target', 'target_end_date', 'horizon', 'location', 'output_type', 'output_type_id']
# From 14 October the weekly run also forecasts the locations whose last observation is at most
# res.MAX_WEEKS_MISSING weeks old, judged by the latest date rather than the first row.
newest = pd.Timestamp(dfv['target_end_date'].to_numpy()[1])
weeks_missing = {}
for key in ('ARI', 'ILI'):
    df = data[key]
    for loc, d in df[df.value.notna()].groupby('location'):
        weeks_missing[(df['target'].to_numpy()[0], loc)] = (newest - pd.Timestamp(d.truth_date.max())).days // 7
sub_series = set(map(tuple, sub[['target', 'location']].drop_duplicates().to_numpy()))
new_series = set(map(tuple, new[['target', 'location']].drop_duplicates().to_numpy()))
expected_extra = {s for s, k in weeks_missing.items() if 0 <= k <= res.MAX_WEEKS_MISSING} - sub_series
in_sub = new[[s in sub_series for s in zip(new.target, new.location)]].reset_index(drop=True)
keys_same = in_sub[kcols].astype(str).equals(sub[kcols].astype(str))
extra_ok = sub_series <= new_series and new_series - sub_series == expected_extra
print(f'   same columns: {cols_same}; submitted series: same rows in the same order (all columns but value): {keys_same}')
print(f'   series added: {sorted(" ".join((t[:3], l)) for t, l in new_series - sub_series)} '
      f'(weeks missing: {sorted({weeks_missing[s] for s in new_series - sub_series})}); as expected: {extra_ok}')
print('   series:', new[new.horizon == 1].groupby('target').location.nunique().to_dict())
q = new[new.output_type == 'quantile'].pivot_table(index=['target', 'location', 'horizon'], columns='output_type_id', values='value')
med = new[new.output_type == 'median'].set_index(['target', 'location', 'horizon']).value
checks = dict(no_nan=bool(new.value.notna().all()), non_negative=bool((new.value >= 0).all()), n_quantiles=q.shape[1] == 23,
              increasing=bool((np.diff(q.to_numpy(), axis=1) >= 0).all()), median_is_q50=bool(np.allclose(med.loc[q.index], q[0.5])))
print('  ', checks)
ok &= cols_same and keys_same and extra_ok and all(checks.values())

# the same series through the replay code (run_stale.py's call for a location behind)
worst = 0.0
for key in ('ARI', 'ILI'):
    df = data[key]
    for loc in np.unique(df['location']):
        d = df[df['location'] == loc]
        k = weeks_missing.get((df['target'].to_numpy()[0], loc), -1)
        if not 0 <= k <= res.MAX_WEEKS_MISSING or (df['target'].to_numpy()[0], loc) not in new_series:
            continue
        x, E, tau = rd.date_series(d)
        case = dict(x=x, E=E, first_week=d.truth_date.min(), target=df['target'].to_numpy()[0], location=loc)
        if k == 0:
            mean, sd, tr, E, info = va.run_variant(NAME, case)
        else:
            tr = gd.log_transform(x)
            mean, sd, _ = gd.direct_forecast(x, E, case['first_week'], transform=tr, season=True, diff=True,
                                             horizons=tuple(k + h for h in gd.HORIZONS))
        qr = sc.quantiles(mean, sd, tr)
        qn = q.loc[(case['target'], loc)].to_numpy()
        worst = max(worst, np.abs(qn - qr).max() / max(1.0, np.abs(qr).max()))
print(f'   trial file against the replay code on the same inputs: max relative difference {worst:.1e}')
ok &= worst < 1e-9

# against the forecasts the replay stored for this round (hub snapshot of the Friday before)
st = pd.read_csv(f'gpedm_method_tests/fc_stage3/{origin}.0of1.csv')
st = st[st.variant == NAME]
st['median'] = [sc.quantiles(np.array([m]), np.array([s]), t)[0, 11] for m, s, t in zip(st['mean'], st['sd'], st['transform'])]
j = st.set_index(KEY)['median'].to_frame('replay').join(med.rename('trial'), how='inner')
rel = (j.replay - j.trial).abs() / j.replay.abs().clip(lower=1)
print(f'   against the stored replay of this round: {len(j)} medians, identical to 1e-9: {(rel < 1e-9).sum()}, max relative difference {rel.max():.1e}')

# ---- 3. the previous method ----
# method="vt" against the module as it was before the change (respi_edm.py.bak_26_10_13_pre_direct),
# in this process, so that both use the same BLAS threads and the same random state.
from importlib.machinery import SourceFileLoader
before = SourceFileLoader('respi_edm_before', 'respi_edm.py.bak_26_10_13_pre_direct').load_module()
files = {}
for label, call in (('before', lambda *a, **k: before.run_analysis(*a, backend='python', **k)),
                    ('now', lambda *a, **k: res.run_analysis(*a, backend='python', method='vt', **k))):
    f = f'{OUT}/{origin}-IceLab-EDM.vt_{label}.csv'
    if os.path.exists(f):
        os.remove(f)
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        for key in ('ARI', 'ILI'):
            df = data[key]
            call(f, df, np.unique(df['location']), dfv, origin_date=origin, target_incidence=df['target'].to_numpy()[0])
    files[label] = open(f, 'rb').read()
    os.remove(f)
same = files['before'] == files['now']
print(f'3. method="vt", backend="python": file identical byte for byte to the one the previous respi_edm.py writes: {same}')
ok &= same
import io as _io
a, b = pd.read_csv(_io.BytesIO(files['now'])), pd.read_csv(f'gpedm_py_validation/{origin}-IceLab-EDM.python.csv')
rel = (a.value - b.value).abs() / b.value.abs().clip(lower=1)
print(f'   against the Python file of 12 October (written with another number of BLAS threads): same rows '
      f'{a[kcols].astype(str).equals(b[kcols].astype(str))}, max relative difference {rel.max():.1e}, '
      f'series above 1e-9: {sorted(set(a[rel > 1e-9].location))}')

# how the two methods differ on this round
m = med.rename('direct').to_frame().join(sub[sub.output_type == 'median'].set_index(['target', 'location', 'horizon']).value.rename('submitted'))
print('\nmedians of this round, new method against the submitted one (ratio new / submitted):')
print((m.direct / m.submitted).groupby(level='horizon').describe()[['min', '25%', '50%', '75%', 'max']].round(2).to_string())
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECK FAILED')
