"""
The weekly run's choice of locations (respi_edm.run_analysis, method="direct", from round
2026-10-21) on past hub snapshots.  For six origin dates, the snapshot of the Friday before is
given to run_analysis the way the run_final notebook gives it the latest hub files, and the
file it writes is compared with the replay of the same series:

    newest week in the first row              -> fc_stage3, the adopted variant
    newest week present, rows oldest first    -> fc_order  (run_stale.py --order)
    newest week missing, last value 1 week old -> fc_stale, k = 1
    last value 2 or more weeks old            -> no forecast

Run from ~/git/Respi on HPC2N after run_stale.py (both kinds), with OMP_NUM_THREADS=1.
"""
import contextlib
import io
import os
import sys
import warnings

sys.path[:0] = ['.', 'gpedm_method_tests']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import replay_data as rd
import replay_scoring as sc
import respi_edm as res

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'weekly_trial')
ADOPTED = 'direct_log_diff_season_off10'
KEY = ['target', 'location', 'horizon']

# the origin dates with the most series one week behind, and with the most rows listed oldest first
sk = pd.read_csv(os.path.join(HERE, 'results', 'snapshot_k.csv'))
sk = sk[sk.nobs >= rd.MIN_OBS]
n1 = sk[sk.k == 1].groupby('od').size().sort_values(ascending=False, kind='stable')
n0 = sk[(sk.k == 0) & ~sk.first_ok].groupby('od').size().sort_values(ascending=False, kind='stable')
ods = list(dict.fromkeys(list(n1.index[:3]) + [od for od in n0.index if od not in n1.index[:3]][:3]))


def replay(od):
    """The replay's Gaussians for one origin date, with the group each series came from."""
    parts = []
    f = os.path.join(HERE, 'fc_stage3', f'{od}.0of1.csv')
    d = pd.read_csv(f)
    parts.append(d[d.variant == ADOPTED].assign(group='newest week in first row', k=0))
    f = os.path.join(HERE, 'fc_order', f'{od}.csv')
    if os.path.getsize(f) > 2:                      # run_stale.py writes an empty file when there is none
        parts.append(pd.read_csv(f).assign(group='rows oldest first'))
    d = pd.read_csv(os.path.join(HERE, 'fc_stale', f'{od}.csv'))
    parts.append(d[d.k == 1].assign(group='one week missing'))
    r = pd.concat(parts, ignore_index=True)
    q = np.full((len(r), len(sc.Q)), np.nan)
    for tr, g in r.groupby('transform'):
        q[g.index] = sc.quantiles(g['mean'].to_numpy(), g['sd'].to_numpy(), tr)
    return r, q


ok = True
summary = []
for od in ods:
    fw = rd.forecasting_weeks()
    dfv = fw[fw.origin_date == od].sort_values('horizon').set_index('origin_date')
    newest = pd.Timestamp(dfv['target_end_date'].to_numpy()[1])
    trial = os.path.join(OUT, f'{od}-IceLab-EDM.missing_week_check.csv')
    if os.path.exists(trial):
        os.remove(trial)
    k_of, nobs_of, msgs = {}, {}, []
    for target in rd.TARGETS:
        df = rd.snapshot(od, target)
        if df is None:
            continue
        for loc, d in df[df.value.notna()].groupby('location'):
            k_of[(target, loc)] = (newest - pd.Timestamp(d.truth_date.max())).days // 7
            nobs_of[(target, loc)] = len(d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res.run_analysis(trial, df, np.unique(df['location']), dfv, origin_date=od, target_incidence=target)
        msgs += [f'{target[:3]} {l}' for l in buf.getvalue().splitlines() if 'no forecast' in l or 'newest week missing' in l]
    new = pd.read_csv(trial)
    os.remove(trial)

    qn = new[new.output_type == 'quantile'].pivot_table(index=KEY, columns='output_type_id', values='value')
    med = new[new.output_type == 'median'].set_index(KEY).value
    valid = dict(no_nan=bool(new.value.notna().all()), non_negative=bool((new.value >= 0).all()),
                 increasing=bool((np.diff(qn.to_numpy(), axis=1) >= 0).all()),
                 median_is_q50=bool(np.allclose(med.loc[qn.index], qn[0.5])),
                 target_weeks=all(set(new[new.horizon == h].target_end_date) == {dfv['target_end_date'].to_numpy()[h + 1]}
                                  for h in (1, 2, 3, 4)))
    written = set(map(tuple, new[['target', 'location']].drop_duplicates().to_numpy()))
    ks = {s: k_of[s] for s in written}

    r, qr = replay(od)
    rkey = list(zip(r.target, r.location, r.horizon))
    have = r['mean'].notna().to_numpy()
    rseries = set(map(tuple, r.loc[have, ['target', 'location']].drop_duplicates().to_numpy()))
    common = sorted(written & rseries)
    worst = 0.0
    for s in common:
        for h in (1, 2, 3, 4):
            i = rkey.index((*s, h))
            a, b = qn.loc[(*s, h)].to_numpy(), qr[i]
            worst = max(worst, np.abs(a - b).max() / max(1.0, np.abs(b).max()))
    only_weekly = sorted(written - rseries)            # expected: fewer observed weeks than the replay's minimum
    only_replay = sorted(rseries - written)
    groups = r.drop_duplicates(['target', 'location']).set_index(['target', 'location']).group
    counts = pd.Series([groups.get(s, 'not replayed') for s in written]).value_counts().to_dict()
    this_ok = (all(valid.values()) and max(ks.values()) <= res.MAX_WEEKS_MISSING and worst < 1e-9 and not only_replay
               and all(nobs_of[s] < rd.MIN_OBS for s in only_weekly))
    ok &= this_ok
    print(f'{od}: {len(written)} series written {counts}; weeks missing among them: {sorted(set(ks.values()))}')
    print(f'   valid: {valid}')
    print(f'   against the replay: {len(common)} series, max relative difference of the quantiles {worst:.1e}; '
          f'only in the weekly file: {[" ".join((t[:3], l)) + f" ({nobs_of[(t, l)]} weeks)" for t, l in only_weekly]}; '
          f'only in the replay: {only_replay}')
    behind = sorted((k, f'{t[:3]} {l}') for (t, l), k in k_of.items() if k > res.MAX_WEEKS_MISSING)
    print(f'   not forecast, 2 or more weeks behind: {len(behind)} ({", ".join(f"{s} {k}" for k, s in behind[:8])}{" ..." if len(behind) > 8 else ""})')
    print(f'   messages: {len(msgs)}, e.g. {msgs[:2]}')
    print('   OK' if this_ok else '   FAILED', flush=True)
    summary.append(dict(origin_date=od, written=len(written), **counts, worst=worst, ok=this_ok))

print(pd.DataFrame(summary).to_string(index=False))
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECK FAILED')
