"""
Closer looks at the scored replay (results/scored.csv.gz, written by analyse.py):
the factorial of the three main changes, the E sweep and the short + long mixes,
the season input by part of the year, each series, and the year x week grid.
Writes results/summary.json and the tables behind the figure.
"""
import json
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import analyse as an

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)

fc = pd.read_csv(os.path.join(RES, 'scored.csv.gz'), dtype={'transform': str, 'error': str})
winter = fc.origin_date.str[5:7].isin(['11', '12', '01', '02', '03'])
summary = {}


def line(t, v):
    r = t.loc[v]
    return dict(rank_score=round(r.rank_score, 3), diff=round(r['diff'], 3), lo=round(r.diff_lo, 3), hi=round(r.diff_hi, 3),
                rel_wis=round(r.rel_wis, 3), cover50=round(r.cover50, 3), cover95=round(r.cover95, 3), n=int(r.n))


# ---- 1. the three changes, one GP per horizon: transform x response x time-of-year input ----
fact = ['direct', 'direct_season', 'direct_diff', 'direct_diff_season',
        'direct_log', 'direct_log_season', 'direct_log_diff', 'direct_log_diff_season']
steps = ['prod', 'prod_exactgrad', 'prod_grid', 'prod_season', 'prod_log', 'prod_log_season', 'direct', 'direct_sqrt'] + fact[1:]
t = an.compare(fc, steps)
an.show('MAIN VARIANTS against prod (all origin dates)', t)
summary['main'] = {v: line(t, v) for v in t.index}

# the same differences with one GP per horizon as the reference, to separate the scheme from the rest
t2 = an.compare(fc, fact, ref='direct')
an.show('ONE GP PER HORIZON: the other changes against plain `direct`', t2)

# ---- 2. E sweep and mixes ----
for base in ('direct', 'direct_log_diff'):
    names = [f'{base}_E{e}' for e in (1, 2, 3, 4, 6, 8, 13, 26, 52)]
    mixes = [f'{base}_mix{a}+{b}_{how}' for a, b in an.SHORT_LONG for how in ('pool', 'qavg')]
    # common set: forecasts where E = 52 exists, so every line is scored on the same forecasts
    has52 = fc[(fc.variant == f'{base}_E52') & fc.wis.notna()].set_index(an.KEY).index
    sub = fc.set_index(an.KEY).index.isin(has52)
    ts = an.compare(fc, names + mixes, subset=sub)
    an.show(f'E SWEEP AND MIXES on `{base}`, forecasts where E = 52 can be fitted', ts)
    ts.to_csv(os.path.join(RES, f'sweep_{base}.csv'))
    summary[f'sweep_{base}'] = {v: line(ts, v) for v in ts.index}
    # mix against the short E alone
    tm = an.compare(fc, mixes + [f'{base}_E52'], ref=f'{base}_E3', subset=sub)
    an.show(f'  the same mixes against `{base}_E3` alone', tm, cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
    summary[f'mix_vs_E3_{base}'] = {v: line(tm, v) for v in tm.index}

# does a very long E add anything once the time of year is an input?
extra = [v for v in pd.unique(fc.variant) if v.startswith('mix[')]
has52 = fc[(fc.variant == 'direct_log_diff_E52') & fc.wis.notna()].set_index(an.KEY).index
sub = fc.set_index(an.KEY).index.isin(has52)
te = an.compare(fc, extra + ['direct_log_diff_E52', 'direct_log_diff_mix3+52_qavg', 'direct_log_diff_mix3+52_pool'],
                ref='direct_log_diff_season', subset=sub)
an.show('AGAINST `direct_log_diff_season`: mixes with a very long E', te, cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
summary['vs_log_diff_season'] = {v: line(te, v) for v in te.index}

# ---- 3. the time-of-year input by part of the year ----
rows = []
for label, m in (('Nov-Mar', winter), ('Apr-Oct', ~winter)):
    for ref, v in (('prod', 'prod_season'), ('direct', 'direct_season'), ('direct_log', 'direct_log_season'),
                   ('direct_log_diff', 'direct_log_diff_season'), ('prod_log', 'prod_log_season'),
                   ('direct_log_diff_E3', 'direct_log_diff_mix3+52_qavg'), ('direct_E3', 'direct_mix3+52_qavg')):
        r = an.compare(fc, [v], ref=ref, subset=m).loc[v]
        rows.append(dict(part=label, reference=ref, variant=v, n=int(r.n), ref_rank_score=r.ref_rank_score, rank_score=r.rank_score,
                         diff=r['diff'], lo=r.diff_lo, hi=r.diff_hi, rel_wis=r.rel_wis))
tp = pd.DataFrame(rows)
print('\nGAIN FROM THE TIME-OF-YEAR INPUT (and from the 3 + 52 mix) BY PART OF THE YEAR')
print(tp.round(3).to_string(index=False))
tp.to_csv(os.path.join(RES, 'season_gain_by_part_of_year.csv'), index=False)
summary['by_part_of_year'] = json.loads(tp.round(4).to_json(orient='records'))

# ---- 4. each series ----
cand = ['prod', 'prod_log_season', 'direct_log_diff_season']
ok = fc[fc.variant.isin(cand) & fc.rank_score.notna()]
ps = ok.pivot_table(index=['target', 'location'], columns='variant', values='rank_score')[cand]
ps['n'] = ok[ok.variant == 'prod'].groupby(['target', 'location']).size()
for v in cand[1:]:
    ps[v + ' - prod'] = ps[v] - ps['prod']
ps = ps.sort_values('direct_log_diff_season - prod')
print('\nBY SERIES (mean rank score)')
print(ps.round(3).to_string())
ps.to_csv(os.path.join(RES, 'by_series.csv'))
big = ps[ps.n >= 40]
summary['by_series'] = {v: dict(series=int(len(big)), better=int((big[v + ' - prod'] > 0).sum()),
                                worst=float(big[v + ' - prod'].min()), best=float(big[v + ' - prod'].max()),
                                worst_series=' '.join(big[v + ' - prod'].idxmin()), median=float(big[v + ' - prod'].median()))
                        for v in cand[1:]}
print(json.dumps(summary['by_series'], indent=1))

# ---- 5. the year x week grid: only the series-dates where it changes the series ----
w = fc[fc.variant.isin(['prod', 'prod_grid'])].pivot_table(index=an.KEY, columns='variant', values='mean')
w = w.dropna()
changed = (w['prod'] - w['prod_grid']).abs() > 1e-6 * w['prod'].abs().clip(lower=1)
ch_series = changed.groupby(level=['target', 'location', 'origin_date']).any()
aff = ch_series[ch_series].index
m = fc.set_index(['target', 'location', 'origin_date']).index.isin(aff)
tg = an.compare(fc, ['prod_grid'], ref='prod', subset=m)
print(f'\nYEAR x WEEK GRID: {int(ch_series.sum())} of {len(ch_series)} series-dates where the production grid can be built give a different forecast')
print('affected series:', sorted({f'{a[0][:3]} {a[1]}' for a in aff}))
an.show('  prod_grid against prod (date grid), affected series-dates only; a negative diff means the date grid is better', tg)
summary['grid'] = dict(affected=int(ch_series.sum()), buildable=int(len(ch_series)), **line(tg, 'prod_grid'))

# ---- 6. calibration ----
cal = fc[fc.wis.notna() & fc.variant.isin(['prod', 'prod_log', 'prod_log_season', 'direct', 'direct_log', 'direct_log_diff_season', 'direct_log_diff_E52'])]
tc = cal.assign(part=np.where(winter[cal.index], 'Nov-Mar', 'Apr-Oct')).groupby(['variant', 'part', 'horizon'])[['in50', 'in95']].mean().unstack('horizon').round(2)
print('\nCOVERAGE of the 50% and 95% intervals by horizon')
print(tc.to_string())
summary['coverage'] = {f'{v}|{p}': dict(in50=[float(x) for x in r['in50']], in95=[float(x) for x in r['in95']])
                       for (v, p), r in tc.iterrows()}

r = fc[(fc.variant == 'prod') & fc.rank_score.notna()]
summary['hub'] = dict(ensemble=float(r.rs_ensemble.mean()), baseline=float(r.rs_baseline.mean()), n_models=float(r.hub_n.mean()),
                      n_ranked=int(len(r)), n_scored=int(fc[fc.variant == 'prod'].wis.notna().sum()),
                      origin_dates=int(fc[fc.wis.notna()].origin_date.nunique()), first=fc.origin_date.min(), last=fc[fc.wis.notna()].origin_date.max(),
                      series=int(fc.groupby(['target', 'location']).ngroups))
print('\n', json.dumps(summary['hub'], indent=1))
json.dump(summary, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
