"""
Score the neighbour covariates of the fourth round (fc_stage4/: nb_mean, nb_each, all_mean;
fc_stage4b/: nb_growth, nb_gap, nb_growth_gap) against the adopted model without covariates (fc_stage3/direct_log_diff_season_off10).
"""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import analyse as an
import replay_scoring as sc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
ADOPTED = 'direct_log_diff_season_off10'
NB = ['nb_mean', 'nb_each', 'all_mean', 'nb_growth', 'nb_gap', 'nb_growth_gap']

files = sorted(f for d in ('fc_stage4', 'fc_stage4b', 'fc_stage3') for f in glob.glob(os.path.join(HERE, d, '*.csv')))
fc = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str}) for f in files if os.path.getsize(f) > 2], ignore_index=True)
fc = fc[fc.variant.isin(NB + [ADOPTED])].reset_index(drop=True)
q = np.full((len(fc), len(sc.Q)), np.nan)
for tr, g in fc.groupby('transform'):
    q[g.index] = sc.quantiles(g['mean'].to_numpy(), g['sd'].to_numpy(), tr)
fc = an.score(fc, q)
cols = ('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'rel_wis_lo', 'rel_wis_hi', 'better', 'cover50', 'cover95')
print('failed fits (rows):', fc[fc.error.fillna('') != ''].groupby('variant').size().to_dict())
NB = [v for v in NB if (fc.variant == v).any()]
used = fc[fc.variant.isin(NB)].groupby('variant').n_nb.agg(share_with_covariates=lambda s: (s > 0).mean(), mean_countries=lambda s: s[s > 0].mean())
print(used.round(2).to_string())

t = an.compare(fc, NB, ref=ADOPTED)
an.show('ALL FORECASTS, against the adopted model (a series with no neighbour available gets the adopted forecast)', t, cols)
t.to_csv(os.path.join(RES, 'neighbours.csv'))
key = ['target', 'location', 'origin_date']
for v in NB:
    has = fc[(fc.variant == v) & (fc.n_nb > 0)].set_index(key).index.unique()
    sub = fc.set_index(key).index.isin(has)
    an.show(f'{v}: only the series-dates where covariates were used', an.compare(fc, [v], ref=ADOPTED, subset=sub), cols)
for h in (1, 2, 3, 4):
    an.show(f'HORIZON {h}', an.compare(fc, NB, ref=ADOPTED, subset=fc.horizon == h), cols[:5] + ('rel_wis', 'cover50', 'cover95'))
winter = fc.origin_date.str[5:7].isin(['11', '12', '01', '02', '03'])
for label, m in (('NOVEMBER TO MARCH', winter), ('APRIL TO OCTOBER', ~winter)):
    an.show(label, an.compare(fc, NB, ref=ADOPTED, subset=m), cols[:5] + ('rel_wis', 'cover50', 'cover95'))
for s_ in sorted(fc.season.unique()):
    an.show(f'SEASON {s_}', an.compare(fc, NB, ref=ADOPTED, subset=fc.season == s_), cols[:5] + ('rel_wis', 'cover50', 'cover95'))
for tg in ('ARI incidence', 'ILI incidence'):
    an.show(tg, an.compare(fc, NB, ref=ADOPTED, subset=fc.target == tg), cols[:5] + ('rel_wis', 'cover50', 'cover95'))

ok = fc[fc.rank_score.notna()]
ps = ok.pivot_table(index=['target', 'location'], columns='variant', values='rank_score')[[ADOPTED] + NB]
ps['n'] = ok[ok.variant == ADOPTED].groupby(['target', 'location']).size()
ps['countries_nb_mean'] = fc[fc.variant == 'nb_mean'].groupby(['target', 'location']).n_nb.mean()
for v in NB:
    ps[v + ' - adopted'] = ps[v] - ps[ADOPTED]
print('\nBY SERIES')
print(ps.sort_values('nb_mean - adopted').round(3).to_string())
big = ps[ps.n >= 40]
for v in NB:
    print(f"{v}: better than the adopted model in {int((big[v + ' - adopted'] > 0).sum())} of {len(big)} series")
ps.to_csv(os.path.join(RES, 'neighbours_by_series.csv'))
