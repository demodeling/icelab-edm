"""Score the third round (fc_stage3/): the log offset as a share of the series mean."""
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import analyse as an

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
fc, q = an.load()
keep = ['prod', 'prod_log_season', 'prod_log_season_off10', 'direct_log_diff_season', 'direct_log_diff_season_off10']
m = fc.variant.isin(keep).to_numpy()
fc, q = fc[m].reset_index(drop=True), q[m]
fc = an.score(fc, q)
off = fc[fc.variant == 'direct_log_diff_season_off10'].assign(c=lambda d: d['transform'].str.split(':').str[1].astype(float))
print('offsets used: median', round(off.c.median(), 1), ' share at the floor of 1:', round((off.c <= 1).mean(), 2))
t = an.compare(fc, keep)
an.show('against prod', t)
for ref, v in (('direct_log_diff_season', 'direct_log_diff_season_off10'), ('prod_log_season', 'prod_log_season_off10')):
    an.show(f'{v} against {ref}', an.compare(fc, [v], ref=ref), cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'better', 'cover50', 'cover95'))
ok = fc[fc.rank_score.notna()]
ps = ok.pivot_table(index=['target', 'location'], columns='variant', values='rank_score')[keep]
ps['n'] = ok[ok.variant == 'prod'].groupby(['target', 'location']).size()
ps['mean_c'] = off.groupby(['target', 'location']).c.mean()
ps['off10 - log1 (direct)'] = ps['direct_log_diff_season_off10'] - ps['direct_log_diff_season']
ps['off10 - prod (direct)'] = ps['direct_log_diff_season_off10'] - ps['prod']
ps['off10 - prod (prod scheme)'] = ps['prod_log_season_off10'] - ps['prod']
print(ps.sort_values('off10 - log1 (direct)').round(3).to_string())
big = ps[ps.n >= 40]
for c in ('off10 - prod (direct)', 'off10 - prod (prod scheme)'):
    print(c, ': better than prod in', int((big[c] > 0).sum()), 'of', len(big), 'series; worst', big[c].idxmin(), round(big[c].min(), 3))
t.to_csv(os.path.join(an.HERE, 'results', 'log_offset.csv')); ps.to_csv(os.path.join(an.HERE, 'results', 'log_offset_by_series.csv'))
