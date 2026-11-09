"""Score the third round (fc_pooled/, from run_pooled.py): one GP for all countries against
the same model fitted to each country alone, on the origin dates both were run for."""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import analyse as an
import replay_scoring as sc

HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)
files = sorted(glob.glob(os.path.join(HERE, 'fc_pooled', '*.csv')))
fp = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str}) for f in files], ignore_index=True)
q = sc.quantiles(fp['mean'].to_numpy(), fp['sd'].to_numpy(), 'log')
fp = an.score(fp, q)
main = pd.read_csv(os.path.join(HERE, 'results', 'scored.csv.gz'), dtype={'transform': str, 'error': str})
main = main[main.variant.isin(['prod', 'direct_log_diff_season', 'prod_log_season']) & main.origin_date.isin(fp.origin_date.unique())]
fc = pd.concat([fp, main], ignore_index=True)
one = fp[(fp.variant == 'pooled_log_diff_season') & (fp.horizon == 1)]
print(f"{fp.origin_date.nunique()} origin dates, {int(fp[fp.variant == 'pooled_log_diff_season'].wis.notna().sum())} scored pooled forecasts; "
      f"rows per pooled fit: median {one.nrows.median():.0f}, max {one.nrows.max():.0f}; "
      f"fits at the 200-iteration limit: {(fp[fp.variant == 'pooled_log_diff_season'].iter >= 200).mean():.2f}; median rho {fp.rho.median():.2f}")
names = ['prod', 'prod_log_season', 'direct_log_diff_season', 'single_log_diff_season', 'pooled_log_diff_season']
t = an.compare(fc, names)
an.show('against prod', t)
t2 = an.compare(fc, ['pooled_log_diff_season', 'direct_log_diff_season'], ref='single_log_diff_season')
an.show('against the same model fitted to each country alone (E = 3, last 208 weeks)', t2)
for h in (1, 2, 3, 4):
    an.show(f'horizon {h}', an.compare(fc, ['pooled_log_diff_season'], ref='single_log_diff_season', subset=fc.horizon == h),
            cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
winter = fc.origin_date.str[5:7].isin(['11', '12', '01', '02', '03'])
for label, m in (('November to March', winter), ('April to October', ~winter)):
    an.show(label, an.compare(fc, ['pooled_log_diff_season'], ref='single_log_diff_season', subset=m),
            cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
for tg in ('ARI incidence', 'ILI incidence'):
    an.show(tg, an.compare(fc, ['pooled_log_diff_season'], ref='single_log_diff_season', subset=fc.target == tg),
            cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
t.to_csv(os.path.join(HERE, 'results', 'pooled_vs_prod.csv')); t2.to_csv(os.path.join(HERE, 'results', 'pooled_vs_single.csv'))
