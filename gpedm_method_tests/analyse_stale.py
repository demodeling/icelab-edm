"""
Score the fifth round (fc_stale/, from run_stale.py): forecasts for series whose latest week
is missing on the origin date, made from the last known value with horizons shifted by the
number of missing weeks k.  Compared with the forecasts the weekly run already makes
(adopted model, fc_stage3/direct_log_diff_season_off10).
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


def scored(files):
    fc = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str}) for f in files if os.path.getsize(f) > 2], ignore_index=True)
    q = np.full((len(fc), len(sc.Q)), np.nan)
    for tr, g in fc.groupby('transform'):
        q[g.index] = sc.quantiles(g['mean'].to_numpy(), g['sd'].to_numpy(), tr)
    return an.score(fc, q)


st = scored(sorted(glob.glob(os.path.join(HERE, 'fc_stale', '*.csv'))))
reg = scored(sorted(glob.glob(os.path.join(HERE, 'fc_stage3', '*.csv'))))
reg = reg[reg.variant == ADOPTED]
print(f"skipped series-dates with a last value 1 to 8 weeks old: {st.groupby(['origin_date', 'target', 'location']).ngroups}; "
      f"forecasts {len(st)}, with an observation later {int(st.wis.notna().sum())}, ranked against at least 3 hub models {int(st.rank_score.notna().sum())}")
print(f"forecasts the weekly run makes now: {int(reg.wis.notna().sum())} scored, {int(reg.rank_score.notna().sum())} ranked\n")


def line(d):
    r = d[d.rank_score.notna()]
    return dict(series_dates=d.groupby(['origin_date', 'target', 'location']).ngroups, scored=int(d.wis.notna().sum()), ranked=len(r),
                rank_score=r.rank_score.mean(), hub_ensemble=r.rs_ensemble.mean(), hub_baseline=r.rs_baseline.mean(),
                other_models=r.hub_n.mean(), cover50=d.in50.mean(), cover95=d.in95.mean())


rows = [dict(group='up to date (submitted now)', **line(reg))]
for k in sorted(st.k.unique()):
    rows.append(dict(group=f'k = {k}', **line(st[st.k == k])))
rows.append(dict(group='k = 1 to 8', **line(st)))
t = pd.DataFrame(rows).set_index('group')
print('BY NUMBER OF MISSING WEEKS k (hub horizons 1 to 4 are k+1 to k+4 weeks after the last value)')
print(t.round(3).to_string())
t.to_csv(os.path.join(RES, 'stale_by_k.csv'))

print('\nBY HUB HORIZON, k = 1 to 8')
print(pd.DataFrame({h: line(st[st.horizon == h]) for h in (1, 2, 3, 4)}).T.round(3).to_string())

# the model's mean rank score with the extra forecasts included
rr = reg.rank_score.dropna()


def with_extra(extra, label, seed=1):
    """Mean rank score of the up-to-date forecasts plus `extra`, and its change, with a block
    bootstrap over origin dates of the change."""
    both = pd.concat([rr, extra.rank_score.dropna()])
    d = pd.concat([reg.assign(extra=False), extra.assign(extra=True)])
    d = d[d.rank_score.notna()]
    ods = np.array(sorted(d.origin_date.unique()))
    oi = np.searchsorted(ods, d.origin_date)
    S = np.zeros((4, len(ods)))            # sums and counts: regular, all
    np.add.at(S[0], oi[~d.extra.to_numpy()], d.rank_score[~d.extra].to_numpy()); np.add.at(S[1], oi[~d.extra.to_numpy()], 1)
    np.add.at(S[2], oi, d.rank_score.to_numpy()); np.add.at(S[3], oi, 1)
    idx = an._blocks(len(ods), np.random.default_rng(seed))
    tot = S[:, idx].sum(axis=2)
    boot = tot[2] / tot[3] - tot[0] / tot[1]
    return dict(include=label, forecasts=len(both), rank_score=both.mean(),
                change=both.mean() - rr.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))


rows = [dict(include='only up to date', forecasts=len(rr), rank_score=rr.mean(), change=0.0)]
for kmax in (1, 2, 3, 4, 8):
    rows.append(with_extra(st[st.k <= kmax], f'+ last value up to {kmax} week(s) old'))
t2 = pd.DataFrame(rows).set_index('include')
print('\nMEAN RANK SCORE OF THE MODEL IF THESE FORECASTS ARE ALSO SUBMITTED')
print(t2.round(4).to_string())
t2.to_csv(os.path.join(RES, 'stale_overall.csv'))

# are the skipped countries harder, or is it the age of the last value?
by = st[st.rank_score.notna()].groupby(['target', 'location']).agg(stale_ranked=('rank_score', 'size'), stale_rank_score=('rank_score', 'mean'),
                                                                   mean_k=('k', 'mean'), hub_ensemble=('rs_ensemble', 'mean'))
by['up_to_date_ranked'] = reg[reg.rank_score.notna()].groupby(['target', 'location']).size()
by['up_to_date_rank_score'] = reg[reg.rank_score.notna()].groupby(['target', 'location']).rank_score.mean()
by['diff'] = by.stale_rank_score - by.up_to_date_rank_score
print('\nBY SERIES (at least 12 ranked forecasts from an old last value)')
print(by[by.stale_ranked >= 12].sort_values('stale_ranked', ascending=False).round(3).to_string())
both = by.dropna(subset=['up_to_date_rank_score'])
w = both.stale_ranked
print(f"\nseries seen both ways: {len(both)}; weighted by their old-value forecasts, rank score {np.average(both.stale_rank_score, weights=w):.3f} "
      f"from an old last value against {np.average(both.up_to_date_rank_score, weights=w):.3f} when up to date")
only = by[by.up_to_date_rank_score.isna()]
print(f"series never up to date in the replay (never forecast now): {len(only)}, {int(only.stale_ranked.sum())} ranked forecasts, "
      f"rank score {np.average(only.stale_rank_score, weights=only.stale_ranked) if len(only) else float('nan'):.3f}: {sorted(' '.join((a[:3], b)) for a, b in only.index)}")
by.to_csv(os.path.join(RES, 'stale_by_series.csv'))

# WIS relative to the hub ensemble on the same forecasts
for label, d in (('up to date', reg), ('old last value', st)):
    r = d[d.rank_score.notna() & d.hub_ensemble.notna()]
    print(f"{label}: median WIS ratio to the hub ensemble {np.median(r.wis / r.hub_ensemble.clip(lower=1e-9)):.2f}, "
          f"to the hub baseline {np.median(r.wis / r.hub_baseline.clip(lower=1e-9)):.2f} (n = {len(r)})")

# ---- the weekly run from round 2026-10-21 ----
# The location is chosen by its latest date (not its first row) and may be one week behind.  Up to
# round 2026-10-14 the first-row test also skipped up-to-date series whose rows are listed oldest
# first (UK nations, Swiss ILI); run_stale.py --order replays those (fc_order/).
of = [f for f in sorted(glob.glob(os.path.join(HERE, 'fc_order', '*.csv'))) if os.path.getsize(f) > 2]
if of:
    orr = scored(of)
    print('\nTHE WEEKLY RUN FROM ROUND 2026-10-21: SERIES CHOSEN BY THEIR LATEST DATE, UP TO ONE WEEK BEHIND')
    t3 = pd.DataFrame([dict(group='newest week in the first row (submitted up to 2026-10-14)', **line(reg)),
                       dict(group='newest week present, rows oldest first (skipped up to 2026-10-14)', **line(orr)),
                       dict(group='one week behind (k = 1)', **line(st[st.k == 1]))]).set_index('group')
    print(t3.round(3).to_string())
    print('series with rows oldest first:', sorted(' '.join((a[:3], b)) for a, b in orr[['target', 'location']].drop_duplicates().to_numpy()))
    t4 = pd.DataFrame([dict(include='submitted up to 2026-10-14', forecasts=len(rr), rank_score=rr.mean(), change=0.0),
                       with_extra(orr, '+ rows oldest first'),
                       with_extra(pd.concat([orr, st[st.k == 1]]), '+ rows oldest first + one week behind (from 2026-10-21)')]).set_index('include')
    print('\nMEAN RANK SCORE OF THE MODEL')
    print(t4.round(4).to_string())
    pd.concat([t3, t4]).to_csv(os.path.join(RES, 'missing_week.csv'))
