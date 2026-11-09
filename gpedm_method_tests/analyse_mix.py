"""
The mix of a short and a very long E with both tuned (fourth round, fc_stage4/ldo_E*).

Members: one GP per horizon on the log scale (relative offset), change as response, no
time-of-year input, E in variants.MIX_SHORT and variants.MIX_LONG.  A mix is the average of
the two members' quantiles ("qavg") or their equally weighted mixture distribution ("pool").

1. Grid: every (short, long) pair, scored over the whole replay.
2. Pair chosen on season 2024/25, scored on the later rounds.
3. Tuning per series and round, with what is known at the time:
   "loo"   the E with the highest leave-one-out log likelihood per training row of the fit;
   "past"  the E with the lowest mean WIS on the series' earlier forecasts already observed.
4. A long-E member added to the adopted model (which has the time of year as input).
All on the forecasts for which every member exists.
"""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd
from scipy.stats import norm

import analyse as an
import replay_scoring as sc
import variants as va

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
ADOPTED = 'direct_log_diff_season_off10'
SHORT, LONG = list(va.MIX_SHORT), list(va.MIX_LONG)
ALLE = SHORT + LONG
KEY = an.KEY


def read(folder, keep):
    files = [f for f in sorted(glob.glob(os.path.join(HERE, folder, '*.csv'))) if os.path.getsize(f) > 2]
    d = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str}) for f in files], ignore_index=True)
    return d[d.variant.isin(keep)]


mem = read('fc_stage4', [f'ldo_E{e}' for e in ALLE])
ado = read('fc_stage3', [ADOPTED]).set_index(KEY)
prod = read('fc', ['prod']).set_index(KEY)

wide = {c: mem.pivot_table(index=KEY, columns='variant', values=c) for c in ('mean', 'sd', 'lnl_loo', 'nrows_h')}
have_all = wide['mean'].notna().all(axis=1)
idx = wide['mean'].index[have_all].intersection(ado.index[ado['mean'].notna()]).intersection(prod.index[prod['mean'].notna()])
truth = ado.loc[idx, 'truth'].to_numpy()
idx = idx[np.isfinite(truth)]
truth = ado.loc[idx, 'truth'].to_numpy()
print(f"forecasts with every member: {len(idx)} of {int(ado['mean'].notna().sum())} of the adopted model "
      f"({idx.droplevel(['origin_date', 'horizon']).nunique()} series, {idx.get_level_values('origin_date').nunique()} origin dates)")

offset = ado.loc[idx, 'transform'].str.split(':').str[1].astype(float).to_numpy()   # the same offset in every member
M = {e: wide['mean'].loc[idx, f'ldo_E{e}'].to_numpy() for e in ALLE}
S = {e: wide['sd'].loc[idx, f'ldo_E{e}'].to_numpy() for e in ALLE}
CRIT = {e: (wide['lnl_loo'].loc[idx, f'ldo_E{e}'] / wide['nrows_h'].loc[idx, f'ldo_E{e}']).to_numpy() for e in ALLE}


def quant(m, s):
    return np.clip(np.exp(m[:, None] + s[:, None] * sc.ZQ) - offset[:, None], 0, None)


def pool(parts):
    """Quantiles of the equally weighted mixture of log-scale Gaussians (m, s) with the common offset."""
    qs = np.stack([quant(m, s) for m, s in parts])
    lo, hi = qs.min(axis=0), qs.max(axis=0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        F = np.mean([norm.cdf((np.log(mid + offset[:, None]) - m[:, None]) / s[:, None]) for m, s in parts], axis=0)
        below = F < sc.Q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


QE = {e: quant(M[e], S[e]) for e in ALLE}
Q_ADO = quant(ado.loc[idx, 'mean'].to_numpy(), ado.loc[idx, 'sd'].to_numpy())
Q_PROD = sc.quantiles(prod.loc[idx, 'mean'].to_numpy(), prod.loc[idx, 'sd'].to_numpy(), 'none')

# hub field of every forecast
base = an.score(ado.loc[idx].reset_index(), Q_ADO).set_index(KEY).loc[idx]
usable = ((base.hub_n >= 3) & (base.ok == True)).to_numpy()
hub_min, hub_max = base.hub_min.to_numpy(), base.hub_max.to_numpy()
od = idx.get_level_values('origin_date').to_numpy()
ods = np.array(sorted(set(od)))
oi = np.searchsorted(ods, od)
season = np.array([an.season_of(o) for o in od])
BLOCKS = an._blocks(len(ods), np.random.default_rng(1))


def rs(q):
    w = sc.wis(q, truth)
    hi, lo = np.maximum(hub_max, w), np.minimum(hub_min, w)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(usable & (hi > lo), (hi - w) / (hi - lo), np.nan), w


def stats(q, ref_q=None, mask=None):
    """Mean rank score, and its difference from a reference with a block-bootstrap interval."""
    r, w = rs(q)
    m = np.isfinite(r) if mask is None else np.isfinite(r) & mask
    out = dict(n=int(m.sum()), rank_score=float(np.mean(r[m])), cover50=float(np.mean((truth >= q[:, an.I25]) & (truth <= q[:, an.I75]))),
               cover95=float(np.mean((truth >= q[:, an.I025]) & (truth <= q[:, an.I975]))))
    if ref_q is not None:
        r0, w0 = rs(ref_q)
        Sm = np.zeros((2, len(ods)))
        np.add.at(Sm[0], oi[m], (r - r0)[m]); np.add.at(Sm[1], oi[m], 1.0)
        tot = Sm[:, BLOCKS].sum(axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = tot[0] / tot[1]
        out.update(diff=float(np.mean((r - r0)[m])), lo=float(np.nanpercentile(boot, 2.5)), hi=float(np.nanpercentile(boot, 97.5)),
                   wis_ratio=float(np.sum(w[m]) / np.sum(w0[m])))
    return out


print(f"ranked against the hub: {int(np.isfinite(rs(Q_ADO)[0]).sum())}")
print('\nREFERENCES on these forecasts')
refs = {'production model before 21 Oct (prod)': Q_PROD, 'adopted model (time of year, E by FNN)': Q_ADO}
print(pd.DataFrame({k: stats(v, Q_ADO) for k, v in refs.items()}).T.round(3).to_string())

print('\nSINGLE E (no time-of-year input), difference from the adopted model')
single = pd.DataFrame({e: stats(QE[e], Q_ADO) for e in ALLE}).T
single.index.name = 'E'
print(single.round(3).to_string())
single.to_csv(os.path.join(RES, 'mix_single_E.csv'))

# ---- 1. grid ----
grid = {how: pd.DataFrame(index=SHORT, columns=LONG, dtype=float) for how in ('qavg', 'pool')}
QMIX = {}
for a in SHORT:
    for b in LONG:
        QMIX[(a, b, 'qavg')] = 0.5 * (QE[a] + QE[b])
        QMIX[(a, b, 'pool')] = pool([(M[a], S[a]), (M[b], S[b])])
        for how in grid:
            grid[how].loc[a, b] = np.nanmean(rs(QMIX[(a, b, how)])[0])
for how in grid:
    print(f'\n1. GRID, mean rank score of the mix ({how}); rows: short E, columns: long E')
    print(grid[how].round(3).to_string())
    grid[how].to_csv(os.path.join(RES, f'mix_grid_{how}.csv'))
best = {how: max(((a, b) for a in SHORT for b in LONG), key=lambda p: grid[how].loc[p]) for how in grid}
print('\nbest pair over the whole replay:', best)
rows = {}
for how in grid:
    a, b = best[how]
    rows[f'best pair {a}+{b} ({how}) against 3+52'] = stats(QMIX[(a, b, how)], QMIX[(3, 52, how)])
    rows[f'best pair {a}+{b} ({how}) against the adopted model'] = stats(QMIX[(a, b, how)], Q_ADO)
    rows[f'3+52 ({how}) against the adopted model'] = stats(QMIX[(3, 52, how)], Q_ADO)
    rows[f'best pair {a}+{b} ({how}) against short E={a} alone'] = stats(QMIX[(a, b, how)], QE[a])
print(pd.DataFrame(rows).T.round(3).to_string())
spread = grid['qavg'].loc[:, [50, 51, 52, 53, 54]]
print(f"\nrange of the qavg grid over long E = 50..54: {spread.min().min():.3f} to {spread.max().max():.3f}; "
      f"over all long E for short E = 3: {grid['qavg'].loc[3].min():.3f} to {grid['qavg'].loc[3].max():.3f}")

# ---- 2. pair chosen on 2024/25, scored afterwards ----
first = season == '2024/25'
print('\n2. PAIR CHOSEN ON SEASON 2024/25, SCORED ON THE LATER ROUNDS')
rows = {}
for how in grid:
    g1 = pd.DataFrame({b: [np.nanmean(rs(QMIX[(a, b, how)])[0][first]) for a in SHORT] for b in LONG}, index=SHORT)
    a, b = max(((a, b) for a in SHORT for b in LONG), key=lambda p: g1.loc[p])
    rows[f'{how}: chosen {a}+{b}, later rounds, against 3+52'] = stats(QMIX[(a, b, how)], QMIX[(3, 52, how)], mask=~first)
    rows[f'{how}: chosen {a}+{b}, later rounds, against the adopted model'] = stats(QMIX[(a, b, how)], Q_ADO, mask=~first)
    rows[f'{how}: 3+52, later rounds, against the adopted model'] = stats(QMIX[(3, 52, how)], Q_ADO, mask=~first)
print(pd.DataFrame(rows).T.round(3).to_string())

# ---- 3. tuning per series and round ----
def pick(score, cands, largest=True):
    """For every forecast, the candidate E with the best score (score: dict E -> array)."""
    A = np.vstack([score[e] for e in cands])
    A = np.where(np.isfinite(A), A, -np.inf if largest else np.inf)
    j = A.argmax(axis=0) if largest else A.argmin(axis=0)
    return np.array(cands)[j]


def assemble(choice_s, choice_l, how):
    q = np.empty_like(Q_ADO)
    for a in SHORT:
        for b in LONG:
            m = (choice_s == a) & (choice_l == b)
            if m.any():
                q[m] = QMIX[(a, b, how)][m]
    return q


# "past": mean WIS of the series' earlier forecasts whose target week was observed before this round
ser = pd.factorize(idx.get_level_values('target') + '|' + idx.get_level_values('location'))[0]
ted = pd.to_datetime(base.target_end_date).to_numpy()
known_before = (pd.to_datetime(od) - pd.Timedelta(days=3)).to_numpy()   # horizon 1 ends on the Sunday before the origin date;
                                                                        # weeks before it were observed when the round was run
WIS = {e: sc.wis(QE[e], truth) for e in ALLE}
PAST = {e: np.full(len(idx), np.nan) for e in ALLE}
MIN_PAST = 8
for s_ in np.unique(ser):
    rows_s = np.flatnonzero(ser == s_)
    for o in np.unique(od[rows_s]):
        cur = rows_s[od[rows_s] == o]
        prev = rows_s[(od[rows_s] < o) & (ted[rows_s] < known_before[cur[0]])]
        if len(prev) >= MIN_PAST:
            for e in ALLE:
                PAST[e][cur] = WIS[e][prev].mean()
tuned = {'loo': (pick(CRIT, SHORT), pick(CRIT, LONG)),
         'past': (pick(PAST, SHORT, largest=False), pick(PAST, LONG, largest=False))}
no_past = ~np.isfinite(PAST[3])
tuned['past'][0][no_past], tuned['past'][1][no_past] = 3, 52          # no history yet: the untuned pair
print(f'\n3. TUNED PER SERIES AND ROUND (forecasts without {MIN_PAST} earlier observed forecasts use 3+52 in "past": {no_past.mean():.2f} of all)')
rows = {}
for name, (cs, cl) in tuned.items():
    for how in ('qavg', 'pool'):
        q = assemble(cs, cl, how)
        rows[f'tuned by {name} ({how}) against 3+52'] = stats(q, QMIX[(3, 52, how)])
        rows[f'tuned by {name} ({how}) against the adopted model'] = stats(q, Q_ADO)
    print(f'   chosen by {name}: short E', pd.Series(cs).value_counts(normalize=True).sort_index().round(2).to_dict(),
          '\n      long E', pd.Series(cl).value_counts(normalize=True).sort_index().round(2).to_dict())
t3 = pd.DataFrame(rows).T
print(t3.round(3).to_string())
t3.to_csv(os.path.join(RES, 'mix_tuned.csv'))
# the single E chosen by each rule, as a check of the rule itself
for name, score, largest in (('loo', CRIT, True), ('past', PAST, False)):
    c = pick(score, ALLE, largest)
    q = np.empty_like(Q_ADO)
    for e in ALLE:
        q[c == e] = QE[e][c == e]
    print(f'   single E chosen by {name} among all {len(ALLE)}: rank score {np.nanmean(rs(q)[0]):.3f} '
          f'(E = 3 alone {np.nanmean(rs(QE[3])[0]):.3f}); share choosing a long E {np.isin(c, LONG).mean():.2f}')

# ---- 4. long E added to the adopted model ----
print('\n4. ADOPTED MODEL MIXED WITH A LONG-E MEMBER (qavg), against the adopted model alone')
rows = {b: stats(0.5 * (Q_ADO + QE[b]), Q_ADO) for b in LONG}
t4 = pd.DataFrame(rows).T
t4.index.name = 'long E'
print(t4.round(3).to_string())
t4.to_csv(os.path.join(RES, 'mix_adopted_plus_long.csv'))
a, b = best['qavg']
rows = {f'adopted + E={a} + E={b} (thirds)': stats((Q_ADO + QE[a] + QE[b]) / 3, Q_ADO),
        'adopted + E=3 + E=52 (thirds)': stats((Q_ADO + QE[3] + QE[52]) / 3, Q_ADO),
        f'adopted + short E={a}': stats(0.5 * (Q_ADO + QE[a]), Q_ADO)}
print(pd.DataFrame(rows).T.round(3).to_string())
