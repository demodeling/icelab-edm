"""
Score the replay (fc/*.csv from run_replay.py) and compare the variants.

    python3 gpedm_method_tests/analyse.py            # tables to stdout, CSVs to results/

Scores
  rank score   where the forecast's WIS would sit among the other hub models for the same
               target week: 1 = best model, 0 = worst (the hub's own summary measure).
  rel. WIS     WIS relative to the reference variant: ratio of mean WIS within each series,
               then the geometric mean over series (below 1 = better than the reference).
  cover 50/95  share of observations inside the central 50% and 95% intervals.
Intervals are from a moving-block bootstrap over origin dates (blocks of 4 weeks).
"""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import replay_data as rd
import replay_scoring as sc

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ['target', 'location', 'origin_date', 'horizon']
REF = 'prod'
Q = sc.Q
I25, I75, I025, I975, I50 = (int(np.argmin(np.abs(Q - v))) for v in (0.25, 0.75, 0.025, 0.975, 0.5))


def season_of(od):
    y, m = int(od[:4]), int(od[5:7])
    s = y if m >= 7 else y - 1
    return f'{s}/{str(s + 1)[2:]}'


def load():
    """All replay rows with their quantiles: (table of rows, array of quantiles n x 23)."""
    files = sorted(f for d in ('fc', 'fc_stage2', 'fc_stage3') for f in glob.glob(os.path.join(HERE, d, '*.csv')))
    files = [f for f in files if os.path.getsize(f) > 2]                                 # a share with no series is empty
    fc = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str}) for f in files], ignore_index=True)
    fc['transform'] = fc['transform'].fillna('none')
    q = np.full((len(fc), len(Q)), np.nan)
    for tr, g in fc.groupby('transform'):
        q[g.index] = sc.quantiles(g['mean'].to_numpy(), g['sd'].to_numpy(), tr)
    return fc, q


def add_mix(fc, q, name, parts, how):
    """Append a combination of existing variants (same rows of every part)."""
    sub = [fc[fc.variant == p].set_index(KEY) for p in parts]
    idx = sub[0].index
    for s in sub[1:]:
        idx = idx.intersection(s.index)
    sub = [s.loc[idx] for s in sub]
    ok = np.all([s['mean'].notna().to_numpy() & s['sd'].notna().to_numpy() for s in sub], axis=0)
    qm = np.full((len(idx), len(Q)), np.nan)
    if how == 'pool':        # mixture distribution
        # each variant uses a single transform
        comps = [(s['mean'].to_numpy()[ok], s['sd'].to_numpy()[ok], s['transform'][ok].iloc[0]) for s in sub]
        qm[ok] = sc.linear_pool(comps)
    else:                    # average of the quantiles
        qm[ok] = np.mean([q[s['_row'].to_numpy()[ok]] for s in sub], axis=0)
    new = sub[0].reset_index()[KEY + ['target_end_date', 'truth']].copy()
    new['variant'] = name
    for c in ('mean', 'sd', 'E', 'iter', 'nrows', 'seconds'):
        new[c] = np.nan
    new['transform'], new['error'] = 'mix', ''
    return pd.concat([fc, new], ignore_index=True), np.vstack([q, qm])


def score(fc, q):
    """Add WIS, interval coverage and the hub rank score to every row."""
    y = fc.truth.to_numpy()
    fc = fc.copy()
    fc['wis'] = np.where(np.isfinite(q).all(axis=1) & np.isfinite(y), sc.wis(q, y), np.nan)
    fc['in50'] = np.where(np.isfinite(fc.wis), (y >= q[:, I25]) & (y <= q[:, I75]), np.nan)
    fc['in95'] = np.where(np.isfinite(fc.wis), (y >= q[:, I025]) & (y <= q[:, I975]), np.nan)
    fc['median'] = q[:, I50]
    fc['width95'] = q[:, I975] - q[:, I025]

    s = pd.read_csv(rd.SCORES)
    s = s[(s.metric == 'WIS') & ~((s.team_id == 'IceLab') & (s.model_id == 'EDM'))]
    s['model'] = s.team_id + '-' + s.model_id
    hub = s.pivot_table(index=KEY, columns='model', values='value_absolute')
    others = pd.DataFrame({'hub_min': hub.min(axis=1), 'hub_max': hub.max(axis=1), 'hub_n': hub.notna().sum(axis=1),
                           'hub_ensemble': hub.get('respicast-hubEnsemble'), 'hub_baseline': hub.get('respicast-quantileBaseline')})
    fc = fc.merge(others.reset_index(), on=KEY, how='left')
    flag = pd.read_csv(os.path.join(HERE, 'truth_consistent.csv'))
    fc = fc.merge(flag, on=['target', 'location', 'target_end_date'], how='left')
    usable = (fc.hub_n >= 3) & (fc.ok == True) & fc.wis.notna()
    hi, lo = np.maximum(fc.hub_max, fc.wis), np.minimum(fc.hub_min, fc.wis)
    fc['rank_score'] = np.where(usable & (hi > lo), (hi - fc.wis) / (hi - lo), np.nan)
    for m in ('ensemble', 'baseline'):
        v = fc['hub_' + m]
        fc['rs_' + m] = np.where(usable & v.notna() & (hi > lo), (hi - v) / (hi - lo), np.nan)
    fc['season'] = fc.origin_date.map(season_of)
    return fc


def _blocks(n, rng, B=2000, L=4):
    """Indices of B moving-block resamples of n ordered origin dates."""
    L = min(L, n)
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(B, k))
    return (starts[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]


def compare(fc, variants, ref=REF, subset=None, B=2000, seed=1):
    """One line per variant, each scored on the forecasts it shares with the reference."""
    d = fc if subset is None else fc[subset]
    wide = d.pivot_table(index=KEY, columns='variant', values='wis')
    rs = d.pivot_table(index=KEY, columns='variant', values='rank_score')
    in50 = d.pivot_table(index=KEY, columns='variant', values='in50')
    in95 = d.pivot_table(index=KEY, columns='variant', values='in95')
    rng = np.random.default_rng(seed)
    rows = []
    for v in variants:
        if v not in wide.columns:
            continue
        both = pd.DataFrame({'ref': wide[ref], 'v': wide[v]}).dropna()
        if both.empty:
            continue
        od = both.index.get_level_values('origin_date')
        ods = np.array(sorted(od.unique()))
        oi = np.searchsorted(ods, od)
        series = pd.factorize(both.index.get_level_values('target') + both.index.get_level_values('location'))[0]
        ns = series.max() + 1
        A = np.zeros((2, len(ods), ns))                       # WIS sums by origin date and series
        np.add.at(A[0], (oi, series), both['ref'].to_numpy())
        np.add.at(A[1], (oi, series), both['v'].to_numpy())

        def relwis(a):
            num, den = a[1].sum(axis=0), a[0].sum(axis=0)
            ok = (den > 0) & (num > 0)
            return np.exp(np.mean(np.log(num[ok] / den[ok])))

        r2 = pd.DataFrame({'ref': rs[ref], 'v': rs[v]}).reindex(both.index).dropna()
        oi2 = np.searchsorted(ods, r2.index.get_level_values('origin_date'))
        S = np.zeros((3, len(ods)))
        np.add.at(S[0], oi2, r2['ref'].to_numpy()); np.add.at(S[1], oi2, r2['v'].to_numpy()); np.add.at(S[2], oi2, 1.0)
        idx = _blocks(len(ods), rng, B)
        boot_rw = np.array([relwis(A[:, i, :]) for i in idx[:400]])
        tot = S[:, idx].sum(axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            boot_d = (tot[1] - tot[0]) / tot[2]
        rows.append({
            'variant': v, 'n': len(both), 'n_ranked': len(r2),
            'rank_score': r2['v'].mean(), 'ref_rank_score': r2['ref'].mean(),
            'diff': r2['v'].mean() - r2['ref'].mean(),
            'diff_lo': np.nanpercentile(boot_d, 2.5), 'diff_hi': np.nanpercentile(boot_d, 97.5),
            'rel_wis': relwis(A), 'rel_wis_lo': np.percentile(boot_rw, 2.5), 'rel_wis_hi': np.percentile(boot_rw, 97.5),
            'better': (both['v'] < both['ref']).mean(),
            'cover50': in50[v].reindex(both.index).mean(), 'cover95': in95[v].reindex(both.index).mean(),
        })
    return pd.DataFrame(rows).set_index('variant')


def show(title, t, cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'rel_wis_lo', 'rel_wis_hi', 'better', 'cover50', 'cover95')):
    print('\n' + title)
    print(t[list(cols)].round(3).to_string())


SHORT_LONG = [(2, 26), (2, 52), (3, 26), (3, 52), (3, 13)]
# combinations across kinds of variant: does a very long E still add something once the time of year is an input?
EXTRA_MIXES = {
    'mix[log_diff_season + log_diff_E52]': ['direct_log_diff_season', 'direct_log_diff_E52'],
    'mix[log_season + log_diff_E52]': ['direct_log_season', 'direct_log_diff_E52'],
    'mix[log_season + log_diff_season]': ['direct_log_season', 'direct_log_diff_season'],
    'mix[log_diff_season + log_diff_E3 + log_diff_E52]': ['direct_log_diff_season', 'direct_log_diff_E3', 'direct_log_diff_E52'],
    'mix[prod_log_season + log_diff_E52]': ['prod_log_season', 'direct_log_diff_E52'],
    'mix[prod_log_season + log_diff_season]': ['prod_log_season', 'direct_log_diff_season'],
}


def main():
    fc, q = load()
    fc['_row'] = np.arange(len(fc))
    for base in ('direct', 'direct_log_diff'):
        for a, b in SHORT_LONG:
            for how in ('pool', 'qavg'):
                fc, q = add_mix(fc, q, f'{base}_mix{a}+{b}_{how}', [f'{base}_E{a}', f'{base}_E{b}'], how)
                fc['_row'] = np.arange(len(fc))
    for name, parts in EXTRA_MIXES.items():
        for how in ('pool', 'qavg'):
            if all((fc.variant == p).any() for p in parts):
                fc, q = add_mix(fc, q, f'{name}_{how}', parts, how)
                fc['_row'] = np.arange(len(fc))
    fc = score(fc, q)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    fc.drop(columns=['_row']).to_csv(os.path.join(HERE, 'results', 'scored.csv.gz'), index=False)

    names = list(pd.unique(fc.variant))
    print(f"{fc.origin_date.nunique()} origin dates, {fc.groupby(['target', 'location']).ngroups} series, "
          f"{fc[fc.variant == REF].wis.notna().sum()} scored forecasts of the reference, "
          f"{fc[fc.variant == REF].rank_score.notna().sum()} with a hub rank score")
    err = fc[fc.error.fillna('') != ''].groupby('variant').size()
    print('failed fits (rows):', err.to_dict())

    allt = compare(fc, names)
    allt.to_csv(os.path.join(HERE, 'results', 'all_variants.csv'))
    show('ALL VARIANTS, all origin dates, against prod', allt)
    for s_ in sorted(fc.season.unique()):
        t = compare(fc, names, subset=fc.season == s_)
        t.to_csv(os.path.join(HERE, 'results', f"by_season_{s_.replace('/', '_')}.csv"))
        show(f'SEASON {s_}', t, cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))
    for h in (1, 2, 3, 4):
        t = compare(fc, names, subset=fc.horizon == h)
        t.to_csv(os.path.join(HERE, 'results', f'by_horizon_{h}.csv'))
        show(f'HORIZON {h}', t, cols=('n', 'rank_score', 'diff', 'rel_wis', 'cover50', 'cover95'))
    for tg in rd.TARGETS:
        t = compare(fc, names, subset=fc.target == tg)
        t.to_csv(os.path.join(HERE, 'results', f'by_target_{tg[:3]}.csv'))
        show(f'TARGET {tg}', t, cols=('n', 'rank_score', 'diff', 'rel_wis', 'cover50', 'cover95'))

    winter = fc.origin_date.str[5:7].isin(['11', '12', '01', '02', '03'])
    for label, m in (('NOVEMBER TO MARCH', winter), ('APRIL TO OCTOBER', ~winter)):
        t = compare(fc, names, subset=m)
        t.to_csv(os.path.join(HERE, 'results', f"by_part_of_year_{label.split()[0].lower()}.csv"))
        show(f'ORIGIN DATES {label}', t, cols=('n', 'rank_score', 'diff', 'diff_lo', 'diff_hi', 'rel_wis', 'cover50', 'cover95'))

    r = fc[(fc.variant == REF) & fc.rank_score.notna()]
    print('\nhub reference models on the forecasts ranked for prod: ensemble', round(r.rs_ensemble.mean(), 3),
          ' quantile baseline', round(r.rs_baseline.mean(), 3), ' mean number of other models', round(r.hub_n.mean(), 1))


if __name__ == '__main__':
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    main()
