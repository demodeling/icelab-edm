"""
Score the sixth round (run_gaps.py): the adopted model, which interpolates a missing lag week,
against its variable-step version (vt_direct.py).

The hybrid uses the variable-step version where the initial state (weeks s-1 .. s-E+1 of the
forecast row) has a missing week, and the adopted model everywhere else.  Where the
variable-step version cannot be fitted (status other than "ok") the hybrid keeps the adopted model.

real  the weekly run's selection on 85 past rounds: its series with a missing week (fc_gaps_real/),
      and the mean rank score of the whole selection with and without the hybrid
      (fc_stage3/ adopted variant, fc_order/, fc_stale/ k = 1).
fake  the series without a missing week, with weeks removed at random (fc_gaps_fake/).
Differences: rank score, paired, with a moving-block bootstrap over origin dates.
"""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import analyse as an
import replay_scoring as sc
import run_gaps as rg
import run_replay as rr
import run_stale as rs

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
ADOPTED = 'direct_log_diff_season_off10'
SER = ['origin_date', 'target', 'location']
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)


def scored(files):
    files = [f for f in files if os.path.getsize(f) > 2]
    fc = pd.concat([pd.read_csv(f, dtype={'transform': str, 'error': str, 'level': str}) for f in files], ignore_index=True)
    fc['transform'] = fc['transform'].fillna('none')
    q = np.full((len(fc), len(sc.Q)), np.nan)
    for tr, g in fc.groupby('transform'):
        if tr:
            q[g.index] = sc.quantiles(g['mean'].to_numpy(), g['sd'].to_numpy(), tr)
    return an.score(fc, q)


def paired(d, a, b, seed=1, B=2000):
    """Mean of columns a and b over the rows where both are present, and of a - b with a
    95% moving-block bootstrap interval over origin dates (with fewer than 8 origin dates,
    where blocks of 4 cannot vary, a plain bootstrap over series-dates)."""
    d = d.dropna(subset=[a, b])
    if d.empty:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    ods = np.array(sorted(d.origin_date.unique()))
    if len(ods) >= 8:
        oi, idx = np.searchsorted(ods, d.origin_date), an._blocks(len(ods), rng, B)
        nc = len(ods)
    else:
        oi = pd.factorize(d.origin_date + d.target + d.location)[0]
        nc = oi.max() + 1
        idx = rng.integers(0, nc, size=(B, nc))
    S = np.zeros((3, nc))
    np.add.at(S[0], oi, d[a].to_numpy()); np.add.at(S[1], oi, d[b].to_numpy()); np.add.at(S[2], oi, 1.0)
    tot = S[:, idx].sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        boot = (tot[0] - tot[1]) / tot[2]
    return dict(n=len(d), **{a: d[a].mean(), b: d[b].mean()}, diff=d[a].mean() - d[b].mean(),
                lo=np.nanpercentile(boot, 2.5), hi=np.nanpercentile(boot, 97.5))


def relwis(d, a, b):
    """Geometric mean over series of (summed WIS of a) / (summed WIS of b), same forecasts."""
    d = d.dropna(subset=[a, b])
    g = d.groupby(['target', 'location'])[[a, b]].sum()
    g = g[(g[a] > 0) & (g[b] > 0)]
    return float(np.exp(np.mean(np.log(g[a] / g[b])))) if len(g) else np.nan


def wide(fc, idx):
    """One row per forecast, columns <measure>_<model>, plus the variable-step status and the
    initial-state gaps; hyb_* = the hybrid, vtf_* = the variable-step version with the fallback."""
    w = fc.set_index(idx + ['model'])[['rank_score', 'wis', 'in50', 'in95', 'status', 'init_missing', 'n_missing', 'E']].unstack('model')
    out = pd.DataFrame(index=w.index)
    for m in ('rank_score', 'wis', 'in50', 'in95'):
        for model in w[m].columns:
            out[f'{m}_{model}'] = w[m][model]
    out['init_missing'] = w['init_missing']['standard']
    out['n_missing'] = w['n_missing']['standard']
    out['E'] = w['E']['standard']
    if 'vt' in w['status'].columns:
        out['vt_status'] = w['status']['vt']
        ok = out.vt_status == 'ok'
        for m in ('rank_score', 'wis', 'in50', 'in95'):
            out[f'{m}_vtf'] = np.where(ok, out[f'{m}_vt'], out[f'{m}_standard'])
            out[f'{m}_hyb'] = np.where(out.init_missing > 0, out[f'{m}_vtf'], out[f'{m}_standard'])
    return out.reset_index()


def init_run(x, E):
    """Length of the run of missing weeks that reaches into the initial state (0 if none)."""
    n = len(x)
    miss = [n - 1 - j for j in range(1, E) if np.isnan(x[n - 1 - j])]
    if not miss:
        return 0
    i, r = max(miss), 0
    while i - r >= 0 and np.isnan(x[i - r]):
        r += 1
    return r


def case_info(ods):
    """Initial-state gap length of every real series-date, from the input caches."""
    rows = []
    for od in ods:
        sets = (('regular', rr.load_cases(od)), ('order', rs.load_stale(od, order=True)),
                ('one_behind', [c for c in rs.load_stale(od) if c['k'] == 1]))
        for kind, cs in sets:
            for c in cs:
                x = np.asarray(c['x'], dtype=float)
                rows.append(dict(kind=kind, origin_date=od, target=c['target'], location=c['location'],
                                 init_run=init_run(x, c['E'])))
    return pd.DataFrame(rows)


def line(d, label, a='hyb', b='standard'):
    p = paired(d, f'rank_score_{a}', f'rank_score_{b}')
    if not p['n']:
        return dict(group=label, n=0)
    return dict(group=label, series_dates=d.drop_duplicates(SER).shape[0], ranked=p['n'],
                dates=d.dropna(subset=[f'rank_score_{a}', f'rank_score_{b}']).origin_date.nunique(),
                **{b: p[f'rank_score_{b}'], a: p[f'rank_score_{a}']}, diff=p['diff'], lo=p['lo'], hi=p['hi'],
                rel_wis=relwis(d, f'wis_{a}', f'wis_{b}'),
                **{f'cover50_{b}': d[f'in50_{b}'].mean(), f'cover50_{a}': d[f'in50_{a}'].mean(),
                   f'cover95_{b}': d[f'in95_{b}'].mean(), f'cover95_{a}': d[f'in95_{a}'].mean()})


def real():
    fc = scored(sorted(glob.glob(os.path.join(HERE, 'fc_gaps_real', '*.csv'))))
    w = wide(fc, ['kind'] + SER + ['horizon'])
    info = case_info(sorted(w.origin_date.unique()))
    w = w.merge(info, on=['kind'] + SER, how='left')
    g = w[w.init_missing > 0]
    print(f"REAL GAPS: {w.drop_duplicates(['kind'] + SER).shape[0]} series-dates of the weekly run's selection with a missing week, "
          f"{g.drop_duplicates(['kind'] + SER).shape[0]} with one in the initial state")
    print('variable-step status on those:', g.vt_status.value_counts().to_dict())

    rows = [line(g, 'initial state has a gap: all')]
    for kind in ('regular', 'order', 'one_behind'):
        rows.append(line(g[g.kind == kind], f'  {kind}'))
    rows.append(line(g[g.init_run >= 8], '  after a break of 8+ weeks'))
    rows.append(line(g[(g.init_run > 0) & (g.init_run < 8)], '  gap of 1-7 weeks'))
    for h in (1, 2, 3, 4):
        rows.append(line(g[g.horizon == h], f'  horizon {h}'))
    rows.append(line(w[w.init_missing == 0], 'gap only further back: variable step instead (not used by the hybrid)', 'vtf'))
    rows.append(line(w, 'every series with a gap: variable step instead', 'vtf'))
    t = pd.DataFrame(rows).set_index('group')
    print('\nHYBRID (variable step where the initial state has a gap) AGAINST THE ADOPTED MODEL')
    print(t.round(3).to_string())
    t.to_csv(os.path.join(RES, 'gaps_real.csv'))

    by = g.dropna(subset=['rank_score_hyb', 'rank_score_standard']).groupby(['target', 'location']).agg(
        series_dates=('origin_date', 'nunique'), ranked=('horizon', 'size'), init_run=('init_run', 'median'),
        standard=('rank_score_standard', 'mean'), hybrid=('rank_score_hyb', 'mean'))
    by['diff'] = by.hybrid - by.standard
    print('\nBY SERIES, forecasts whose initial state has a gap')
    print(by.sort_values('ranked', ascending=False).round(3).to_string())
    by.to_csv(os.path.join(RES, 'gaps_real_by_series.csv'))

    # the whole selection of the weekly run: adopted model everywhere, or the hybrid
    base = [scored(sorted(glob.glob(os.path.join(HERE, 'fc_stage3', '*.csv')))).query('variant == @ADOPTED').assign(kind='regular'),
            scored(sorted(glob.glob(os.path.join(HERE, 'fc_order', '*.csv')))).assign(kind='order'),
            scored(sorted(glob.glob(os.path.join(HERE, 'fc_stale', '*.csv')))).query('k == 1').assign(kind='one_behind')]
    base = pd.concat(base, ignore_index=True)[['kind'] + SER + ['horizon', 'rank_score', 'wis']]
    m = base.merge(w[['kind'] + SER + ['horizon', 'rank_score_standard', 'rank_score_hyb', 'rank_score_vtf', 'wis_standard']],
                   on=['kind'] + SER + ['horizon'], how='left')
    chk = m.dropna(subset=['wis_standard'])
    print(f"\ncheck: the adopted model here and in the earlier rounds, {len(chk)} forecasts, max |WIS difference| "
          f"{np.nanmax(np.abs(chk.wis - chk.wis_standard)):.2e}")
    m['rank_score_base'] = m.rank_score
    m['rank_score_hybrid'] = m.rank_score_hyb.fillna(m.rank_score)
    m['rank_score_vt_any_gap'] = m.rank_score_vtf.fillna(m.rank_score)
    rows = []
    for kind in ('regular', 'order', 'one_behind', None):
        d = m if kind is None else m[m.kind == kind]
        for alt in ('hybrid', 'vt_any_gap'):
            p = paired(d, f'rank_score_{alt}', 'rank_score_base')
            rows.append(dict(selection=kind or 'all (weekly run from 2026-10-21)', alternative=alt, ranked=p['n'],
                             adopted=p['rank_score_base'], alt=p[f'rank_score_{alt}'], diff=p['diff'], lo=p['lo'], hi=p['hi']))
    t2 = pd.DataFrame(rows).set_index(['selection', 'alternative'])
    print('\nMEAN RANK SCORE OF THE WEEKLY RUN SELECTION')
    print(t2.round(4).to_string())
    t2.to_csv(os.path.join(RES, 'gaps_real_selection.csv'))


def forced_lag(c, level, draw):
    """Lag positions (1 .. E-1) missing in the forced arm of one fake series."""
    for lv, dr, arm, miss in rg.fake_masks(c, len(c['x'])):
        if lv == level and dr == draw and arm == 'forced':
            n = len(miss)
            return min(j for j in range(1, c['E']) if miss[n - 1 - j])


def fake():
    fc = scored(sorted(glob.glob(os.path.join(HERE, 'fc_gaps_fake', '*.csv'))))
    fc['level'] = fc.level.astype(str)
    idx = SER + ['horizon']
    ref = fc[fc.arm == 'none'].set_index(idx)[['rank_score', 'wis', 'in50', 'in95']].add_suffix('_ref')
    sub = fc[fc.arm != 'none']
    w = wide(sub, ['level', 'draw', 'arm'] + idx).merge(ref.reset_index(), on=idx, how='left')
    nfail = sub.groupby(['level', 'arm', 'model'])['mean'].apply(lambda v: int(v.isna().sum())).unstack('model')
    print(f"\nFAKE GAPS: {w.drop_duplicates(SER).shape[0]} series-dates without a missing week, "
          f"{w.drop_duplicates(['target', 'location']).shape[0]} series, {fc.origin_date.nunique()} origin dates")
    print('forecasts not made (horizon rows), by level, arm and model:')
    print(nfail.to_string())

    order = ['0', 'one', '0.02', '0.05', '0.1', '0.2', '0.3']
    rows = []
    for arm in ('random', 'forced'):
        for lv in order:
            d = w[(w.level == lv) & (w.arm == arm)]
            if d.empty:
                continue
            d = d.dropna(subset=['rank_score_ref', 'rank_score_standard', 'rank_score_hyb'])
            init = d[d.init_missing > 0]
            r = dict(arm=arm, level=lv, ranked=len(d), weeks_missing=d.n_missing.mean(), init_gappy=(d.init_missing > 0).mean(),
                     vt_fallback=(init.vt_status != 'ok').mean() if len(init) else np.nan,
                     no_gaps=d.rank_score_ref.mean(), standard=d.rank_score_standard.mean(), hybrid=d.rank_score_hyb.mean(),
                     vt_everywhere=d.rank_score_vtf.mean())
            for a, b, name in (('hyb', 'standard', 'hyb-std'), ('vtf', 'standard', 'vt-std'), ('standard', 'ref', 'std-nogap'), ('hyb', 'ref', 'hyb-nogap')):
                p = paired(d, f'rank_score_{a}', f'rank_score_{b}')
                r[name], r[name + '_lo'], r[name + '_hi'] = p['diff'], p['lo'], p['hi']
            p = paired(init, 'rank_score_vtf', 'rank_score_standard')
            r.update(init_ranked=p['n'], init_std=p.get('rank_score_standard'), init_vt=p.get('rank_score_vtf'),
                     **{'init_vt-std': p.get('diff'), 'init_lo': p.get('lo'), 'init_hi': p.get('hi')},
                     cover50_std=d.in50_standard.mean(), cover50_hyb=d.in50_hyb.mean(), cover95_std=d.in95_standard.mean(), cover95_hyb=d.in95_hyb.mean())
            rows.append(r)
    t = pd.DataFrame(rows).set_index(['arm', 'level'])
    cols1 = ['ranked', 'weeks_missing', 'init_gappy', 'vt_fallback', 'no_gaps', 'standard', 'hybrid', 'vt_everywhere',
             'hyb-std', 'hyb-std_lo', 'hyb-std_hi', 'std-nogap', 'std-nogap_lo', 'std-nogap_hi', 'hyb-nogap']
    cols2 = ['init_ranked', 'init_std', 'init_vt', 'init_vt-std', 'init_lo', 'init_hi', 'vt-std', 'vt-std_lo', 'vt-std_hi',
             'cover50_std', 'cover50_hyb', 'cover95_std', 'cover95_hyb']
    print('\nRANK SCORE BY FAKE GAP LEVEL (share of weeks removed at random; "one" = one week), same forecasts in a row')
    print(t[cols1].round(3).to_string())
    print('\n... the forecasts whose initial state has a gap, variable step against interpolation; '
          'variable step everywhere against interpolation; interval coverage')
    print(t[cols2].round(3).to_string())
    t.to_csv(os.path.join(RES, 'gaps_fake.csv'))

    # forced arm: by which lag of the initial state is missing (the most recent missing one)
    f = w[(w.arm == 'forced')].dropna(subset=['rank_score_ref', 'rank_score_standard', 'rank_score_vtf']).copy()
    cases = {}
    for od in f.origin_date.unique():
        for c in rr.load_cases(od):
            cases[(od, c['target'], c['location'])] = c
    keys = f[SER + ['level', 'draw']].drop_duplicates()
    lag = {(r.origin_date, r.target, r.location, r.level, r.draw):
           forced_lag(cases[(r.origin_date, r.target, r.location)], 0 if r.level == '0' else (r.level if r.level == 'one' else float(r.level)), r.draw)
           for r in keys.itertuples()}
    f['lag'] = [lag[k] for k in zip(f.origin_date, f.target, f.location, f.level, f.draw)]
    rows = []
    for j in sorted(f.lag.unique()):
        d = f[f.lag == j]
        p1, p2 = paired(d, 'rank_score_standard', 'rank_score_ref'), paired(d, 'rank_score_vtf', 'rank_score_standard')
        rows.append(dict(missing_lag=j, ranked=p1['n'], no_gaps=p1['rank_score_ref'], standard=p1['rank_score_standard'],
                         vt=p2['rank_score_vtf'], **{'std-nogap': p1['diff'], 'vt-std': p2['diff'], 'vt-std_lo': p2['lo'], 'vt-std_hi': p2['hi']}))
    t3 = pd.DataFrame(rows).set_index('missing_lag')
    print('\nFORCED ARM, ALL LEVELS: BY THE MOST RECENT MISSING WEEK OF THE INITIAL STATE (lag 1 = the week before the newest)')
    print(t3.round(3).to_string())
    t3.to_csv(os.path.join(RES, 'gaps_fake_by_lag.csv'))
    return t


def figure(t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'0': '0', 'one': '1 week', '0.02': '2%', '0.05': '5%', '0.1': '10%', '0.2': '20%', '0.3': '30%'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, arm, title in ((axes[0], 'random', 'Weeks removed at random'),
                           (axes[1], 'forced', 'One week of the initial state removed, and others at random')):
        d = t.loc[arm]
        xs = np.arange(len(d))
        ax.axhline(d.no_gaps.mean(), color='#888888', lw=1, ls='--')
        ax.text(len(d) - 1, d.no_gaps.mean() + 0.002, 'no gaps', ha='right', va='bottom', fontsize=8, color='#555555')
        ax.plot(xs, d.standard, marker='o', ms=4, lw=2, color='#2a6fdb', label='adopted (interpolation)')
        ax.plot(xs, d.hybrid, marker='s', ms=4, lw=2, color='#e8833a', label='hybrid (variable step if the initial state has a gap)')
        ax.set_xticks(xs, [labels[v] for v in d.index])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('share of weeks removed' if arm == 'random' else 'share of the other weeks removed')
        ax.grid(axis='y', color='#dddddd', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[1].annotate('no other gap: the variable step has\nnothing to learn from, falls back', (0, t.loc['forced'].hybrid.iloc[0]),
                     xytext=(0.3, t.loc['forced'].hybrid.iloc[0] - 0.012), fontsize=7, color='#555555',
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=0.6))
    axes[0].set_ylabel('mean hub rank score (1 = best model)')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(RES, 'gaps_fake.png'), dpi=150)


if __name__ == '__main__':
    os.makedirs(RES, exist_ok=True)
    real()
    t = fake()
    figure(t)
