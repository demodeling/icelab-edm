"""
Numbers and checks behind gpedm_py_math.pdf.

Run on HPC2N from ~/git/Respi, with the modules of the weekly run loaded:
    python3 gpedm_py_docs/make_numbers.py
Writes gpedm_py_docs/numbers.json and gpedm_py_docs/trace_<series>.csv.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gpedm_py_validation'))
import numpy as np
import pandas as pd

from hub_inputs import load_inputs, series
import gpedm_py as gp

out = {}

# ---- 1. the toy series of the lag example: y_i = i, gaps at i = 4, 7, 8 ---------------
y = np.arange(1.0, 11.0)
y[[3, 6, 7]] = np.nan
toy = pd.DataFrame({'t': np.arange(1, 11), 'y': y})
lags = gp.makelags(toy, y='y', time='t', E=2, tau=1, vtimestep=True)
fore = gp.makelags(toy, y='y', time='t', E=2, tau=1, vtimestep=True, forecast=True, Tdiff_fore=[1, 2])
out['toy_lags'] = pd.concat([toy, lags], axis=1).astype(object).where(pd.notna(pd.concat([toy, lags], axis=1)), None).values.tolist()
out['toy_fore'] = fore.values.tolist()
out['toy_cols'] = list(toy.columns) + list(lags.columns)
fixed = gp.makelags(toy, y='y', time='t', E=2, tau=1)
out['toy_fixed'] = fixed.astype(object).where(pd.notna(fixed), None).values.tolist()

# ---- 2. two hub series: one where the horizons differ, one where they do not -----------
dfv, targets = load_inputs()
out['origin_date'] = str(dfv.index[2])
out['target_end_dates'] = [str(d) for d in dfv['target_end_date'].to_numpy()[1:6]]
for loc in ('BE', 'SK'):
    loc_, df1, E = next(s for s in series(targets['ARI'], dfv) if s[0] == loc)
    lags = gp.makelags(df1, y='X', time='Time', E=E, tau=1, vtimestep=True)
    fit = gp.fitGP(pd.concat([df1, lags], axis=1), y='X', x=list(lags.columns), time='Time', trace=True)
    fore = gp.makelags(df1, y='X', time='Time', E=E, tau=1, vtimestep=True, forecast=True, Tdiff_fore=[1, 2, 3, 4])
    pred = gp.predict(fit, newdata=fore).outsampresults
    fit.trace.to_csv(os.path.join(HERE, f'trace_ARI_{loc}.csv'), index=False)

    Y, X = fit.inputs['Y'], fit.inputs['X']
    N, d = X.shape
    sc = fit.scaling
    td1 = lags['Tdiff_1']
    M = int(td1.notna().sum())
    phi_td1 = float(fit.pars[f'phi{E + 1}'])
    s_td1 = float(sc['xsds'][E])
    # shrink factor of section 6: kernel between a forecast row at horizon h and a training row with gap 1
    kappa = [float(np.exp(-phi_td1 * (h - 1) ** 2 / s_td1 ** 2)) for h in (1, 2, 3, 4)]
    mu1_std = (pred['predmean'][0] - sc['ymeans']) / sc['ysds']
    sigma2, ve = float(fit.pars['sigma2']), float(fit.pars['ve'])

    # leave-one-out: the loop of predict() against the closed form  Y_i - a_i / [Sigma^-1]_ii
    iK = fit.covm['iKVs']
    a = iK @ Y

    # section 6: the forecast at horizon h rebuilt from the week-1 kernel, row by row.
    # c_j(h) = c_j(1) * exp(-phi_gap * ((h - gap_j)^2 - (1 - gap_j)^2) / s^2), gap_j = Tdiff_1 of training row j
    phi = fit.pars.to_numpy()[:d]
    xnew = (fore[list(lags.columns)].to_numpy(dtype=float) - sc['xmeans']) / sc['xsds']
    _, Cs = gp.getcov(phi, sigma2, 0.5, X, xnew, fit.inputs['Pop'], np.ones(4, dtype=int))
    gap = td1.to_numpy()[fit.inputs['completerows']]
    terms1 = Cs[0] * a                       # contribution of each training row to the week-1 mean
    rebuilt, by_gap = [], {}
    for h in (1, 2, 3, 4):
        factor = np.exp(-phi_td1 * ((h - gap) ** 2 - (1 - gap) ** 2) / s_td1 ** 2)
        rebuilt.append(float(sc['ymeans'] + sc['ysds'] * np.sum(factor * terms1)))
    for g in sorted(set(gap)):
        by_gap[str(int(g))] = dict(rows=int(np.sum(gap == g)), sum_terms=float(np.sum(terms1[gap == g])),
                                   factors=[float(np.exp(-phi_td1 * ((h - g) ** 2 - (1 - g) ** 2) / s_td1 ** 2))
                                            for h in (1, 2, 3, 4)])
    prior_sd = float(sc['ysds'] * np.sqrt(sigma2 + ve))
    loo = gp.predict(fit, predictmethod='loo').outsampresults
    cr = fit.inputs['completerows']
    loo_mean_std = (loo['predmean'].to_numpy()[cr] - sc['ymeans']) / sc['ysds']
    loo_sd_std = loo['predsd'].to_numpy()[cr] / sc['ysds']
    closed_mean = Y - a / np.diag(iK)
    closed_sd = np.sqrt(1 / np.diag(iK))
    lnL_from_loo = float(np.sum(-0.5 * np.log(loo_sd_std ** 2) - 0.5 * (Y - loo_mean_std) ** 2 / loo_sd_std ** 2))

    # effective prior on phi: the coded gradient is a positive rescaling of the gradient of F_tilde
    D = [gp.sqdist(X[:, i], X[:, i]) for i in range(d)]
    fixedp = np.full(d + 2, np.nan)
    p = fit.parst.to_numpy()

    def F_tilde(q):
        phi = np.exp(q[:d])
        lam = np.pi / 2
        return gp.getlikegrad(q, Y, X, fit.inputs['Pop'], 1, fixedp, None, D, True)['nllpost'] + 0.5 * np.sum(phi ** 2) / lam

    coded = gp.getlikegrad(p, Y, X, fit.inputs['Pop'], 1, fixedp, None, D, True)['grad']
    h_ = 1e-4
    num = np.zeros(d + 2)
    for i in range(d + 2):
        e = np.zeros(d + 3)
        e[i] = h_
        num[i] = (F_tilde(p + e) - F_tilde(p - e)) / (2 * h_)
    big = np.abs(num) > 1e-3
    ratio = coded[:d + 2][big] / num[big]

    n_obs = 30
    xobs = df1['X'].to_numpy()
    out[f'ARI_{loc}'] = dict(
        n=int(len(df1)), gaps=int(df1['X'].isna().sum()), E=int(E), d=int(d), N=int(N),
        iterations=int(fit.iter), F=float(fit.nllpost), gradnorm=float(np.sqrt(fit.grad.to_numpy() @ fit.grad.to_numpy())),
        pars={k: float(v) for k, v in fit.pars.items()}, columns=list(lags.columns),
        ymean=float(sc['ymeans']), ysd=float(sc['ysds']),
        M_Tdiff1=M, Tdiff1_counts={str(k): int(v) for k, v in td1.value_counts().items()},
        Tdiff1_mean=float(sc['xmeans'][E]), Tdiff1_sd=s_td1, one_over_sqrt_M=float(1 / np.sqrt(M)),
        z_h=[float((h - sc['xmeans'][E]) / s_td1) for h in (1, 2, 3, 4)],
        phi_Tdiff1=phi_td1, kappa=kappa,
        predmean=[float(v) for v in pred['predmean']], predsd=[float(v) for v in pred['predsd']],
        predfsd=[float(v) for v in pred['predfsd']],
        rebuilt_mean=rebuilt, by_gap=by_gap, prior_sd=prior_sd, mu1_std=float(mu1_std),
        abs_terms_sum=float(np.sum(np.abs(terms1))),
        fitstats={k: float(v) for k, v in fit.insampfitstats.items()},
        loo_max_abs_diff_mean=float(np.max(np.abs(loo_mean_std - closed_mean))),
        loo_max_abs_diff_sd=float(np.max(np.abs(loo_sd_std - closed_sd))),
        lnL_LOO_reported=float(fit.insampfitstats['lnL_LOO']), lnL_LOO_from_loop=lnL_from_loo,
        Ftilde_ratio_phi=[float(r) for r in ratio[:int(big[:d].sum())]],
        Ftilde_ratio_other=[float(r) for r in ratio[int(big[:d].sum()):]],
        last_obs=[None if np.isnan(v) else float(v) for v in xobs[-n_obs:]],
        length_scales={k: (float(1 / np.sqrt(2 * v)) if v > 0 else None) for k, v in list(fit.pars.items())[:d]},
    )

# ---- 3. the forecast submitted from round 2026-10-21 (gpedm_direct) on the same two series --
sys.path.insert(0, os.path.join(HERE, '..'))
import contextlib
import io
import gpedm_direct as gd
import respi_edm as res
from scipy.stats import norm

for loc in ('BE', 'SK'):
    d = targets['ARI'][(targets['ARI'].location == loc) & targets['ARI'].value.notna()]
    x, first = res.series_by_date(d)
    with contextlib.redirect_stdout(io.StringIO()):
        E, tau = res.choose_embedding(x)
    fc = gd.forecast(x, E, first)
    z = gd.forward(x, fc.transform)
    c = float(fc.transform.split(':')[1])
    hz = []
    for h, fit in zip(gd.HORIZONS, fc.fits):
        train, fore, xcols = gd.direct_design(z, h, range(E), first, season=True, diff=True)
        phi = [float(fit.pars[f'phi{i + 1}']) for i in range(len(xcols))]
        hz.append(dict(h=h, N=int(len(fit.inputs['Y'])), rows_in_table=int(len(train)), iterations=int(fit.iter),
                       columns=xcols, phi=phi, length_scales=[float(1 / np.sqrt(2 * p)) if p > 0 else None for p in phi],
                       ve=float(fit.pars['ve']), sigma2=float(fit.pars['sigma2']),
                       ymean=float(fit.scaling['ymeans']), ysd=float(fit.scaling['ysds']),
                       forecast_row=[float(v) for v in fore[xcols].to_numpy()[0]]))
    mu = fc.mean - z[-1]                      # forecast change on the log scale
    qi = {q: gd.HUB_QUANTILES.index(q) for q in (0.025, 0.25, 0.5, 0.75, 0.975)}
    out[f'direct_ARI_{loc}'] = dict(
        n=int(len(x)), missing=int(np.isnan(x).sum()), first_week=str(first), E=int(E), tau=int(tau),
        series_mean=float(np.nanmean(x)), offset=c, last_value=float(x[-1]), z_last=float(z[-1]),
        change_mean=[float(v) for v in mu], sd=[float(v) for v in fc.sd],
        median=[float(v) for v in fc.quantiles[:, qi[0.5]]],
        q025=[float(v) for v in fc.quantiles[:, qi[0.025]]], q975=[float(v) for v in fc.quantiles[:, qi[0.975]]],
        q25=[float(v) for v in fc.quantiles[:, qi[0.25]]], q75=[float(v) for v in fc.quantiles[:, qi[0.75]]],
        median_check=[float(np.exp(z[-1] + m) - c) for m in mu],
        horizons=hz, last_obs=[None if np.isnan(v) else float(v) for v in x[-30:]])

json.dump(out, open(os.path.join(HERE, 'numbers.json'), 'w'), indent=1)
for k in ('direct_ARI_BE', 'direct_ARI_SK'):
    o = out[k]
    print(k, {kk: o[kk] for kk in ('n', 'missing', 'first_week', 'E', 'tau', 'series_mean', 'offset', 'last_value', 'z_last')})
    print('   change mean', np.round(o['change_mean'], 4), 'sd', np.round(o['sd'], 4))
    print('   median', np.round(o['median'], 1), 'check', np.round(o['median_check'], 1))
    print('   95%', np.round(o['q025'], 1), np.round(o['q975'], 1))
    for hh in o['horizons']:
        print('   h', hh['h'], 'N', hh['N'], 'of', hh['rows_in_table'], 'iter', hh['iterations'], 've %.3g sigma2 %.3g' % (hh['ve'], hh['sigma2']),
              'ysd %.4g' % hh['ysd'], 'ell', [None if v is None else round(v, 2) for v in hh['length_scales']])
for k in ('ARI_BE', 'ARI_SK'):
    o = out[k]
    print(k, {kk: o[kk] for kk in ('n', 'gaps', 'E', 'N', 'iterations', 'F', 'gradnorm', 'ymean', 'ysd', 'M_Tdiff1',
                                    'Tdiff1_counts', 'Tdiff1_mean', 'Tdiff1_sd', 'one_over_sqrt_M', 'z_h', 'phi_Tdiff1', 'kappa')})
    print('   pars', {a: float('%.4g' % b) for a, b in o['pars'].items()})
    print('   predmean', np.round(o['predmean'], 4), 'rebuilt', np.round(o['rebuilt_mean'], 4))
    print('   predsd  ', np.round(o['predsd'], 2), 'prior sd', round(o['prior_sd'], 2), 'mu1_std', round(o['mu1_std'], 4),
          'sum|terms|', round(o['abs_terms_sum'], 2))
    print('   by gap', o['by_gap'])
    print('   LOO loop vs closed form: mean', o['loo_max_abs_diff_mean'], 'sd', o['loo_max_abs_diff_sd'],
          '| lnL_LOO reported', o['lnL_LOO_reported'], 'from loop', o['lnL_LOO_from_loop'])
    print('   coded grad / grad F_tilde: phi', np.round(o['Ftilde_ratio_phi'], 4), 'others', np.round(o['Ftilde_ratio_other'], 4))
    print('   fitstats', {a: float('%.5g' % b) for a, b in o['fitstats'].items()})
print('toy columns', out['toy_cols'])
for r in out['toy_lags']:
    print('  ', r)
print('toy forecast rows', out['toy_fore'])
