"""
Fit every series of the current RespiCast round with the R package GPEDM and
with its port gpedm_py, and compare lags, fitted parameters and forecasts.

Run on HPC2N from ~/git/Respi, with the modules of the weekly run loaded:
    python3 gpedm_py_validation/compare_hub_series.py
Writes gpedm_py_validation/hub_series_comparison.csv.
"""
import os
import time

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from hub_inputs import HERE, load_inputs, series
import gpedm_py as gp

pandas2ri.activate()
ro.r('suppressMessages(library(GPEDM))')


def fit_r(df1, E):
    ro.globalenv['mydf1'] = pandas2ri.py2rpy(df1)
    ro.globalenv['E'] = E
    t0 = time.time()
    ro.r('''
    mydf1misslags=makelags(data=mydf1, y="X", time="Time", E=E, tau=1, vtimestep=T)
    HPmissdata=cbind(mydf1,mydf1misslags)
    vtdemo=fitGP(data=HPmissdata, y="X", x=colnames(mydf1misslags), time="Time")
    HPmissfore=makelags(data=mydf1, y="X", time="Time", E=E, tau=1,vtimestep=T,
                        forecast=T, Tdiff_fore=c(1,2,3,4))
    vtpred=predict(vtdemo, newdata = HPmissfore)
    ''')
    return dict(
        seconds=time.time() - t0,
        lags=np.asarray(ro.r('as.matrix(mydf1misslags)')),
        fore=np.asarray(ro.r('as.matrix(HPmissfore)')),
        pars=np.asarray(ro.r('as.numeric(vtdemo$pars)')),
        iter=int(np.asarray(ro.r('vtdemo$iter'))[0]),
        ln_post=float(np.asarray(ro.r('as.numeric(vtdemo$insampfitstats["ln_post"])'))[0]),
        insamp=np.asarray(ro.r('as.matrix(vtdemo$insampresults[, c("predmean", "predfsd", "predsd")])')),
        out=np.asarray(ro.r('as.matrix(vtpred$outsampresults[, c("predmean", "predfsd", "predsd")])')),
    )


def fit_py(df1, E, one_ulp_up=False):
    if one_ulp_up:   # the same data, every value moved to the next representable number
        df1 = df1.assign(X=np.nextafter(df1['X'].to_numpy(), np.inf))
    t0 = time.time()
    lags = gp.makelags(df1, y="X", time="Time", E=E, tau=1, vtimestep=True)
    fit = gp.fitGP(pd.concat([df1, lags], axis=1), y="X", x=list(lags.columns), time="Time")
    fore = gp.makelags(df1, y="X", time="Time", E=E, tau=1, vtimestep=True,
                       forecast=True, Tdiff_fore=[1, 2, 3, 4])
    pred = gp.predict(fit, newdata=fore)
    cols = ["predmean", "predfsd", "predsd"]
    return dict(
        seconds=time.time() - t0,
        lags=lags.to_numpy(), fore=fore.to_numpy(), pars=fit.pars.to_numpy(), iter=fit.iter,
        ln_post=fit.insampfitstats["ln_post"],
        insamp=fit.insampresults[cols].to_numpy(), out=pred.outsampresults[cols].to_numpy(),
        ve=fit.pars["ve"], sigma2=fit.pars["sigma2"], cond=np.linalg.cond(fit.covm["Sigma"]),
    )


def maxrel(a, b):
    """Largest relative difference; NaN must sit in the same places."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    ok = ~np.isnan(a) & (a != b)
    return float(np.max(np.abs(a[ok] - b[ok]) / np.abs(a[ok]))) if ok.any() else 0.0


dfv, targets = load_inputs()
rows = []
for target, df in targets.items():
    for loc, df1, E in series(df, dfv):
        r, p, p_ulp = fit_r(df1, E), fit_py(df1, E), fit_py(df1, E, one_ulp_up=True)
        rows.append(dict(
            target=target, location=loc, n=len(df1), n_missing=int(df1['X'].isna().sum()), E=E,
            lags_identical=np.array_equal(r['lags'], p['lags'], equal_nan=True),
            fore_lags_identical=np.array_equal(r['fore'], p['fore'], equal_nan=True),
            iter_r=r['iter'], iter_py=p['iter'],
            pars_maxrel=maxrel(r['pars'], p['pars']),
            ln_post_r=r['ln_post'], ln_post_absdiff=abs(r['ln_post'] - p['ln_post']),
            insample_predmean_maxrel=maxrel(r['insamp'][:, 0], p['insamp'][:, 0]),
            insample_predfsd_maxrel=maxrel(r['insamp'][:, 1], p['insamp'][:, 1]),
            insample_predsd_maxrel=maxrel(r['insamp'][:, 2], p['insamp'][:, 2]),
            predmean_maxrel=maxrel(r['out'][:, 0], p['out'][:, 0]),
            predsd_maxrel=maxrel(r['out'][:, 2], p['out'][:, 2]),
            predmean_maxabs=float(np.max(np.abs(r['out'][:, 0] - p['out'][:, 0]))),
            # how far the Python forecast itself moves when the data change in the last digit
            predmean_maxrel_1ulp=maxrel(p['out'][:, 0], p_ulp['out'][:, 0]),
            predsd_maxrel_1ulp=maxrel(p['out'][:, 2], p_ulp['out'][:, 2]),
            ve=p['ve'], sigma2=p['sigma2'], cond_Sigma=p['cond'],
            seconds_r=round(r['seconds'], 2), seconds_py=round(p['seconds'], 2),
        ))
        print({k: rows[-1][k] for k in ('target', 'location', 'n', 'E', 'iter_r', 'iter_py',
                                         'pars_maxrel', 'predmean_maxrel', 'predsd_maxrel')}, flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, 'hub_series_comparison.csv'), index=False)
print()
print('series compared:           ', len(res))
print('lags identical:            ', int(res.lags_identical.sum()), 'of', len(res))
print('forecast lags identical:   ', int(res.fore_lags_identical.sum()), 'of', len(res))
print('same number of iterations: ', int((res.iter_r == res.iter_py).sum()), 'of', len(res))
print('fits that hit 200 iterations:', int((res.iter_r == 200).sum()))
for c in ('pars_maxrel', 'ln_post_absdiff', 'insample_predmean_maxrel', 'insample_predfsd_maxrel',
          'insample_predsd_maxrel', 'predmean_maxrel', 'predsd_maxrel',
          'predmean_maxrel_1ulp', 'predsd_maxrel_1ulp'):
    print(f'max {c:<26}', res[c].max(), ' median', res[c].median())
print()
print('the five series where the forecast sd differs most:')
print(res.sort_values('predsd_maxrel', ascending=False).head(5)[
    ['target', 'location', 'E', 've', 'cond_Sigma', 'predmean_maxrel', 'predsd_maxrel',
     'predmean_maxrel_1ulp', 'predsd_maxrel_1ulp']].to_string(index=False))
print('fit time, R / Python (s):  ', res.seconds_r.sum().round(1), '/', res.seconds_py.sum().round(1))
