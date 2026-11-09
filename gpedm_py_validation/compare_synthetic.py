"""
Compare gpedm_py with the R package GPEDM on the options the weekly forecast
does not use: several populations, fixed-step lags, the scaling modes, fixed
parameters, and the in-sample prediction methods.  The data are simulated.

Run on HPC2N from ~/git/Respi, with the modules of the weekly run loaded:
    python3 gpedm_py_validation/compare_synthetic.py
"""
import sys

sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import gpedm_py as gp

pandas2ri.activate()
ro.r('suppressMessages(library(GPEDM))')
R_NULL = ro.r('NULL')

failures = []


def report(name, worst, tol):
    ok = worst <= tol
    print(f'{"ok  " if ok else "FAIL"} {name:<58} max rel diff {worst:.2e}', flush=True)
    if not ok:
        failures.append(name)


def maxrel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    ok = ~np.isnan(a)
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-12)))


def numeric(df):
    """Numeric columns of a data frame, as a matrix."""
    df = pd.DataFrame(df)
    return df[[c for c in df.columns if df[c].dtype != object]].to_numpy(dtype=float)


# ---- data: two populations, noisy logistic maps, gaps, unequal lengths ------------
rng = np.random.default_rng(20261012)
frames = []
for name, n, r in (('A', 70, 3.75), ('B', 55, 3.6)):
    x = np.empty(n)
    x[0] = 0.3
    for t in range(1, n):
        x[t] = min(max(r * x[t - 1] * (1 - x[t - 1]) + rng.normal(0, 0.02), 0.01), 0.99)
    z = np.roll(x, 2) + rng.normal(0, 0.05, n)
    x[rng.choice(np.arange(5, n - 3), 6, replace=False)] = np.nan
    frames.append(pd.DataFrame(dict(pop=name, Time=np.arange(1, n + 1), X=x, Z=z)))
dat = pd.concat(frames, ignore_index=True)
one = dat[dat['pop'] == 'A'].drop(columns='pop').reset_index(drop=True)
ro.globalenv['dat'] = pandas2ri.py2rpy(dat)
ro.globalenv['one'] = pandas2ri.py2rpy(one)

# ---- makelags ----------------------------------------------------------------------
lag_cases = [
    ('fixed step, 2 pops, E=3 tau=1',
     'makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1)',
     lambda: gp.makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1)),
    ('fixed step, 2 variables, E=2 tau=2',
     'makelags(dat, y=c("X","Z"), pop="pop", time="Time", E=2, tau=2)',
     lambda: gp.makelags(dat, y=["X", "Z"], pop="pop", time="Time", E=2, tau=2)),
    ('fixed step, forecast, 2 pops, E=2 tau=2',
     'makelags(dat, y="X", pop="pop", time="Time", E=2, tau=2, forecast=T)',
     lambda: gp.makelags(dat, y="X", pop="pop", time="Time", E=2, tau=2, forecast=True)),
    ('fixed step, forecast, 1 pop, no time',
     'makelags(one, y="X", E=3, tau=1, forecast=T)',
     lambda: gp.makelags(one, y="X", E=3, tau=1, forecast=True)),
    ('vtimestep, 2 pops, E=3 tau=1',
     'makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1, vtimestep=T)',
     lambda: gp.makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1, vtimestep=True)),
    ('vtimestep, 2 variables, E=2 tau=2, Tdiff_max=4',
     'makelags(dat, y=c("X","Z"), pop="pop", time="Time", E=2, tau=2, vtimestep=T, Tdiff_max=4)',
     lambda: gp.makelags(dat, y=["X", "Z"], pop="pop", time="Time", E=2, tau=2, vtimestep=True, Tdiff_max=4)),
    ('vtimestep, forecast, Tdiff_fore=1:3',
     'makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1, vtimestep=T, forecast=T, Tdiff_fore=c(1,2,3))',
     lambda: gp.makelags(dat, y="X", pop="pop", time="Time", E=3, tau=1, vtimestep=True, forecast=True,
                         Tdiff_fore=[1, 2, 3])),
    ('vtimestep, forecast, default Tdiff_fore',
     'makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=T, forecast=T)',
     lambda: gp.makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=True, forecast=True)),
    ('vtimestep, 1 pop, no time column',
     'makelags(one, y="X", E=3, tau=1, vtimestep=T)',
     lambda: gp.makelags(one, y="X", E=3, tau=1, vtimestep=True)),
]
print('makelags')
for name, rcode, pyfun in lag_cases:
    r = pandas2ri.rpy2py(ro.r(f'as.data.frame({rcode})'))
    p = pyfun()
    worst = maxrel(numeric(r), numeric(p))
    if list(r.columns) != list(p.columns):
        worst = np.inf
    report(name, worst, 0)

# ---- fitGP and predict -------------------------------------------------------------
ro.r('''
lags2 = makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=T)
d2 = cbind(dat, lags2)
fore2 = makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=T, forecast=T, Tdiff_fore=c(1,2,3))
lags1 = makelags(one, y="X", E=3, tau=1)
d1 = cbind(one, lags1)
fore1 = makelags(one, y="X", time="Time", E=3, tau=1, forecast=T)
d2s = d2
for (v in c("X", colnames(lags2))) d2s[, v] = as.numeric(scale(d2s[, v]))
''')
lags2 = gp.makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=True)
d2 = pd.concat([dat, lags2], axis=1)
fore2 = gp.makelags(dat, y="X", pop="pop", time="Time", E=2, tau=1, vtimestep=True, forecast=True,
                    Tdiff_fore=[1, 2, 3])
lags1 = gp.makelags(one, y="X", E=3, tau=1)
d1 = pd.concat([one, lags1], axis=1)
fore1 = gp.makelags(one, y="X", time="Time", E=3, tau=1, forecast=True)
d2s = d2.copy()
for v in ["X"] + list(lags2.columns):
    d2s[v] = (d2s[v] - d2s[v].mean()) / d2s[v].std()
x2, x1 = list(lags2.columns), list(lags1.columns)

fit_cases = [
    # name, R data, R extra args, Python data, Python kwargs, x columns, forecast frames (R name, Python)
    ('2 pops, global scaling, rho estimated', 'd2', '', d2, {}, x2, ('fore2', fore2)),
    ('2 pops, local scaling', 'd2', ', scaling="local"', d2, dict(scaling="local"), x2, ('fore2', fore2)),
    ('2 pops, no scaling (pre-scaled data)', 'd2s', ', scaling="none"', d2s, dict(scaling="none"), x2, None),
    ('2 pops, rhofixed=0.3', 'd2', ', rhofixed=0.3', d2, dict(rhofixed=0.3), x2, ('fore2', fore2)),
    ('2 pops, phi2 and ve fixed', 'd2', ', fixedpars=c(NA,0.2,NA,NA,0.01,NA)', d2,
     dict(fixedpars=[np.nan, 0.2, np.nan, np.nan, 0.01, np.nan]), x2, ('fore2', fore2)),
    ('2 pops, modeprior=2, own initpars', 'd2', ', modeprior=2, initpars=c(0.5,0.2,0.3,0.1,0.01,0.8,0.3)', d2,
     dict(modeprior=2, initpars=[0.5, 0.2, 0.3, 0.1, 0.01, 0.8, 0.3]), x2, ('fore2', fore2)),
    ('1 pop, fixed-step lags', 'd1', '', d1, {}, x1, ('fore1', fore1)),
]
cols = ["predmean", "predfsd", "predsd"]
TOL = 1e-6
for name, rdata, rargs, pdata, pkwargs, xcols, fore in fit_cases:
    print('\nfitGP:', name)
    haspop = 'pop' in pdata.columns
    popr = ', pop="pop"' if haspop else ''
    xr = 'c(' + ','.join(f'"{c}"' for c in xcols) + ')'
    ro.r(f'fit = fitGP(data={rdata}, y="X", x={xr}, time="Time"{popr}{rargs})')
    fit = gp.fitGP(pdata, y="X", x=xcols, time="Time", pop="pop" if haspop else None, **pkwargs)

    it_r = int(np.asarray(ro.r('fit$iter'))[0])
    print(f'     iterations: R {it_r}, Python {fit.iter}; pars (Python): '
          + ' '.join(f'{k}={v:.4g}' for k, v in fit.pars.items()))
    if it_r != fit.iter:
        failures.append(name + ': iterations')
    report('parameters', maxrel(ro.r('as.numeric(fit$pars)'), fit.pars), TOL)
    report('fit statistics', maxrel(ro.r('as.numeric(fit$insampfitstats)[1:8]'), fit.insampfitstats), TOL)
    report('in-sample results', maxrel(ro.r('as.matrix(fit$insampresults[, c("predmean","predfsd","predsd")])'),
                                       fit.insampresults[cols]), TOL)
    for method, radius in (('loo', 0), ('loo', 1), ('lto', 0), ('lto', 1), ('sequential', 0)):
        r = ro.r(f'as.matrix(predict(fit, predictmethod="{method}", exclradius={radius})'
                 '$outsampresults[, c("predmean","predfsd","predsd")])')
        p = gp.predict(fit, predictmethod=method, exclradius=radius).outsampresults[cols]
        report(f'predict {method}, exclradius={radius}', maxrel(r, p), TOL)
    # the training frame as newdata (has observations), then the forecast rows
    r = ro.r(f'as.matrix(predict(fit, newdata={rdata})$outsampresults[, c("predmean","predfsd","predsd","obs")])')
    p = gp.predict(fit, newdata=pdata)
    report('predict newdata = training frame', maxrel(r, p.outsampresults[cols + ["obs"]]), TOL)
    report('   its R2 and rmse', maxrel(ro.r(f'as.numeric(predict(fit, newdata={rdata})$outsampfitstats)[1:2]'),
                                        p.outsampfitstats), TOL)
    if fore is not None:
        r = ro.r(f'as.matrix(predict(fit, newdata={fore[0]})$outsampresults[, c("predmean","predfsd","predsd")])')
        report('predict newdata = forecast rows', maxrel(r, gp.predict(fit, newdata=fore[1]).outsampresults[cols]), TOL)

print()
print('ALL CASES AGREE' if not failures else f'{len(failures)} FAILED: {failures}')
