"""
Check the gradient that drives the optimiser against finite differences of the
objective, for a few hub series, at the starting point and at the fitted point.

The objective is -(log likelihood + log prior).  The prior and its gradient are
simple closed forms, so the script isolates the likelihood: per parameter it
reports
    (likelihood part of the coded gradient) / (finite difference of the likelihood).
A ratio of 1 means the coded gradient is the derivative of the objective.
Parameters whose likelihood gradient is below 1e-3 (lags the fit has switched
off, and time-gap columns that hardly vary) are left out: the finite difference
of an objective of size 1e3 is only good to about 1e-6.

Run on HPC2N from ~/git/Respi, with the modules of the weekly run loaded:
    python3 gpedm_py_validation/check_gradient.py
"""
import numpy as np
import pandas as pd

from hub_inputs import load_inputs, series
import gpedm_py as gp


def natural(parst, d):
    phi = np.exp(parst[:d])
    ve = (gp.VEMAX - gp.VEMIN) / (1 + np.exp(-parst[d])) + gp.VEMIN
    sigma2 = (gp.SIGMA2MAX - gp.SIGMA2MIN) / (1 + np.exp(-parst[d + 1])) + gp.SIGMA2MIN
    return phi, ve, sigma2


def ratios(parst, Y, X, Pop):
    d = X.shape[1]
    fixed = np.full(d + 2, np.nan)
    D = [gp.sqdist(X[:, i], X[:, i]) for i in range(d)]

    def negloglik(p):   # objective with the prior taken out
        lp, _ = gp.getpriors(*natural(p, d), 0.5, 1, gp.SIGMA2MAX, gp.VEMAX)
        return gp.getlikegrad(p, Y, X, Pop, 1, fixed, None, D, True)["nllpost"] + lp

    h = 1e-4
    numeric = np.zeros(d + 2)
    for i in range(d + 2):
        e = np.zeros(d + 3)
        e[i] = h
        numeric[i] = (negloglik(parst + e) - negloglik(parst - e)) / (2 * h)

    phi, ve, sigma2 = natural(parst, d)
    _, dlp = gp.getpriors(phi, ve, sigma2, 0.5, 1, gp.SIGMA2MAX, gp.VEMAX)
    dpars = np.concatenate([phi, [
        (ve - gp.VEMIN) * (1 - (ve - gp.VEMIN) / (gp.VEMAX - gp.VEMIN)),
        (sigma2 - gp.SIGMA2MIN) * (1 - (sigma2 - gp.SIGMA2MIN) / (gp.SIGMA2MAX - gp.SIGMA2MIN))]])
    coded = gp.getlikegrad(parst, Y, X, Pop, 1, fixed, None, D, True)["grad"][:d + 2] + dlp[:d + 2] * dpars
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(numeric) > 1e-3, coded / numeric, np.nan)


dfv, targets = load_inputs()
rows = []
for target, df in targets.items():
    for k, (loc, df1, E) in enumerate(series(df, dfv)):
        if k >= 6:
            break
        lags = gp.makelags(df1, y="X", time="Time", E=E, tau=1, vtimestep=True)
        fit = gp.fitGP(pd.concat([df1, lags], axis=1), y="X", x=list(lags.columns), time="Time")
        Y, X, Pop = fit.inputs["Y"], fit.inputs["X"], fit.inputs["Pop"]
        d = X.shape[1]
        start = np.concatenate([np.log(np.full(d, 0.1)), [gp.logit(0.001, gp.VEMIN, gp.VEMAX),
                                                           gp.logit(0.999, gp.SIGMA2MIN, gp.SIGMA2MAX), 0]])
        for where, parst in (('start', start), ('fitted', fit.parst.to_numpy())):
            r = ratios(parst, Y, X, Pop)
            rows.append(dict(target=target, location=loc, E=E, point=where,
                             phi_checked=f'{int(np.sum(~np.isnan(r[:d])))} of {d}',
                             phi_ratio_min=np.nanmin(r[:d]), phi_ratio_max=np.nanmax(r[:d]),
                             ve_ratio=r[d], sigma2_ratio=r[d + 1]))
print(pd.DataFrame(rows).round(4).to_string(index=False))
