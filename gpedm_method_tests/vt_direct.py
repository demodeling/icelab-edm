"""
Variable-step version of the adopted model (gpedm_direct.forecast), for the gap tests in run_gaps.py.

The adopted model fills a missing week among the lag inputs by linear interpolation.  Here
the lag inputs are the last E observed weeks instead, as in GPEDM's makelags(vtimestep=True),
with the time between them as extra inputs.  For horizon h, row s (week s observed) holds

    inputs    L0 = z_s, L1, ..., L(E-1)        the E-1 observed weeks before s, newest first
              D1, ..., D(E-1)                  weeks between consecutive lags (D1 = s - week of L1)
              sin and cos of the time of year of the target week s + h
    response  z_(s+h) - z_s

with z = log(x + c), c a tenth of the series mean (at least 1), as in the adopted model.
Without any gap every D is 1, the D inputs are constant and the two models coincide.

A D input that is constant over the training rows has zero variance, which the GP's scaling
cannot handle (the reason production masked entry 20 of the old model).  Such an input is
dropped when the forecast row has the same value; when it does not, the training rows hold
no case like the forecast row, and the horizon is not forecast (status "tdiff_unseen").
"""
import numpy as np
import pandas as pd

import gpedm_py as gp
import gpedm_direct as gd


def vt_design(z, h, E, first_date):
    """Training rows and forecast row (s = last week) of the horizon-h model; see the module docstring."""
    n = len(z)
    obs = np.flatnonzero(~np.isnan(z))
    k = np.arange(len(obs))
    lag = np.full((n, E), np.nan)        # value of lag j
    week = np.full((n, E), np.nan)       # the week it comes from
    for j in range(E):
        ok = k >= j
        lag[obs[ok], j] = z[obs[k[ok] - j]]
        week[obs[ok], j] = obs[k[ok] - j]
    cols = {f"L{j}": lag[:, j] for j in range(E)}
    for j in range(1, E):
        cols[f"D{j}"] = week[:, j - 1] - week[:, j]
    s = np.arange(n)
    cols["sea_sin"], cols["sea_cos"] = gd.season_columns(s + h, first_date)
    y = np.full(n, np.nan)
    y[:n - h] = z[h:] - z[:n - h]
    table = pd.DataFrame({"Time": s, "y": y, **cols})
    table.loc[np.isnan(z), list(cols)] = np.nan     # week s must be observed
    xcols = list(cols)
    return table.iloc[:n - h], table.iloc[[n - 1]][["Time"] + xcols], xcols


def forecast(x, E, first_date, offset_rel=gd.LOG_OFFSET_REL, horizons=gd.HORIZONS):
    """Returns (mean, sd, transform, status, nrows): the Gaussian of each horizon on the log
    scale (NaN where not forecast), the transform name, and per horizon a status
    ("ok", "few_rows", "tdiff_unseen") and the number of training rows."""
    x = np.asarray(x, dtype=float)
    transform = gd.log_transform(x, offset_rel)
    z = gd.forward(x, transform)
    mean, sd = np.full(len(horizons), np.nan), np.full(len(horizons), np.nan)
    status, nrows = [], []
    for i, h in enumerate(horizons):
        train, fore, xcols = vt_design(z, h, E, first_date)
        complete = train[["y"] + xcols].notna().all(axis=1)
        keep, st = [], "ok"
        for c in xcols:
            if c.startswith("D"):
                v = train.loc[complete, c].unique()
                if len(v) <= 1:
                    if len(v) == 1 and fore[c].iloc[0] == v[0]:
                        continue                    # constant, and the same at the forecast row
                    st = "tdiff_unseen"
            keep.append(c)
        nrows.append(int(complete.sum()))
        if st == "ok" and (complete.sum() < gd.MIN_ROWS or fore[keep].isna().any(axis=None)):
            st = "few_rows"
        status.append(st)
        if st != "ok":
            continue
        fit = gp.fitGP(train, y="y", x=keep, time="Time")
        pred = gp.predict(fit, newdata=fore[["Time"] + keep]).outsampresults
        mean[i] = pred["predmean"].iloc[0] + z[-1]
        sd[i] = pred["predsd"].iloc[0]
    return mean, sd, transform, status, nrows
