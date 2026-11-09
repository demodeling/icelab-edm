"""
gpedm_direct: forecasts of a weekly series 1 to 4 weeks ahead, one GP per horizon.

This is the model respi_edm.run_analysis(method="direct") submits.  The GP itself is
gpedm_py (the port of GPEDM's fitGP and predict); this file builds its inputs.

For horizon h, row s of the training table (s = the week the forecast is made from) holds

    inputs    z_s, z_(s-1), ..., z_(s-E+1)       the last E weeks, on the log scale
              sin and cos of the time of year of the target week s + h
    response  z_(s+h) - z_s                      the change over the next h weeks

with z = log(x + c).  The offset c is a tenth of the series mean, and at least 1, so that a
zero in a series of large values is not an extreme point on the log scale.  The forecast row
is s = the last week of the series.  The GP's Gaussian for the response is shifted by z_s
and mapped back through the log, so the forecast of the incidence is a shifted log-normal,
cut at zero.

Week s itself must be observed.  Earlier weeks that are missing are filled by linear
interpolation between their neighbours, which lie at or before s.

A series whose newest week is missing (k weeks old) is forecast from its last observation with
horizons k+1..k+4, which land on the same target weeks.  respi_edm allows k <= 1.

Chosen on a replay of 85 hub rounds, October 2024 to October 2026
(gpedm_method_tests/, variant direct_log_diff_season_off10).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

import gpedm_py as gp

HUB_QUANTILES = (0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
HORIZONS = (1, 2, 3, 4)
MIN_ROWS = 30             # a fit needs at least this many complete training rows
LOG_OFFSET = 1.0          # offset of the transform named "log"
LOG_OFFSET_REL = 0.10     # offset of the production forecast, as a share of the series mean


# ----------------------------------------------------------------------------
# transform
# ----------------------------------------------------------------------------
# A transform is named by a string: "none", "sqrt", "log" (log(x + 1)) or "log:<c>" (log(x + c)).

def log_transform(x, rel=LOG_OFFSET_REL):
    """Name of the log transform whose offset is `rel` times the mean of x, and at least 1.
    The offset is rounded to 6 significant digits, so the name carries it exactly."""
    return f"log:{max(1.0, rel * np.nanmean(x)):.6g}"


def _offset(transform):
    return float(transform.split(":")[1]) if ":" in transform else LOG_OFFSET


def forward(x, transform):
    if transform.startswith("log"):
        return np.log(x + _offset(transform))
    if transform == "sqrt":
        return np.sqrt(x)
    return np.asarray(x, dtype=float)


def inverse(z, transform):
    if transform.startswith("log"):
        return np.exp(z) - _offset(transform)
    if transform == "sqrt":
        return np.maximum(z, 0) ** 2
    return z


def quantiles(mean, sd, transform, levels=HUB_QUANTILES):
    """Quantiles of a forecast that is Gaussian(mean, sd) on the transformed scale, cut at 0.
    mean, sd: arrays of the same shape.  Result: that shape + (number of levels,)."""
    z = np.asarray(mean, dtype=float)[..., None] + np.asarray(sd, dtype=float)[..., None] * norm.ppf(levels)
    return np.clip(inverse(z, transform), 0, None)


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def season_columns(week_index, first_date):
    """sin and cos of the time of year of each week (week_index counts from first_date)."""
    dates = pd.Timestamp(first_date) + pd.to_timedelta(np.asarray(week_index, dtype=float) * 7, unit="D")
    phase = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25
    return np.sin(phase), np.cos(phase)


def interpolate(z):
    """Fill missing weeks inside the series by linear interpolation (used for lag inputs only)."""
    ok = ~np.isnan(z)
    return np.interp(np.arange(len(z)), np.flatnonzero(ok), z[ok])


def direct_design(z, h, lags, first_date, seaslags=(), season=False, diff=False, extra=None):
    """
    Training table and forecast row of the horizon-h model.

    Row s (the week the forecast is made from) holds
      L<j>   the value j weeks before s, for j in lags (0 = week s itself);
      S<k>   the value k weeks after the week one year before the target, for k in seaslags;
      sea_sin, sea_cos   time of year of the target week s + h (if season);
      any column of `extra` (name -> array of length len(z), the value known at week s);
      y      the value at s + h, or its change from week s (if diff).
    Week s itself must be observed.  Earlier weeks that are missing are interpolated.
    Returns (training rows, forecast row, names of the input columns).
    """
    n = len(z)
    zf = interpolate(z)
    s = np.arange(n)
    cols = {}
    for j in lags:
        v = np.full(n, np.nan)
        if j < n:                                   # a lag longer than the series stays missing
            v[j:] = (z if j == 0 else zf)[:n - j]
        cols[f"L{j}"] = v
    for k in seaslags:
        idx = s + h - 52 + k
        v = np.full(n, np.nan)
        ok = (idx >= 0) & (idx <= s)
        v[ok] = zf[idx[ok]]
        cols[f"S{k}"] = v
    if season:
        cols["sea_sin"], cols["sea_cos"] = season_columns(s + h, first_date)
    for name, v in (extra or {}).items():
        cols[name] = np.asarray(v, dtype=float)
    y = np.full(n, np.nan)
    y[:n - h] = z[h:]
    if diff:
        y = y - z
    table = pd.DataFrame({"Time": s, "y": y, **cols})
    table.loc[np.isnan(z), list(cols)] = np.nan     # week s must be observed
    xcols = list(cols)
    return table.iloc[:n - h], table.iloc[[n - 1]][["Time"] + xcols], xcols


def direct_forecast(x, E, first_date, transform="none", season=False, diff=False,
                    seaslags=(), extra=None, horizons=HORIZONS):
    """
    One GP per horizon on the series x (NaN where a week is missing).

    Returns (mean, sd, fits): the Gaussian of each horizon on the transformed scale, NaN
    where a horizon has fewer than MIN_ROWS complete training rows or the forecast row is
    incomplete, and the gpedm_py fits (None for those horizons).
    """
    z = forward(np.asarray(x, dtype=float), transform)
    mean, sd, fits = np.full(len(horizons), np.nan), np.full(len(horizons), np.nan), []
    for i, h in enumerate(horizons):
        train, fore, xcols = direct_design(z, h, range(E), first_date, seaslags, season, diff, extra)
        if train[["y"] + xcols].notna().all(axis=1).sum() < MIN_ROWS or fore[xcols].isna().any(axis=None):
            fits.append(None)
            continue
        fit = gp.fitGP(train, y="y", x=xcols, time="Time")
        pred = gp.predict(fit, newdata=fore).outsampresults
        mean[i] = pred["predmean"].iloc[0] + (z[-1] if diff else 0.0)
        sd[i] = pred["predsd"].iloc[0]
        fits.append(fit)
    return mean, sd, fits


# ----------------------------------------------------------------------------
# the production forecast
# ----------------------------------------------------------------------------

@dataclass
class Forecast:
    mean: np.ndarray         # per horizon, on the log scale
    sd: np.ndarray
    transform: str           # "log:<offset>"
    quantiles: np.ndarray    # horizons x len(HUB_QUANTILES), incidence scale, cut at 0
    fits: list               # the gpedm_py fit of each horizon (None if it could not be fitted)

    @property
    def ok(self):
        return bool(np.isfinite(self.quantiles).all())


def forecast(x, E, first_date, offset_rel=LOG_OFFSET_REL, weeks_missing=0):
    """The submitted model: log scale with a relative offset, change from the last week as
    the response, time of year as input, embedding dimension E.
    weeks_missing = k: the series ends k weeks before the newest week, so the hub's horizons
    1..4 are k+1..k+4 weeks after its last observation (one GP for each of those)."""
    transform = log_transform(x, offset_rel)
    mean, sd, fits = direct_forecast(x, E, first_date, transform=transform, season=True, diff=True,
                                     horizons=tuple(weeks_missing + h for h in HORIZONS))
    return Forecast(mean, sd, transform, quantiles(mean, sd, transform), fits)
