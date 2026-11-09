"""
Forecast variants built on gpedm_py, for the replay in run_replay.py.

Every variant returns, for horizons 1 to 4, a Gaussian on a transformed scale:
(mean, sd, transform).  The forecast of the incidence itself is that Gaussian mapped
back through the transform and cut off at zero (replay_scoring.py does this).

Two schemes:
  "vt"      the production model: one GP on variable-step lags, the four horizons
            requested as time gaps 1 to 4 (respi_edm.run_analysis).
  "direct"  one GP per horizon h, trained on rows (values known at week s) -> value at s+h.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd

import gpedm_py as gp
import gpedm_direct as gd
from gpedm_direct import LOG_OFFSET, MIN_ROWS, forward, inverse, season_columns, interpolate, direct_design  # noqa: F401

import neighbours as nb


@contextmanager
def exact_phi_gradient(on=True):
    """GPEDM halves the likelihood gradient of every phi (see gpedm_py.getlikegrad).
    Halving the prior gradient of phi as well makes the whole phi gradient half the exact
    one, and Rprop only uses its sign, so inside this block the optimiser follows the
    exact gradient of the reported objective."""
    if not on:
        yield
        return
    original = gp.getpriors

    def patched(phi, *args):
        lp, dlp = original(phi, *args)
        dlp = dlp.copy()
        dlp[:len(phi)] *= 0.5
        return lp, dlp

    gp.getpriors = patched
    try:
        yield
    finally:
        gp.getpriors = original


# ----------------------------------------------------------------------------
# production scheme
# ----------------------------------------------------------------------------

def vt_forecast(x, E, first_date, transform="none", season=False, exactgrad=False, blank=20):
    """The production model.  blank=20 is the production line df1.iloc[20, 1] = np.nan."""
    z = forward(np.asarray(x, dtype=float), transform)
    df1 = pd.DataFrame({"Time": np.arange(len(z)), "X": z})
    if blank is not None:
        df1.iloc[blank, 1] = np.nan
    lags = gp.makelags(df1, y="X", time="Time", E=E, tau=1, vtimestep=True)
    data = pd.concat([df1, lags], axis=1)
    fore = gp.makelags(df1, y="X", time="Time", E=E, tau=1, vtimestep=True,
                       forecast=True, Tdiff_fore=[1, 2, 3, 4])
    xcols = list(lags.columns)
    if season:
        data["sea_sin"], data["sea_cos"] = season_columns(data["Time"], first_date)
        fore["sea_sin"], fore["sea_cos"] = season_columns(fore["Time"], first_date)
        xcols += ["sea_sin", "sea_cos"]
    with exact_phi_gradient(exactgrad):
        fit = gp.fitGP(data, y="X", x=xcols, time="Time")
    pred = gp.predict(fit, newdata=fore).outsampresults
    return pred["predmean"].to_numpy(), pred["predsd"].to_numpy(), fit


# ----------------------------------------------------------------------------
# direct scheme: the code is the production module gpedm_direct
# ----------------------------------------------------------------------------

def direct_forecast(x, E, first_date, transform="none", season=False, diff=False,
                    seaslags=(), exactgrad=False, extra=None):
    """gpedm_direct.direct_forecast, optionally with the exact phi gradient."""
    with exact_phi_gradient(exactgrad):
        return gd.direct_forecast(x, E, first_date, transform=transform, season=season, diff=diff,
                                  seaslags=seaslags, extra=extra)


# ----------------------------------------------------------------------------
# the variants of the replay
# ----------------------------------------------------------------------------
# name -> (scheme, options).  E "fnn" is the embedding dimension production chooses.

SWEEP_E = (1, 2, 3, 4, 6, 8, 13, 26, 52)

VARIANTS = {
    # production scheme
    "prod":            ("vt", dict()),
    "prod_exactgrad":  ("vt", dict(exactgrad=True)),
    "prod_season":     ("vt", dict(season=True)),
    "prod_log":        ("vt", dict(transform="log")),
    # one GP per horizon
    "direct":                  ("direct", dict()),
    "direct_exactgrad":        ("direct", dict(exactgrad=True)),
    "direct_season":           ("direct", dict(season=True)),
    "direct_log":              ("direct", dict(transform="log")),
    "direct_sqrt":             ("direct", dict(transform="sqrt")),
    "direct_diff":             ("direct", dict(diff=True)),
    "direct_diff_season":      ("direct", dict(diff=True, season=True)),
    "direct_log_diff":         ("direct", dict(transform="log", diff=True)),
    "direct_log_season":       ("direct", dict(transform="log", season=True)),
    "direct_log_diff_season":  ("direct", dict(transform="log", diff=True, season=True)),
    "direct_log_diff_seaslag": ("direct", dict(transform="log", diff=True, seaslags=(-1, 0, 1))),
}
for _e in SWEEP_E:
    VARIANTS[f"direct_E{_e}"] = ("direct", dict(E=_e))
    VARIANTS[f"direct_log_diff_E{_e}"] = ("direct", dict(E=_e, transform="log", diff=True))


# second round, chosen after the first: the two changes that helped most, inside the production scheme
STAGE2 = {
    "prod_log_season": ("vt", dict(transform="log", season=True)),
}


# third round: the offset of the log transform as a share of the series mean (not below 1),
# for series that jump between zero and several hundred
STAGE3 = {
    "direct_log_diff_season_off10": ("direct", dict(transform="log", diff=True, season=True, offset_rel=0.10)),
    "prod_log_season_off10": ("vt", dict(transform="log", season=True, offset_rel=0.10)),
}


# fourth round (14 October 2026), on the model adopted for the weekly run
# (direct_log_diff_season_off10, in gpedm_direct.forecast):
#  - the mix of a short and a very long E with both tuned: single-E members on the log scale
#    with the relative offset, change as response, no time-of-year input;
#  - neighbouring countries as covariates of the adopted model (neighbours.py).
MIX_SHORT = (1, 2, 3, 4, 5, 6, 8)
MIX_LONG = (26, 39, 44, 48, 50, 51, 52, 53, 54, 56, 60, 65, 78)
_ADOPTED = dict(transform="log", diff=True, season=True, offset_rel=0.10)
STAGE4 = {f"ldo_E{e}": ("direct", dict(E=e, transform="log", diff=True, offset_rel=0.10)) for e in MIX_SHORT + MIX_LONG}
STAGE4.update({
    "nb_mean": ("direct", dict(_ADOPTED, covariates="nb_mean")),     # mean of the neighbours, this week and last
    "nb_each": ("direct", dict(_ADOPTED, covariates="nb_each")),     # one input per neighbour
    "all_mean": ("direct", dict(_ADOPTED, covariates="all_mean")),   # mean of all other countries (control)
})

# after the first neighbour results: more targeted inputs, one or two instead of the neighbours' level
STAGE4B = {
    "nb_growth": ("direct", dict(_ADOPTED, covariates="nb_growth")),
    "nb_gap": ("direct", dict(_ADOPTED, covariates="nb_gap")),
    "nb_growth_gap": ("direct", dict(_ADOPTED, covariates="nb_growth_gap")),
}

ALL = {**VARIANTS, **STAGE2, **STAGE3, **STAGE4, **STAGE4B}


def run_variant(name, case, peers=None):
    """(mean[4], sd[4], transform, E used, info) for one replay case.
    peers: all cases of the same origin date (needed by the variants with covariates)."""
    scheme, opt = ALL[name]
    opt = dict(opt)
    E = opt.pop("E", case["E"])
    if "offset_rel" in opt:
        opt["transform"] = gd.log_transform(case["x"], opt.pop("offset_rel"))
    transform = opt.get("transform", "none")
    info = dict(n_nb=np.nan, lnl=[np.nan] * 4, nrows_h=[0] * 4)
    if scheme == "vt":
        mean, sd, fit = vt_forecast(case["x"], E, case["first_week"], **opt)
        info.update(iter=fit.iter, nrows=len(fit.inputs["Y"]))
    else:
        kind = opt.pop("covariates", None)
        if kind is not None:
            opt["extra"], info["n_nb"] = nb.covariates(case, peers, kind)
        mean, sd, fits = direct_forecast(case["x"], E, case["first_week"], **opt)
        ok = [f for f in fits if f is not None]
        info.update(iter=max((f.iter for f in ok), default=np.nan),
                    nrows=min((len(f.inputs["Y"]) for f in ok), default=0),
                    lnl=[np.nan if f is None else float(f.insampfitstats["lnL_LOO"]) for f in fits],
                    nrows_h=[0 if f is None else len(f.inputs["Y"]) for f in fits])
    return mean, sd, transform, E, info
