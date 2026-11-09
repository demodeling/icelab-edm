"""IceLab-EDM for the Swedish influenza forecast hub.

Adapted from ~/git/Respi/respi_edm.py (RespiCast). Embedding dimension E from
false nearest neighbours, then GP-EDM (R package GPEDM, or its Python port
gpedm_py for the 'log' method).

Methods:

- 'direct' (model_version 1.0.0, submitted for the round of 2026-10-04): one GP
  per forecast step k = 1..4 on the raw counts, trained to map the lag vector
  (x_t, ..., x_{t-E+1}) to x_{t+k}. R GPEDM through rpy2.
- 'log' (model_version 2.0.0, the retrospective and the live rounds from
  2026-10-18; chosen on a 6-round backtest on 2026-10-14): Respi's production model,
  ~/git/Respi/gpedm_direct.py (imported, not copied), fitted with its Python GP
  port gpedm_py. One GP per step on log(x + c), c = 10% of the series mean, with
  the time of year of the target week as an extra input and the change from the
  last week as the response. The point forecast is the mean of the predictive
  distribution (a shifted log-normal, cut at zero).
- 'vtimestep': the old Respi method, a single GP with time-difference inputs.
  On a gap-free weekly series those inputs are almost always 1, the GP gives
  them zero weight, and all four horizons come out identical. Never submitted.

Model version: the retrospective and the live rounds run DEFAULT_METHOD under one
version, MODEL_VERSIONS[DEFAULT_METHOD], which metadata/dvirnas-edm.yml must
carry. A later method change gets a new version here and in the metadata
(charter §8); a version that has been submitted is never reused for another
method.

Other differences from Respi:

- The series is built directly from `target_end_date` (a continuous weekly grid),
  instead of the Respi year x week matrix.
- The hub wants point forecasts (`output_type = mean`), so only `predmean` is
  kept. Hub horizon h = GP step h+1, because the data cutoff is
  reference_date - 7 days.
- The last observation is included in the fit (Respi drops it with
  `vector[minIdx:maxIdx]`).
"""

import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(str(Path(__file__).resolve().parent.parent))  # gpedm_direct, gpedm_py, efm_params
import efm_params as efp  # noqa: E402
import gpedm_direct as gd  # noqa: E402

LOCATIONS = ('SE', 'SE-M', 'SE-O')
HORIZONS = (0, 1, 2, 3)
TARGET = 'weekly incident influenza cases'
OUTPUT_COLUMNS = ['reference_date', 'target', 'horizon', 'location',
                  'output_type', 'output_type_id', 'value']


def next_reference_date(today=None):
    """Sunday closing the current submission window (today if Sunday)."""
    today = today or date.today()
    return today + timedelta(days=(6 - today.weekday()) % 7)


def build_live_input(hub_dir, out_path, target_data=None):
    """Long history for live rounds.

    hub/target-data/time-series.csv only starts at 2026W30, so prepend the
    hub's own 2025/26 retrospective history (2015W27-2026W19) and final
    outcomes (2025W40-2026W23). All of it is public Folkhälsodata. For a
    week present in several files, the live file wins. `target_data` replaces
    the live file (a path or file object, e.g. upstream's copy read from git).
    """
    hub_dir = Path(hub_dir)
    retro = hub_dir / 'retrospective-data/2025-2026'
    parts = [pd.read_csv(retro / 'rounds/2026-05-17.csv'),
             pd.read_csv(retro / 'final-outcomes.csv'),
             pd.read_csv(target_data or hub_dir / 'target-data/time-series.csv')]
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(['location', 'target_end_date'], keep='last')
            .sort_values(['target_end_date', 'location']))
    df.to_csv(out_path, index=False)
    return out_path


def load_input(path, reference_date):
    """Hub target data cut at reference_date - 7 days, as the hub template does."""
    df = pd.read_csv(path, parse_dates=['target_end_date'])
    cutoff = pd.Timestamp(reference_date - timedelta(days=7))
    return df[df['target_end_date'] <= cutoff], cutoff


def weekly_series(df, location, cutoff):
    """Continuous weekly series ending at cutoff; suppressed/missing weeks are NaN."""
    d = df[df['location'] == location].copy()
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    d.loc[d['status'] != 'available', 'value'] = np.nan
    s = d.set_index('target_end_date')['value'].sort_index()
    grid = pd.date_range(s.index.min(), cutoff, freq='7D')
    return s.reindex(grid)


def choose_embedding(vector, NN=8, max_dim=8, seed=0):
    """(tau, E) as in Respi. Seeded because the MI estimator is random (charter §11)."""
    np.random.seed(seed)
    _, tau, _ = efp.find_best_tau(vector, NN, 3, mi_threshold=np.nan, mi_fraction=np.nan)
    _, _, E, _ = efp.fnn_from_sparse_embeddings(vector, tau, max_dim=max_dim)
    return int(tau), int(E)


def gpedm_forecast_direct(series, steps=(1, 2, 3, 4)):
    """Direct GP-EDM: a separate GP per step k, x_{t+k} ~ GP(x_t, ..., x_{t-E+1})."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    vector = series.to_numpy(dtype=float)
    vector = vector[np.where(~np.isnan(vector))[0][0]:]
    tau, E = choose_embedding(vector)

    n = len(vector)
    lagcols = [f'X_{j + 1}' for j in range(E)]
    # row t holds x_t, x_{t-1}, ..., x_{t-E+1}
    lags = np.full((n, E), np.nan)
    for j in range(E):
        lags[j:, j] = vector[:n - j]
    if np.isnan(lags[-1]).any():
        raise ValueError(f'last {E} weeks are not all observed; cannot form the lag vector')

    newd = pd.DataFrame([lags[-1]], columns=lagcols)
    newd.insert(0, 'y', np.nan)
    newd.insert(0, 'Time', n - 1)
    conv = ro.default_converter + pandas2ri.converter
    means = []
    for k in steps:
        train = pd.DataFrame(lags[:n - k], columns=lagcols)
        train.insert(0, 'y', vector[k:])
        train.insert(0, 'Time', np.arange(n - k))
        train = train.dropna().reset_index(drop=True)
        with conv.context():
            ro.globalenv['train'] = conv.py2rpy(train)
            ro.globalenv['newd'] = conv.py2rpy(newd)
            ro.globalenv['xcols'] = ro.StrVector(lagcols)
            ro.r('suppressMessages(library(GPEDM))\n'
                 'fitk  <- fitGP(data=train, y="y", x=xcols, time="Time")\n'
                 'predk <- predict(fitk, newdata=newd)')
            pred = conv.rpy2py(ro.r('predk$outsampresults'))
        means.append(float(pred['predmean'].iloc[0]))
    return np.array(means), dict(tau=tau, E=E)


def gpedm_forecast_vtimestep(series, NN=8, max_dim=8):
    """Respi method: one GP with time-difference inputs (flat horizons on weekly data)."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    vector = series.to_numpy(dtype=float)
    obs = np.where(~np.isnan(vector))[0]
    vector = vector[obs[0]:]  # start at the first observation, keep trailing point

    tau, E = choose_embedding(vector, NN, max_dim)  # lags below use tau=1, as in Respi

    df1 = pd.DataFrame({'Time': np.arange(len(vector)), 'X': vector})
    if not df1['X'].isna().any():
        df1.iloc[20, 1] = np.nan  # Respi trick: vtimestep path needs a gap

    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv['mydf1'] = ro.conversion.get_conversion().py2rpy(df1)
        ro.globalenv['E'] = int(E)
        ro.r('''
        suppressMessages(library(GPEDM))
        lags  <- makelags(data=mydf1, y="X", time="Time", E=E, tau=1, vtimestep=T)
        fit   <- fitGP(data=cbind(mydf1, lags), y="X", x=colnames(lags), time="Time")
        fore  <- makelags(data=mydf1, y="X", time="Time", E=E, tau=1, vtimestep=T,
                          forecast=T, Tdiff_fore=c(1,2,3,4))
        pred  <- predict(fit, newdata=fore)
        ''')
        pred = ro.conversion.get_conversion().rpy2py(ro.r('pred$outsampresults'))
    return pred['predmean'].to_numpy(), dict(tau=int(tau), E=int(E))


def cut_lognormal_mean(mean, sd, c):
    """E[max(exp(Z) - c, 0)] for Z ~ N(mean, sd^2): the mean of gpedm_direct's forecast."""
    a = np.log(c)
    return (np.exp(mean + sd ** 2 / 2) * norm.cdf((mean + sd ** 2 - a) / sd)
            - c * norm.cdf((mean - a) / sd))


def gpedm_forecast_log(series):
    """Respi's gpedm_direct.forecast: log scale, offset 10% of the mean, time of year.

    The series is cut to its first and last observations. If the last observation
    is k weeks before the cutoff, steps k+1..k+4 are forecast, as in Respi.
    """
    obs = series.dropna()
    x = series.loc[obs.index[0]:obs.index[-1]].to_numpy(dtype=float)
    k = len(series.loc[obs.index[-1]:]) - 1
    tau, E = choose_embedding(x)
    fc = gd.forecast(x, E, obs.index[0], weeks_missing=k)
    if not fc.ok:
        raise ValueError(f'fewer than {gd.MIN_ROWS} complete training rows')
    mean = cut_lognormal_mean(fc.mean, fc.sd, gd._offset(fc.transform))
    median = fc.quantiles[:, gd.HUB_QUANTILES.index(0.5)]
    return mean, dict(tau=tau, E=E, weeks_missing=k, transform=fc.transform, median=median)


METHODS = {'direct': gpedm_forecast_direct, 'log': gpedm_forecast_log,
           'vtimestep': gpedm_forecast_vtimestep}

# The method behind every file sent to the hub, retrospective and live, and the
# model_version of each method that may be sent ('vtimestep' has none).
DEFAULT_METHOD = 'log'
MODEL_VERSIONS = {'direct': '1.0.0', 'log': '2.0.0'}
METADATA = Path(__file__).resolve().parent.parent / 'metadata' / 'dvirnas-edm.yml'

# Modules from outside this repository (~/git/Respi, edm_playground; neither under
# git) that a version runs, with the md5 of each file when the version was fixed
# (2.0.0: 14 October 2026, the retrospective's run). An edit there that changes the
# forecasts is a method change and gets a new version; one that does not (a comment,
# say) only updates the md5 here.
PINNED_CODE = {'2.0.0': {'gpedm_direct': '56646ea0699db888fb476bea9969eb79',
                         'gpedm_py': '3640280e8c0fd29dc3f09455117c4f15',
                         'efm_params': 'd430bdd6b8fda454e4d1eb5592078fc6'}}


def check_metadata(method=DEFAULT_METHOD, path=METADATA):
    """Raise unless the metadata file gives the model_version of `method` and the
    imported modules pinned for that version are unchanged."""
    if method not in MODEL_VERSIONS:
        raise ValueError(f"method '{method}' has no model version and is not submitted")
    lines = [l for l in Path(path).read_text().splitlines() if l.startswith('model_version:')]
    found = lines[0].split(':', 1)[1].strip().strip('\'"') if lines else None
    if found != MODEL_VERSIONS[method]:
        raise ValueError(f"{path} gives model_version {found}, but method '{method}' is "
                         f"{MODEL_VERSIONS[method]}: set it, and describe the method there (charter §8)")
    for name, md5 in PINNED_CODE.get(found, {}).items():
        f = Path(sys.modules[name].__file__)
        now = hashlib.md5(f.read_bytes()).hexdigest()
        if now != md5:
            raise ValueError(f'{f} has changed since model_version {found} was fixed (md5 {now}, pinned {md5}). '
                             'If the forecasts change, that is a new version (charter §8); if not, update PINNED_CODE')
    return found


def forecast_round(input_path, reference_date, locations=LOCATIONS, method=None):
    """One hub file's rows (plus per-location diagnostics) for reference_date.

    method=None is a submission: DEFAULT_METHOD, after check_metadata(). Name a
    method to compare methods without the check (backtest.py).
    """
    if method is None:
        method = DEFAULT_METHOD
        check_metadata(method)
    df, cutoff = load_input(input_path, reference_date)
    rows, info = [], {}
    for loc in locations:
        s = weekly_series(df, loc, cutoff)
        mean, params = METHODS[method](s)
        info[loc] = dict(params, last_obs_date=s.last_valid_index().date(),
                         last_value=s.dropna().iloc[-1])
        for h in HORIZONS:
            rows.append({'reference_date': reference_date.isoformat(), 'target': TARGET,
                         'horizon': h, 'location': loc, 'output_type': 'mean',
                         'output_type_id': '', 'value': max(float(mean[h]), 0.0)})
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), info
