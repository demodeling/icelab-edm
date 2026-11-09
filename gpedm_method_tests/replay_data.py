"""
Replay inputs: for one hub origin date, the series every location had in the hub
snapshot of the Friday before, built the way respi_edm.run_analysis builds them,
and the values that were later observed for horizons 1 to 4.

Run from ~/git/Respi on HPC2N (the hub clone is gpedm_method_tests/hub).
"""
import contextlib
import io
import os
import sys

sys.path[:0] = ['.', '../edm_playground/src/eDNAcode/']
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.join(HERE, 'hub')
SCORES = os.path.join(HERE, 'latest-forecast_scores.csv')
TARGETS = ('ARI incidence', 'ILI incidence')

_fw = None
_truth = {}


def forecasting_weeks():
    global _fw
    if _fw is None:
        _fw = pd.read_csv(f'{HUB}/supporting-files/forecasting_weeks.csv')
    return _fw


def horizon_dates(od):
    """(week of the last observation available at submission, [target weeks of horizons 1..4])"""
    w = forecasting_weeks()
    w = w[w.origin_date == od].set_index('horizon').target_end_date
    return w[0], [w[h] for h in (1, 2, 3, 4)]


def truth(target):
    """Latest published value per (location, week)."""
    if target not in _truth:
        t = pd.read_csv(f"{HUB}/target-data/latest-{target.split()[0]}_incidence.csv")
        _truth[target] = t.drop_duplicates(['location', 'truth_date']).set_index(['location', 'truth_date']).value
    return _truth[target]


def snapshot(od, target):
    """All locations of one target as published on the Friday before origin date od (None if no snapshot)."""
    sd = (pd.Timestamp(od) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    parts = []
    for src in ('ERVISS', 'FluID'):
        f = f"{HUB}/target-data/{src}/snapshots/{sd}-{target.split()[0]}_incidence.csv"
        if os.path.exists(f):
            parts.append(pd.read_csv(f))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else None


MIN_OBS = 60   # series with fewer observed weeks are not replayed


def _embedding(x):
    """Lag and embedding dimension as respi_edm chooses them (respi_edm.choose_embedding)."""
    import respi_edm as res
    with contextlib.redirect_stdout(io.StringIO()):
        return res.choose_embedding(x)


def date_series(df_sel):
    """The series on a weekly grid by date (respi_edm.series_by_date, the data step of the
    weekly run from round 2026-10-21), with the embedding dimension and lag chosen on it."""
    import respi_edm as res
    x, first_week = res.series_by_date(df_sel)
    return x, *_embedding(x)


def grid_series(df_sel):
    """Copy of the data preparation in respi_edm.run_analysis: the series laid out on a
    year x week grid.  Two things differ from date_series():
    - it raises ValueError if the data contain an ISO week 53;
    - if some week numbers never occur (Hungary reports weeks 40 to 20 only), the weeks
      that do occur are packed to the left of each year's row, so the summer gap moves to
      the turn of the year."""
    import respi_edm as res
    import edm_mp as edm
    with contextlib.redirect_stdout(io.StringIO()):
        weeks, years, uw, uy, row_idx, col_idx, datatmp = res.extract_weekly_data_ari(df_sel, 0, 52, 2027)
        linear_index, row, col, all_years, nweeks = edm.make_linear_index(years, weeks, 0)
    grid = np.full((len(all_years), nweeks), np.nan)
    grid[row_idx, col_idx] = datatmp
    vector = grid.flatten()
    flat_idx = np.ravel_multi_index((row_idx, col_idx), grid.shape)
    E, tau = _embedding(vector)
    return vector[min(flat_idx):max(flat_idx) + 1], E, tau


def cases(od):
    """One dict per series that respi_edm would have forecast on origin date od.
    x, E, tau: the series by date.  x_prod, E_prod: the series as run_analysis builds it
    (None if that step raises)."""
    h0, teds = horizon_dates(od)
    out = []
    for target in TARGETS:
        snap = snapshot(od, target)
        if snap is None:
            continue
        tr = truth(target)
        for loc in np.unique(snap.location):
            d = snap[snap.location == loc]
            if d.truth_date.to_numpy()[0] != h0:       # the production condition: latest week is present
                continue
            if d.value.notna().sum() < MIN_OBS:
                continue
            x, E, tau = date_series(d)
            try:
                x_prod, E_prod, _ = grid_series(d)
            except ValueError:
                x_prod, E_prod = None, None
            out.append(dict(origin_date=od, target=target, location=loc, x=x, E=E, tau=tau,
                            x_prod=x_prod, E_prod=E_prod, first_week=d.truth_date.min(), h0=h0, teds=teds,
                            truth=np.array([tr.get((loc, t), np.nan) for t in teds], dtype=float)))
    return out


def stale_cases(od, kmax=8, kmin=1):
    """Series the weekly run skipped up to round 2026-10-14 on origin date od, because their
    first row was not the newest week: last observation k weeks before the newest week,
    kmin <= k <= kmax.  With kmin = 0 that includes series whose newest week is present but
    whose rows are listed oldest first (UK nations, Swiss ILI).  Same fields as cases(),
    plus k.  The hub's horizons 1..4 are k+1..k+4 weeks after the last observation."""
    h0, teds = horizon_dates(od)
    out = []
    for target in TARGETS:
        snap = snapshot(od, target)
        if snap is None:
            continue
        tr = truth(target)
        for loc in np.unique(snap.location):
            d = snap[snap.location == loc]
            if d.truth_date.to_numpy()[0] == h0:           # forecast by cases()
                continue
            last = d.truth_date.max()
            k = (pd.Timestamp(h0) - pd.Timestamp(last)).days // 7
            if not kmin <= k <= kmax or d.value.notna().sum() < MIN_OBS:
                continue
            x, E, tau = date_series(d)
            out.append(dict(origin_date=od, target=target, location=loc, x=x, E=E, tau=tau, k=int(k),
                            first_week=d.truth_date.min(), h0=h0, teds=teds, last_week=last,
                            truth=np.array([tr.get((loc, t), np.nan) for t in teds], dtype=float)))
    return out
