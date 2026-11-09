"""
Sixth round: missing weeks.  The adopted model (gpedm_direct.forecast, a missing lag week filled
by linear interpolation) against its variable-step version (vt_direct.py, the last E observed
weeks and the time between them).

    python3 gpedm_method_tests/run_gaps.py real --task 3 --ntasks 92
    python3 gpedm_method_tests/run_gaps.py fake --task 3 --ntasks 92

real  every series-date the weekly run forecasts (newest week present, whichever order the rows
      are in, or one week behind) whose series has a missing week; both models.
fake  the series-dates whose series has no missing week.  Weeks are removed at random, every
      week but the newest one with probability LEVEL ("one": exactly one week, like the old
      model's entry 20), DRAWS times.  Arm "random" is that series; arm "forced" also removes one
      week of the initial state (lag 1 to E-1, at random), so that the forecast row has a gap.
      Both models on each, and the adopted model on the series as it is (level 0, arm "none").
      E stays the one chosen on the complete series.

Output fc_gaps_<mode>/<origin date>.csv: one row per series, model and horizon, with the
Gaussian on the log scale (as in run_replay.py) and init_gappy = some week of the initial state
(weeks s-1 .. s-E+1 of the forecast row) is missing.
"""
import argparse
import os
import sys
import time
import warnings
import zlib

sys.path[:0] = ['.', 'gpedm_method_tests']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import gpedm_direct as gd
import run_replay as rr
import run_stale as rs
import vt_direct as vt

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ('one', 0.02, 0.05, 0.10, 0.20, 0.30)
DRAWS = 3


def init_missing(x, E):
    """Number of missing weeks among s-1 .. s-E+1 of the forecast row (s = last week)."""
    n = len(x)
    return int(np.isnan(x[max(0, n - E):n - 1]).sum())


def standard(x, E, first_week, horizons):
    t = gd.log_transform(x)
    mean, sd, fits = gd.direct_forecast(x, E, first_week, transform=t, season=True, diff=True, horizons=horizons)
    return mean, sd, t, ['ok' if f is not None else 'not_fitted' for f in fits], [0 if f is None else len(f.inputs['Y']) for f in fits]


def both_models(c, x, horizons, models=('standard', 'vt')):
    out = []
    for m in models:
        t0 = time.time()
        try:
            if m == 'standard':
                r = standard(x, c['E'], c['first_week'], horizons)
            else:
                r = vt.forecast(x, c['E'], c['first_week'], horizons=horizons)
            err = ''
        except Exception as e:           # a failed fit is recorded, not fatal
            r, err = (np.full(4, np.nan), np.full(4, np.nan), '', ['error'] * 4, [0] * 4), f'{type(e).__name__}: {e}'[:120]
        out.append((m, r, err, round(time.time() - t0, 3)))
    return out


def rows_of(c, x, results, **extra):
    rows = []
    for m, (mean, sd, transform, status, nrows), err, sec in results:
        for h in range(4):
            rows.append(dict(origin_date=c['origin_date'], target=c['target'], location=c['location'], horizon=h + 1,
                             target_end_date=c['teds'][h], model=m, **extra, E=c['E'], n=len(x),
                             n_missing=int(np.isnan(x).sum()), init_missing=init_missing(x, c['E']),
                             mean=mean[h], sd=sd[h], transform=transform, status=status[h], nrows_h=nrows[h],
                             seconds=sec, truth=c['truth'][h], error=err))
    return rows


def run_real(od):
    rows = []
    sets = (('regular', rr.load_cases(od)), ('order', rs.load_stale(od, order=True)),
            ('one_behind', [c for c in rs.load_stale(od) if c['k'] == 1]))
    for kind, cs in sets:
        for c in cs:
            x = np.asarray(c['x'], dtype=float)
            if not np.isnan(x).any():
                continue
            k = c.get('k', 0)
            rows += rows_of(c, x, both_models(c, x, tuple(k + h for h in gd.HORIZONS)), kind=kind, k=k)
    return rows


def fake_masks(c, n):
    """(level, draw, arm, missing weeks) of every fake series of one case."""
    out = [(0, 0, 'none', np.zeros(n, bool))]
    for level in (0,) + LEVELS:
        for draw in range(DRAWS):
            rng = np.random.default_rng(zlib.crc32(f"{c['origin_date']}|{c['target']}|{c['location']}|{level}|{draw}".encode()))
            miss = np.zeros(n, bool)
            if level == 'one':
                miss[rng.integers(0, n - 1)] = True
            elif level > 0:
                miss[:n - 1] = rng.random(n - 1) < level
            if level != 0:
                out.append((level, draw, 'random', miss))
            forced = miss.copy()
            forced[n - 1 - rng.integers(1, c['E'])] = True
            out.append((level, draw, 'forced', forced))
    return out


def run_fake(od, chunk=0, nchunks=1):
    rows = []
    for c in [c for c in rr.load_cases(od) if not np.isnan(c['x']).any()][chunk::nchunks]:
        x0 = np.asarray(c['x'], dtype=float)
        for level, draw, arm, miss in fake_masks(c, len(x0)):
            x = np.where(miss, np.nan, x0)
            models = ('standard',) if arm == 'none' else ('standard', 'vt')
            rows += rows_of(c, x, both_models(c, x, gd.HORIZONS, models), kind='regular', k=0,
                            level=str(level), draw=draw, arm=arm)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('real', 'fake'))
    ap.add_argument('--od', default=None)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--chunk', type=int, default=0, help='fake: every nchunks-th series of the date, starting here')
    ap.add_argument('--nchunks', type=int, default=1)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, f'fc_gaps_{a.mode}')
    os.makedirs(out, exist_ok=True)
    for od in (a.od.split(',') if a.od else rr.origin_dates()[a.task::a.ntasks]):
        f = os.path.join(out, od + (f'.{a.chunk}of{a.nchunks}' if a.nchunks > 1 else '') + '.csv')
        if os.path.exists(f):
            continue
        t0 = time.time()
        rows = run_real(od) if a.mode == 'real' else run_fake(od, a.chunk, a.nchunks)
        pd.DataFrame(rows).to_csv(f + '.tmp', index=False)
        os.replace(f + '.tmp', f)
        print(f'{od}: {len(rows)} rows, {time.time() - t0:.0f} s', flush=True)
