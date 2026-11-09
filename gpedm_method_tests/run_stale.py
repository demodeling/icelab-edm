"""
Fifth round: forecasts for the series the weekly run skips because their latest week is missing.

    python3 gpedm_method_tests/run_stale.py --task 3 --ntasks 92
    python3 gpedm_method_tests/run_stale.py --order --task 3 --ntasks 20

--order: instead, the series whose newest week is present but not in their first row (rows listed
oldest first), which the weekly run also skipped up to round 2026-10-14 (k = 0), into fc_order/.

The adopted model (gpedm_direct.forecast) is fitted on the series up to its last observation,
k weeks before the newest week, with horizons k+1..k+4, which are the hub's horizons 1..4.
Output: fc_stale/<origin date>.csv, same columns as run_replay.py plus k.
"""
import argparse
import os
import pickle
import sys
import time
import warnings

sys.path[:0] = ['.', 'gpedm_method_tests']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import gpedm_direct as gd
import replay_data as rd
import run_replay as rr

HERE = os.path.dirname(os.path.abspath(__file__))
KMAX = 8


def load_stale(od, order=False):
    f = os.path.join(HERE, 'cases_order' if order else 'cases_stale', od + '.pkl')
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    cs = rd.stale_cases(od, 0, 0) if order else rd.stale_cases(od, KMAX)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f'{f}.{os.getpid()}.{os.uname().nodename}.tmp'
    with open(tmp, 'wb') as fh:
        pickle.dump(cs, fh)
    os.replace(tmp, f)
    return cs


def run_origin(od, outdir, order=False):
    rows = []
    for c in load_stale(od, order):
        t0 = time.time()
        k = c['k']
        transform = gd.log_transform(c['x'])
        mean, sd, fits = gd.direct_forecast(c['x'], c['E'], c['first_week'], transform=transform, season=True,
                                            diff=True, horizons=tuple(k + h for h in gd.HORIZONS))
        for h in range(4):
            rows.append(dict(origin_date=od, target=c['target'], location=c['location'], horizon=h + 1,
                             target_end_date=c['teds'][h], variant='stale_adopted', mean=mean[h], sd=sd[h],
                             transform=transform, E=c['E'], k=k, last_week=c['last_week'],
                             nrows_h=0 if fits[h] is None else len(fits[h].inputs['Y']),
                             seconds=round(time.time() - t0, 3), truth=c['truth'][h], error=''))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, od + '.csv')
    pd.DataFrame(rows).to_csv(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--od', default=None)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--order', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, 'fc_order' if a.order else 'fc_stale')
    for od in (a.od.split(',') if a.od else rr.origin_dates()[a.task::a.ntasks]):
        if os.path.exists(os.path.join(out, od + '.csv')):
            continue
        t0 = time.time()
        print(f'{od}: {run_origin(od, out, a.order)} rows, {time.time() - t0:.0f} s', flush=True)
