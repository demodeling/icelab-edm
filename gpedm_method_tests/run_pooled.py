"""
Third round: one GP for all countries of a target (GPEDM's multi-population model).

    python3 gpedm_method_tests/run_pooled.py --task 3 --ntasks 43

For each origin date and target, the horizon-h model of variants.direct_design (log scale,
change from the last week as the response, time of year as input, E = 3) is fitted once to
the rows of all series together: pop = country, scaling within country, rho estimated.
The same model fitted to each series alone, on the same 208 weeks, is written next to it
so that the two differ only in the pooling.  Output: fc_pooled/<origin date>.csv.
"""
import argparse
import os
import sys
import time
import warnings

sys.path[:0] = ['.', 'gpedm_method_tests']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import gpedm_py as gp
import run_replay as rr
import variants as va

HERE = os.path.dirname(os.path.abspath(__file__))
E, WEEKS, TRANSFORM = 3, 208, 'log'


def recent(c):
    """The last WEEKS weeks of a series (starting at an observed week) and the date of its first week."""
    x = np.asarray(c['x'], dtype=float)[-WEEKS:]
    lead = int(np.flatnonzero(~np.isnan(x))[0])
    x = x[lead:]
    first = pd.Timestamp(c['first_week']) + pd.Timedelta(weeks=len(c['x']) - len(x))
    return x, first


def pooled(cases, h):
    """(mean, sd) on the log scale for every case, from one multi-population fit."""
    tables, fores, last = [], [], []
    for c in cases:
        x, first = recent(c)
        z = va.forward(x, TRANSFORM)
        train, fore, xcols = va.direct_design(z, h, range(E), first, (), True, True)
        tables.append(train.assign(pop=c['location']))
        fores.append(fore.assign(pop=c['location']))
        last.append(z[-1])
    data, foredata = pd.concat(tables, ignore_index=True), pd.concat(fores, ignore_index=True)
    fit = gp.fitGP(data, y='y', x=xcols, pop='pop', time='Time', scaling='local')
    pred = gp.predict(fit, newdata=foredata).outsampresults
    return pred['predmean'].to_numpy() + np.array(last), pred['predsd'].to_numpy(), fit


def run_origin(od, outdir):
    rows = []
    for target in sorted({c['target'] for c in rr.load_cases(od)}):
        cases = [c for c in rr.load_cases(od) if c['target'] == target]
        res = {}
        t0 = time.time()
        mean, sd, info = np.full((len(cases), 4), np.nan), np.full((len(cases), 4), np.nan), []
        for h in (1, 2, 3, 4):
            mean[:, h - 1], sd[:, h - 1], fit = pooled(cases, h)
            info.append((fit.iter, len(fit.inputs['Y']), float(fit.pars['rho'])))
        res['pooled_log_diff_season'] = (mean, sd, time.time() - t0, info)
        t0 = time.time()
        mean, sd = np.full((len(cases), 4), np.nan), np.full((len(cases), 4), np.nan)
        for i, c in enumerate(cases):
            x, first = recent(c)
            mean[i], sd[i], _ = va.direct_forecast(x, E, first, transform=TRANSFORM, season=True, diff=True)
        res['single_log_diff_season'] = (mean, sd, time.time() - t0, [(np.nan, 0, np.nan)] * 4)
        for name, (mean, sd, secs, info) in res.items():
            for i, c in enumerate(cases):
                for h in range(4):
                    rows.append(dict(origin_date=od, target=target, location=c['location'], horizon=h + 1,
                                     target_end_date=c['teds'][h], variant=name, mean=mean[i, h], sd=sd[i, h],
                                     transform=TRANSFORM, E=E, iter=info[h][0], nrows=info[h][1], rho=info[h][2],
                                     seconds=round(secs, 1), truth=c['truth'][h], error=''))
        print(f'{od} {target}: {len(cases)} series, pooled {res["pooled_log_diff_season"][2]:.0f} s, '
              f'iter/rows/rho {res["pooled_log_diff_season"][3]}', flush=True)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, od + '.csv')
    pd.DataFrame(rows).to_csv(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--od', default=None)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--every', type=int, default=2, help='use every n-th origin date')
    ap.add_argument('--out', default=os.path.join(HERE, 'fc_pooled'))
    a = ap.parse_args()
    ods = a.od.split(',') if a.od else rr.origin_dates()[::a.every][a.task::a.ntasks]
    for od in ods:
        if os.path.exists(os.path.join(a.out, od + '.csv')) or not rr.load_cases(od):
            continue
        run_origin(od, a.out)
