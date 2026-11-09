"""
Replay: what each variant in variants.py would have forecast on past hub origin dates.

    python3 gpedm_method_tests/run_replay.py --task 3 --ntasks 92      # one share of the origin dates
    python3 gpedm_method_tests/run_replay.py --od 2026-09-09 --variants prod,direct

Writes gpedm_method_tests/fc/<origin date>.<chunk>of<nchunks>.csv, one row per series, variant and horizon:
the Gaussian (mean, sd) on the scale named in `transform`, and the value observed later.
Run from ~/git/Respi on HPC2N.
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

import replay_data as rd
import variants as va

HERE = os.path.dirname(os.path.abspath(__file__))


def origin_dates():
    ods = sorted(rd.forecasting_weeks().origin_date.unique())
    return [od for od in ods if rd.snapshot(od, rd.TARGETS[0]) is not None]


def load_cases(od):
    """The inputs of one origin date, cached in cases/."""
    f = os.path.join(HERE, 'cases', od + '.pkl')
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    cs = rd.cases(od)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f'{f}.{os.getpid()}.{os.uname().nodename}.tmp'   # several jobs may build the same date at once
    with open(tmp, 'wb') as fh:
        pickle.dump(cs, fh)
    os.replace(tmp, f)
    return cs


def special(name, c):
    """Variants that need the production year x week series."""
    xp = c['x_prod']
    if xp is None:
        return None
    if name == 'prod_grid':            # production model on the series as run_analysis builds it
        mean, sd, fit = va.vt_forecast(xp, c['E_prod'], c['first_week'])
    elif name == 'prod_submitted':     # as submitted before 12 October 2026: newest observation dropped
        mean, sd, fit = va.vt_forecast(xp[:-1], c['E_prod'], c['first_week'])
    return mean, sd, 'none', c['E_prod'], dict(iter=fit.iter, nrows=len(fit.inputs['Y']))


def run_origin(od, names, outdir, chunk=0, nchunks=1):
    rows = []
    peers = load_cases(od)
    for c in peers[chunk::nchunks]:
        for name in names:
            t0 = time.time()
            err = ''
            try:
                r = special(name, c) if name in ('prod_grid', 'prod_submitted') else va.run_variant(name, c, peers)
            except Exception as e:           # a failed fit is recorded, not fatal
                r, err = None, f'{type(e).__name__}: {e}'[:120]
            if r is None:
                mean, sd, transform, E, info = np.full(4, np.nan), np.full(4, np.nan), '', np.nan, dict(iter=np.nan, nrows=0)
            else:
                mean, sd, transform, E, info = r
            lnl, nrows_h = info.get('lnl', [np.nan] * 4), info.get('nrows_h', [0] * 4)
            for h in range(4):
                rows.append(dict(origin_date=od, target=c['target'], location=c['location'], horizon=h + 1,
                                 target_end_date=c['teds'][h], variant=name, mean=mean[h], sd=sd[h],
                                 transform=transform, E=E, iter=info['iter'], nrows=info['nrows'],
                                 lnl_loo=lnl[h], nrows_h=nrows_h[h], n_nb=info.get('n_nb', np.nan),
                                 seconds=round(time.time() - t0, 3), truth=c['truth'][h], error=err))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'{od}.{chunk}of{nchunks}.csv')
    pd.DataFrame(rows).to_csv(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--od', default=None, help='comma-separated origin dates')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--variants', default='all')
    ap.add_argument('--out', default=os.path.join(HERE, 'fc'))
    ap.add_argument('--chunk', type=int, default=0, help='run every nchunks-th series of each origin date, starting here')
    ap.add_argument('--nchunks', type=int, default=1)
    ap.add_argument('--skip-done', action='store_true')
    a = ap.parse_args()
    names = {'all': list(va.VARIANTS) + ['prod_grid'], 'stage2': list(va.STAGE2), 'stage3': list(va.STAGE3),
             'stage4': list(va.STAGE4), 'stage4b': list(va.STAGE4B)}.get(a.variants) or a.variants.split(',')
    ods = a.od.split(',') if a.od else origin_dates()[a.task::a.ntasks]
    for od in ods:
        if a.skip_done and os.path.exists(os.path.join(a.out, f'{od}.{a.chunk}of{a.nchunks}.csv')):
            continue
        t0 = time.time()
        n = run_origin(od, names, a.out, a.chunk, a.nchunks)
        print(f'{od}: {n} rows, {time.time() - t0:.0f} s', flush=True)
