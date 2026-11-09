"""Backtest IceLab-EDM on 2025/26 retrospective rounds against the hub's final outcomes.

Usage (from the project root, respi env):
    python models/backtest.py 2025-10-05 2025-11-23 ...   # selected rounds
    python models/backtest.py                             # all 33 rounds
Writes data/backtest-forecasts.csv and prints absolute errors by method and horizon.
'log_median' is the median of the 'log' forecast, scored for comparison only.
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import flu_edm as fe  # noqa: E402

RETRO = Path('hub/retrospective-data/2025-2026')
METHODS = ('direct', 'log')


def persistence(input_path, ref):
    df, cutoff = fe.load_input(input_path, ref)
    rows = []
    for loc in fe.LOCATIONS:
        last = fe.weekly_series(df, loc, cutoff).dropna().iloc[-1]
        rows += [{'location': loc, 'horizon': h, 'value': float(last)} for h in fe.HORIZONS]
    return pd.DataFrame(rows)


def main(refs):
    rows = []
    for ref in refs:
        inp = RETRO / f'rounds/{ref}.csv'
        for method in METHODS:
            t0 = time.time()
            out, info = fe.forecast_round(inp, ref, method=method)
            print(f'{ref} {method:9s} {time.time() - t0:5.1f}s', flush=True)
            rows.append(out.assign(method=method)[['location', 'horizon', 'value', 'method']]
                        .assign(reference_date=ref))
            if method == 'log':  # diagnostic: the median of the same forecast
                rows.append(pd.DataFrame([{'location': loc, 'horizon': h, 'value': float(info[loc]['median'][h])}
                                          for loc in fe.LOCATIONS for h in fe.HORIZONS])
                            .assign(method='log_median', reference_date=ref))
        rows.append(persistence(inp, ref).assign(method='persistence', reference_date=ref))
    fc = pd.concat(rows, ignore_index=True)
    fc['target_end_date'] = [r + timedelta(days=7 * h) for r, h in zip(fc.reference_date, fc.horizon)]

    truth = pd.read_csv(RETRO / 'final-outcomes.csv', parse_dates=['target_end_date'])
    truth = truth[['location', 'target_end_date', 'value']].rename(columns={'value': 'observed'})
    truth['target_end_date'] = truth.target_end_date.dt.date
    fc = fc.merge(truth, on=['location', 'target_end_date'], how='left')
    fc['abs_err'] = (fc.value - fc.observed).abs()
    fc.to_csv('data/backtest-forecasts.csv', index=False)

    pd.set_option('display.width', 200)
    print('\nMAE by method x horizon (rows: location)')
    print(fc.pivot_table(index=['location', 'method'], columns='horizon', values='abs_err').round(1))
    print('\nMAE relative to persistence (<1 = better)')
    m = fc.groupby(['location', 'method']).abs_err.mean().unstack()
    print(m.div(m['persistence'], axis=0).round(2))
    print('\nSE forecasts vs observed')
    se = fc[fc.location == 'SE'].pivot_table(index=['reference_date', 'horizon'],
                                             columns='method', values='value').round(0)
    se['observed'] = fc[(fc.location == 'SE') & (fc.method == 'direct')].set_index(
        ['reference_date', 'horizon']).observed
    print(se)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        refs = [date.fromisoformat(a) for a in sys.argv[1:]]
    else:
        refs = [date.fromisoformat(d) for d in pd.read_csv(RETRO / 'manifest.csv').reference_date]
    main(refs)
