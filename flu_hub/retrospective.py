"""2025/26 retrospective submission: the 33 hub rounds with one method.

Usage (from the project root, respi env):
    python models/retrospective.py                 # flu_edm.DEFAULT_METHOD
    python models/retrospective.py direct          # another method
    python models/retrospective.py --workers 28    # rounds in parallel (cluster node)
Each round uses only its own fixed input file, so the rounds are independent and
may run in parallel. No file is written until all of them succeed (as in the
hub's model_template.py). Files go to submissions/<model_id>/ only; copying them
into hub/model-output/ is a separate step. They are checked with the hub
validator when it is installed (it reads hub/hub-config, writes nothing there).
Prints MAE against the final outcomes next to persistence.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import flu_edm as fe  # noqa: E402
from backtest import RETRO, persistence  # noqa: E402

MODEL_ID = 'dvirnas-edm'


def run_round(row, method):
    ref = date.fromisoformat(row['reference_date'])
    t0 = time.time()
    out, info = fe.forecast_round(RETRO / row['input_file'], ref, method=method)
    print(f"{row['round_number']:2d} {ref} {time.time() - t0:5.1f}s  E =",
          {loc: v['E'] for loc, v in info.items()}, flush=True)
    return ref, out


def main(method, workers=1):
    # the retrospective's model version is the one the live rounds keep (charter §8)
    print('model_version', fe.check_metadata(method), 'for method', method, flush=True)
    rows = pd.read_csv(RETRO / 'manifest.csv').to_dict('records')  # plain dicts pickle to workers
    with ProcessPoolExecutor(workers) as ex:
        outs = dict(ex.map(partial(run_round, method=method), rows))
    pers = [persistence(RETRO / row['input_file'], date.fromisoformat(row['reference_date']))
            .assign(reference_date=row['reference_date']) for row in rows]

    outdir = Path('submissions') / MODEL_ID
    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    for ref, out in outs.items():
        files.append(outdir / f'{ref}-{MODEL_ID}.csv')
        out.to_csv(files[-1], index=False)
    print(f'wrote {len(files)} files to {outdir}/')

    validator = Path(sys.executable).parent / 'validate-forecast-submission'
    if validator.exists() and Path('hub/hub-config').exists():
        r = subprocess.run([str(validator), '--repo-root', 'hub', '--metadata-root', 'metadata',
                            '--require-all-locations', *map(str, files)], capture_output=True, text=True)
        verdicts = [l.split(':')[0] for l in r.stdout.splitlines() if l.split(':')[0] in
                    ('PASS', 'WARN', 'PARTIAL', 'FAIL')]
        print('validator:', pd.Series(verdicts).value_counts().to_dict(), r.stderr.strip())
    else:
        print('validator: not installed here, not run')

    fc = pd.concat([pd.concat(outs.values()).assign(method=method),
                    pd.concat(pers).assign(method='persistence')], ignore_index=True)
    fc['reference_date'] = pd.to_datetime(fc.reference_date)
    fc['target_end_date'] = fc.reference_date + pd.to_timedelta(7 * fc.horizon, unit='D')
    truth = pd.read_csv(RETRO / 'final-outcomes.csv', parse_dates=['target_end_date'])
    fc = fc.merge(truth[['location', 'target_end_date', 'value']].rename(columns={'value': 'observed'}),
                  on=['location', 'target_end_date'], how='left')
    fc['abs_err'] = (fc.value - fc.observed).abs()
    print(f'\nMAE over {len(outs)} rounds (rows: location)')
    print(fc.pivot_table(index=['location', 'method'], columns='horizon', values='abs_err').round(1))
    m = fc.groupby(['location', 'method']).abs_err.mean().unstack()
    print('\nMAE relative to persistence (<1 = better)')
    print(m.div(m['persistence'], axis=0).round(2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('method', nargs='?', default=fe.DEFAULT_METHOD)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    main(a.method, a.workers)
