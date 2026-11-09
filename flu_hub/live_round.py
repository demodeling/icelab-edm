"""One live round, from the hub's upstream target data, without pulling the hub clone.

Usage (from the project root, respi env):
    python models/live_round.py 2026-10-18
Exit status 3: the round's data (through reference_date - 7) is not on upstream/main yet.
Exit status 2: this round's file already exists in submissions/ (not overwritten).

Runs `git -C hub fetch upstream` (remote-tracking refs only; no branch or working
file changes) and reads target-data/time-series.csv from upstream/main. Saves
data/live-input-<date>.csv, runs flu_edm.forecast_round() (DEFAULT_METHOD, after
check_metadata), writes submissions/dvirnas-edm/<date>-dvirnas-edm.csv, adds a copy
to hub/model-output/dvirnas-edm/ and checks it with the hub validator.
"""

import io
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import flu_edm as fe  # noqa: E402

MODEL_ID = 'dvirnas-edm'


def main(reference_date):
    filename = f'{reference_date}-{MODEL_ID}.csv'
    sub = Path('submissions') / MODEL_ID / filename
    if sub.exists():
        print(f'{sub} already exists; not overwritten')
        return 2

    subprocess.run(['git', '-C', 'hub', 'fetch', '--quiet', 'upstream'], check=True)
    commit = subprocess.run(['git', '-C', 'hub', 'log', '-1', '--format=%h %s', 'upstream/main'],
                            check=True, capture_output=True, text=True).stdout.strip()
    live = subprocess.run(['git', '-C', 'hub', 'show', 'upstream/main:target-data/time-series.csv'],
                          check=True, capture_output=True, text=True).stdout
    cutoff = reference_date - timedelta(days=7)
    newest = pd.read_csv(io.StringIO(live), parse_dates=['target_end_date']).target_end_date.max().date()
    print(f'upstream/main {commit}: target data through {newest}; round {reference_date} needs {cutoff}')
    if newest < cutoff:
        return 3

    input_path = fe.build_live_input('hub', f'data/live-input-{reference_date}.csv', io.StringIO(live))
    out, info = fe.forecast_round(input_path, reference_date)
    print('method', fe.DEFAULT_METHOD, 'model_version', fe.MODEL_VERSIONS[fe.DEFAULT_METHOD])
    print(pd.DataFrame(info).T.drop(columns='median', errors='ignore'))
    print(out.pivot(index='location', columns='horizon', values='value').round(1))

    hub_file = Path('hub/model-output') / MODEL_ID / filename
    for p in (sub, hub_file):
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    print('wrote', sub, 'and', hub_file)
    r = subprocess.run([str(Path(sys.executable).parent / 'validate-forecast-submission'),
                        f'model-output/{MODEL_ID}/{filename}'], cwd='hub', capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return 0 if r.returncode == 0 else 1


if __name__ == '__main__':
    sys.exit(main(date.fromisoformat(sys.argv[1])))
