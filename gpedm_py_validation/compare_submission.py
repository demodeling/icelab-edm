"""
Run the whole forecast (respi_edm.run_analysis) with backend="r" and with
backend="python" on the same inputs, and compare both submission files with the
one of the last weekly run.

Run on HPC2N from ~/git/Respi, with the modules of the weekly run loaded:
    python3 gpedm_py_validation/compare_submission.py
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd

from hub_inputs import HERE, load_inputs
import respi_edm as res

dfv, targets = load_inputs()
current_date = dfv.index[2]
last_run = f'{current_date}-IceLab-EDM.csv'

files = {'last run': last_run}
for backend in ('r', 'python'):
    filename = os.path.join(HERE, f'{current_date}-IceLab-EDM.{backend}.csv')
    if os.path.exists(filename):
        os.remove(filename)   # run_analysis appends
    for df in targets.values():
        with contextlib.redirect_stdout(io.StringIO()):
            res.run_analysis(filename, df, np.unique(df['location']), dfv, origin_date=current_date,
                             target_incidence=df['target'].to_numpy()[0], backend=backend)
    files[backend] = filename

keys = ['origin_date', 'target', 'target_end_date', 'horizon', 'location', 'output_type', 'output_type_id']
tables = {k: pd.read_csv(f) for k, f in files.items()}
for k, t in tables.items():
    print(f'{k:<9} {len(t)} rows, {t.groupby("target").location.nunique().to_dict()}  ({files[k]})')

print()
for a, b in (('last run', 'r'), ('last run', 'python'), ('r', 'python')):
    ta, tb = tables[a], tables[b]
    same_rows = len(ta) == len(tb) and ta[keys].equals(tb[keys])
    line = f'{a} vs {b}: same rows in the same order: {same_rows}'
    if same_rows:
        diff = np.abs(ta['value'] - tb['value'])
        rel = diff / np.maximum(np.abs(ta['value']), 1e-300)
        line += (f'; values identical: {bool((diff == 0).all())}; max abs diff {diff.max():.3g}; '
                 f'max rel diff {rel[ta["value"] != 0].max():.3g}')
    print(line)
