"""run_analysis(method="direct") on locations with very little data: under 30 observed weeks
a location is skipped with a message, and the other locations are still forecast
(40 weeks leave 30 complete training rows, so that one is forecast).  Run from ~/git/Respi."""
import contextlib
import io
import os
import sys
import warnings

sys.path[:0] = ['.', 'gpedm_method_tests', 'gpedm_py_validation', '../edm_playground/src/eDNAcode/']
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import gpedm_direct as gd
import respi_edm as res
from hub_inputs import load_inputs

dfv, data = load_inputs()
origin = dfv.index[2]
df = data['ARI']
be = df[df.location == 'BE']
parts = [be]
for name, weeks in (('X5', 5), ('X25', 25), ('X40', 40)):          # made-up locations: the last 5, 25, 40 Belgian weeks
    parts.append(be.head(weeks).assign(location=name))
d = pd.concat(parts, ignore_index=True)
out = 'gpedm_method_tests/weekly_trial/short_series_test.csv'
if os.path.exists(out):
    os.remove(out)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res.run_analysis(out, d, np.unique(d['location']), dfv, origin_date=origin, target_incidence='ARI incidence')
msgs = [l for l in buf.getvalue().splitlines() if 'no forecast' in l]
got = sorted(pd.read_csv(out).location.unique())
os.remove(out)
print('messages:', msgs)
print('locations forecast:', got)
# a lag longer than the series gives no forecast instead of an error
mean, sd, fits = gd.direct_forecast(be.value.to_numpy()[::-1][:60], 78, '2025-01-05', transform='log', diff=True)
print('E longer than the series: forecast is NaN:', bool(np.isnan(mean).all()))
print('PASSED' if got == ['BE', 'X40'] and len(msgs) == 2 and np.isnan(mean).all() else 'FAILED')
