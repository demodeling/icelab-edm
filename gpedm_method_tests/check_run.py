"""Completeness of the replay: every cached series has a row for every variant and horizon,
and the cached inputs equal a fresh rebuild from the hub snapshots."""
import glob, os, pickle, sys
sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np, pandas as pd
import replay_data as rd, variants as va, run_replay as rr

HERE = os.path.dirname(os.path.abspath(__file__))
ods = rr.origin_dates()
ncase = {od: len(pickle.load(open(f'{HERE}/cases/{od}.pkl', 'rb'))) for od in ods}
print(len(ods), 'origin dates;', sum(ncase.values()), 'series-dates;', sum(v == 0 for v in ncase.values()), 'dates with no series')
for d, names in (('fc', list(va.VARIANTS) + ['prod_grid']), ('fc_stage2', list(va.STAGE2))):
    files = [f for f in glob.glob(f'{HERE}/{d}/*.csv') if os.path.getsize(f) > 2]
    fc = pd.concat([pd.read_csv(f, usecols=['origin_date', 'target', 'location', 'horizon', 'variant', 'mean', 'error', 'seconds', 'iter']) for f in files])
    got = fc.groupby('origin_date').size()
    want = pd.Series({od: n * len(names) * 4 for od, n in ncase.items() if n})
    print(d, 'rows', len(fc), 'expected', int(want.sum()), 'dates complete', int((got.reindex(want.index) == want).sum()), 'of', len(want))
    print('  duplicates:', int(fc.duplicated(['origin_date', 'target', 'location', 'horizon', 'variant']).sum()),
          ' rows without a forecast:', int(fc['mean'].isna().sum()), ' rows with an error:', int(fc.error.notna().sum()))
    print('  missing forecasts by variant:', fc[fc['mean'].isna()].groupby('variant').size().to_dict())
    print('  errors:', fc[fc.error.notna()].groupby(['variant', 'error']).size().to_dict())
    one = fc[fc.horizon == 1]
    print('  CPU hours:', round(one.seconds.sum() / 3600, 1), ' fits at the 200-iteration limit:', one.groupby('variant').iter.apply(lambda s: (s >= 200).mean()).round(2).to_dict())
rng = np.random.default_rng(0)
bad = 0
for od in map(str, rng.choice(ods, 12, replace=False)):
    cached = pickle.load(open(f'{HERE}/cases/{od}.pkl', 'rb'))
    fresh = rd.cases(od)
    same = len(cached) == len(fresh) and all(
        a['location'] == b['location'] and a['E'] == b['E'] and np.array_equal(a['x'], b['x'], equal_nan=True)
        and np.array_equal(a['truth'], b['truth'], equal_nan=True) for a, b in zip(cached, fresh))
    bad += not same
print('cached inputs rebuilt for 12 random origin dates: differing', bad)
