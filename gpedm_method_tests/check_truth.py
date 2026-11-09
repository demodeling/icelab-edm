"""Which truth value does the hub score against?  Recompute the hub ensemble's WIS from its
submitted quantiles against (a) the latest value of each week, (b) the value in the first
snapshot that contains the week, and compare with the hub's score file."""
import glob, os, sys
sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np, pandas as pd
import replay_data as rd, replay_scoring as sc

first = {}
for target in rd.TARGETS:
    seen = {}
    for src in ('ERVISS', 'FluID'):
        for f in sorted(glob.glob(f"{rd.HUB}/target-data/{src}/snapshots/*-{target.split()[0]}_incidence.csv")):
            snap = os.path.basename(f)[:10]
            d = pd.read_csv(f)
            for loc, w, v in zip(d.location, d.truth_date, d.value):
                seen.setdefault((loc, w), (snap, v))
    first[target] = seen

s = pd.read_csv(rd.SCORES)
s = s[(s.metric == 'WIS') & (s.team_id == 'respicast') & (s.model_id == 'hubEnsemble')]
rows = []
for f in sorted(glob.glob(f'{rd.HUB}/model-output/respicast-hubEnsemble/*.csv')):
    e = pd.read_csv(f)
    e = e[e.output_type == 'quantile'].copy()
    e['q'] = e.output_type_id.astype(float).round(3)
    w = e.pivot_table(index=['origin_date', 'target', 'location', 'horizon', 'target_end_date'], columns='q', values='value')
    if w.shape[1] != 23:
        continue
    rows.append(w)
w = pd.concat(rows).reset_index()
qcols = [c for c in w.columns if isinstance(c, float)]
w['latest'] = [rd.truth(t).get((l, d), np.nan) for t, l, d in zip(w.target, w.location, w.target_end_date)]
w['first'] = [first[t].get((l, d), (None, np.nan))[1] for t, l, d in zip(w.target, w.location, w.target_end_date)]
w['first_snap'] = [first[t].get((l, d), (None, np.nan))[0] for t, l, d in zip(w.target, w.location, w.target_end_date)]
qv = w[qcols].to_numpy()
w['wis_latest'] = sc.wis(qv, w.latest.to_numpy()); w['wis_first'] = sc.wis(qv, w['first'].to_numpy())
j = w.merge(s[['origin_date', 'target', 'location', 'horizon', 'target_end_date', 'value_absolute']],
            on=['origin_date', 'target', 'location', 'horizon', 'target_end_date'])
print(len(w), 'ensemble forecasts,', len(j), 'with a hub score')
for k in ('latest', 'first'):
    rel = (j['wis_' + k] - j.value_absolute).abs() / j.value_absolute.clip(lower=1e-9)
    print(f'{k:6s}: truth available {j[k].notna().mean():.3f}; hub WIS reproduced to 1e-6: {(rel < 1e-6).mean():.4f}; within 1%: {(rel < 0.01).mean():.4f}')
rl = (j.wis_latest - j.value_absolute).abs() / j.value_absolute.clip(lower=1e-9)
rf = (j.wis_first - j.value_absolute).abs() / j.value_absolute.clip(lower=1e-9)
print('neither reproduces:', int(((rl >= 1e-6) & (rf >= 1e-6)).sum()), ' latest only:', int(((rl < 1e-6) & (rf >= 1e-6)).sum()),
      ' first only:', int(((rl >= 1e-6) & (rf < 1e-6)).sum()), ' both:', int(((rl < 1e-6) & (rf < 1e-6)).sum()))
j['changed'] = (j.latest - j['first']).abs() > 1e-9
print('share of scored weeks whose value was revised after first publication:', round(j.changed.mean(), 3))
bad = j[(rl >= 1e-6) & (rf >= 1e-6)]
print(bad.groupby('origin_date').size().describe().to_string())
print(bad[['origin_date', 'target', 'location', 'horizon', 'target_end_date', 'latest', 'first', 'first_snap', 'wis_latest', 'wis_first', 'value_absolute']].sample(12, random_state=1).to_string())
by = j.assign(ok_latest=rl < 1e-6, ok_first=rf < 1e-6).groupby(j.origin_date.str[:7])[['ok_latest', 'ok_first', 'changed']].mean().round(2)
print(by.to_string())

# weeks whose hub scores were computed on a value that has since been revised: left out of rank scores
flag = j.assign(ok=rl < 1e-6).groupby(['target', 'location', 'target_end_date']).ok.min().reset_index()
flag.to_csv('gpedm_method_tests/truth_consistent.csv', index=False)
print('weeks checked:', len(flag), ' consistent with the hub score file:', int(flag.ok.sum()))
