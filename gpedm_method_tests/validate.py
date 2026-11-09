"""
Check the replay against what was really submitted and scored.
1. 'prod_submitted' (production model, newest observation dropped, as before 12 October 2026)
   against the medians in the hub's IceLab-EDM files.
2. WIS computed here against the hub's own WIS for those forecasts.
Run after:  run_replay.py --od <dates> --variants prod_submitted --out gpedm_method_tests/fc_validate
"""
import glob
import os
import sys

sys.path[:0] = ['.', 'gpedm_method_tests']
import numpy as np
import pandas as pd

import replay_data as rd
import replay_scoring as sc

fc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('gpedm_method_tests/fc_validate/*.csv'))])
fc = fc[fc.variant == 'prod_submitted'].dropna(subset=['mean'])
rows = []
for od, g in fc.groupby('origin_date'):
    f = f'{rd.HUB}/model-output/IceLab-EDM/{od}-IceLab-EDM.csv'
    if not os.path.exists(f):
        continue
    sub = pd.read_csv(f)
    med = sub[sub.output_type == 'median'][['target', 'location', 'horizon', 'value']]
    m = g.merge(med, on=['target', 'location', 'horizon'])
    m['rel'] = (m['mean'].clip(lower=0) - m.value).abs() / m.value.abs().clip(lower=1)
    rows.append(m)
m = pd.concat(rows)
per = m.groupby(['origin_date', 'target', 'location']).rel.max()
print(f'{len(per)} submitted series-weeks replayed; median relative difference of the medians {per.median():.1e}')
print(f'reproduced to 1e-4: {(per < 1e-4).sum()}; not reproduced: {(per >= 1e-4).sum()}')
print(per[per >= 1e-4].round(4).to_string())

s = pd.read_csv(rd.SCORES)
s = s[(s.metric == 'WIS') & (s.team_id == 'IceLab') & (s.model_id == 'EDM')]
ok = m[m.set_index(['origin_date', 'target', 'location']).index.isin(per[per < 1e-4].index)].copy()
ok['wis'] = sc.wis(sc.quantiles(ok['mean'].to_numpy(), ok['sd'].to_numpy(), 'none'), ok.truth.to_numpy())
j = ok.merge(s[['origin_date', 'target', 'location', 'horizon', 'value_absolute']], on=['origin_date', 'target', 'location', 'horizon'])
j = j.dropna(subset=['truth'])
rel = (j.wis - j.value_absolute).abs() / j.value_absolute.clip(lower=1e-9)
print(f'WIS against the hub score file: {len(j)} forecasts, median relative difference {rel.median():.1e}, '
      f'share within 1%: {(rel < 0.01).mean():.3f}, max {rel.max():.2e}')
print(j.assign(rel=rel).sort_values('rel').tail(5)[['origin_date', 'target', 'location', 'horizon', 'truth', 'wis', 'value_absolute']].to_string())
