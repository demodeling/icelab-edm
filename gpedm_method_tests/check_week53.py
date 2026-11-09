"""What respi_edm's data step does when the hub data reach ISO week 2026-W53 (week ending
2027-01-03): the latest Belgian ARI series, extended with made-up weekly values to that week."""
import sys
sys.path[:0] = ['.', 'gpedm_method_tests', '../edm_playground/src/eDNAcode/']
import numpy as np, pandas as pd
import replay_data as rd
import respi_edm as res, edm_mp as edm

d = pd.read_csv(f'{rd.HUB}/target-data/latest-ARI_incidence.csv')
d = d[d.location == 'BE'].copy()
last = pd.Timestamp(d.truth_date.max())
for end in ('2026-12-27', '2027-01-03', '2027-01-10'):
    extra = pd.date_range(last + pd.Timedelta(weeks=1), end, freq='7D')[::-1]
    iso = extra.isocalendar()
    add = pd.DataFrame({'target': 'ARI incidence', 'location': 'BE', 'truth_date': extra.strftime('%Y-%m-%d'),
                        'year_week': (iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2)).to_numpy(),
                        'value': 1000.0, 'data_source': 'made up'})
    dd = pd.concat([add, d], ignore_index=True)
    try:
        weeks, years, uw, uy, row_idx, col_idx, datatmp = res.extract_weekly_data_ari(dd, 0, 52, 2027)
        li, row, col, all_years, nweeks = edm.make_linear_index(years, weeks, 0)
        grid = np.full((len(all_years), nweeks), np.nan)
        grid[row_idx, col_idx] = datatmp
        print(f'data to {end} ({dd.year_week.iloc[0]}): data step runs, {len(datatmp)} values')
    except Exception as e:
        print(f'data to {end} ({dd.year_week.iloc[0]}): {type(e).__name__}: {e}')
