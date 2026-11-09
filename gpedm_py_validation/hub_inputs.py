"""Shared by the comparison scripts: one snapshot of the hub inputs, and the series built from it."""
import os
import sys
import urllib.request

sys.path[:0] = ['.', '../edm_playground/src/eDNAcode/']
import numpy as np
import pandas as pd
import respi_edm as res
import efm_params as efp
import edm_mp as edm

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'inputs')
HUB = 'https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators/blob/main'
FILES = {
    'weeks': 'supporting-files/forecasting_weeks.csv',
    'ARI': 'target-data/latest-ARI_incidence.csv',
    'ILI': 'target-data/latest-ILI_incidence.csv',
}


def load_inputs():
    """Download the three hub files once into inputs/ and read them from there."""
    os.makedirs(INPUTS, exist_ok=True)
    for path in FILES.values():
        local = os.path.join(INPUTS, os.path.basename(path))
        if not os.path.exists(local):
            urllib.request.urlretrieve(f'{HUB}/{path}?raw=true', local)
    dfv = pd.read_csv(os.path.join(INPUTS, 'forecasting_weeks.csv'), index_col=0)
    ari = pd.read_csv(os.path.join(INPUTS, 'latest-ARI_incidence.csv'))
    ili = pd.read_csv(os.path.join(INPUTS, 'latest-ILI_incidence.csv'))
    return dfv, {'ARI': ari, 'ILI': ili}


def series(df, dfv):
    """Yield (location, df1, E) for every series, built the way respi_edm.run_analysis builds them."""
    NN, maxtau = 8, 0
    for loc in np.unique(df['location']):
        df_sel = df[df['location'] == loc]
        if df_sel['truth_date'].to_numpy()[0] != dfv['target_end_date'].to_numpy()[1]:
            continue
        weeks, years, uw, uy, row_idx, col_idx, datatmp = res.extract_weekly_data_ari(df_sel, 0, 52, 2027)
        linear_index, row, col, all_years, nweeks = edm.make_linear_index(years, weeks, maxtau)
        testData = np.full((len(all_years), nweeks), np.nan)
        testData[row_idx, col_idx] = datatmp
        vector = testData.flatten()
        flat_idx = np.ravel_multi_index((row_idx, col_idx), testData.shape)
        np.random.seed(42)   # find_best_tau jitters its input with the global generator
        MI, tau, emb_len = efp.find_best_tau(vector, NN, 3, mi_threshold=np.nan, mi_fraction=np.nan)
        minIdx, maxIdx = min(flat_idx), max(flat_idx)
        dims, fnn_fracs, best_dim, n_pts = efp.fnn_from_sparse_embeddings(vector, tau, max_dim=8)
        x = vector[minIdx:maxIdx + 1]
        df1 = pd.DataFrame({'Time': np.arange(len(x)), 'X': x})
        df1.iloc[20, 1] = np.nan
        yield loc, df1, best_dim
