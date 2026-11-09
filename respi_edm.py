"""
respi_edm: the IceLab-EDM submission to the RespiCast syndromic-indicators hub.

run_analysis() forecasts every location of one target (ARI or ILI incidence) 1 to 4 weeks
ahead and appends the rows in the hub's quantile format to the round's CSV.
method="direct" (every round from 2026-10-21) is gpedm_direct.forecast, fitted with gpedm_py.
method="vt" (rounds up to 2026-10-14) is one GP on variable-step lags, fitted either with the
R package GPEDM through rpy2 (backend="r") or with gpedm_py (backend="python").
The lag tau and embedding dimension E come from efm_params (mutual information, false nearest
neighbours).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
except ImportError: # R is only needed for method="vt", backend="r"
    ro = pandas2ri = None

import efm_params as efp
import gpedm_py as gp
import gpedm_direct as gd

# Required quantiles for RespiCast hub
HUB_QUANTILES = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    0.975, 0.99
]


def make_linear_index(years, weeks, tau=1):
    """
    Position of every (year, week) on a year x week grid with `tau` spare columns per year,
    so that lags never wrap from one year into the next (moved here from the author's
    edm_playground; used by method="vt" only).
    """
    even_years = sorted(set(years))
    year_rank = {y: i for i, y in enumerate(even_years)}
    nweeks = max(weeks) - min(weeks) + 1 + tau
    linear_index = np.array([year_rank[y] * nweeks + (w - min(weeks)) for y, w in zip(years, weeks)])
    row = linear_index // nweeks
    col = linear_index % nweeks
    return linear_index, row, col, even_years, nweeks


def extract_weekly_data_ari(df, stweek=0, stopweek=52, stopyear=2027):
    
    # stweek=0
    # stopweek=53
    # stopyear=2026
    
    parts   = np.array([x.split('-W') for x in df['year_week']], dtype=int)
    yearnum = parts[:, 0]
    weeknum = parts[:, 1]
    weekidx = np.where((weeknum >= stweek) & (weeknum <= stopweek) & (yearnum <= stopyear))[0]
    
    weeks = weeknum[weekidx]
    years = yearnum[weekidx]
    
    datatmp = df['value'].to_numpy()
    
    unique_weeks = np.unique(weeks)
    unique_years = np.unique(years)
    
    year_to_idx = {yr: i for i, yr in enumerate(unique_years)}
    week_to_idx = {wk: i for i, wk in enumerate(unique_weeks)}
    
    row_idx = np.array([year_to_idx[y] for y in years])
    col_idx = np.array([week_to_idx[w] for w in weeks])
    
    
    
    # clrData = rclr(datatmp.T)
    return weeks, years, unique_weeks, unique_years, row_idx, col_idx, datatmp


def gpedm_to_hub_format(
    predmean: np.ndarray,
    predsd: np.ndarray,
    origin_date: str,
    target: str,
    target_end_date: str,
    location: str,
    horizon: int,
    quantiles: list = HUB_QUANTILES,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """
    Convert GP-EDM Gaussian predictive outputs to RespiCast hub submission format.

    Parameters
    ----------
    predmean : array-like
        Posterior predictive mean from GP-EDM (predmean column).
    predsd : array-like
        Full predictive SD from GP-EDM (predsd column, includes process noise ve).
    origin_date : str
        Forecast origin date, e.g. "2026-05-13".
    target : str
        Target name, e.g. "ILI incidence".
    location : str
        ISO country/location code, e.g. "DE".
    horizon : int
        Weeks ahead (1, 2, 3, 4).
    quantiles : list
        Quantile levels to generate. Defaults to full RespiCast set.
    clip_negative : bool
        Clip values at 0 (incidence cannot be negative). Default True.

    Returns
    -------
    pd.DataFrame in RespiCast hub submission format.
    """
    predmean = np.asarray(predmean)
    predsd = np.asarray(predsd)

    # --- Validate ---
    if predmean.shape != predsd.shape:
        raise ValueError("predmean and predsd must have the same shape.")
    if np.any(np.isnan(predmean)) or np.any(np.isnan(predsd)):
        raise ValueError("predmean or predsd contains NaN — check GP-EDM convergence.")
    if np.any(predsd <= 0):
        raise ValueError("predsd must be > 0 for all points.")

    rows = []

    for mu, sigma in zip(predmean, predsd):

        # --- MEDIAN row (mean == median for Gaussian) ---
        rows.append({
            "origin_date":    origin_date,
            "target":         target,
            "target_end_date": target_end_date[0],
            "horizon":        horizon,
            "location":       location,
            "output_type":    "median",
            "output_type_id": None,
            "value":          float(mu),
        })

        # --- QUANTILE rows: mu + Phi^{-1}(q) * sigma ---
        for q in quantiles:
            value = mu + norm.ppf(q) * sigma
            if clip_negative:
                value = max(value, 0.0)
            rows.append({
                "origin_date":    origin_date,
                "target":         target,
                "target_end_date": target_end_date[0],
                "horizon":        horizon,
                "location":       location,
                "output_type":    "quantile",
                "output_type_id": q,
                "value":          float(value),
            })

    df = pd.DataFrame(rows)

    # Clip median too if requested
    if clip_negative:
        df.loc[df["output_type"] == "median", "value"] = \
            df.loc[df["output_type"] == "median", "value"].clip(lower=0)

    return df



def series_by_date(df_sel):
    """
    The series of one location on a weekly grid by date, from its first to its last
    observation (NaN where a week is missing), and the date of its first week.
    """
    d = df_sel.drop_duplicates('truth_date')
    dates = pd.to_datetime(d['truth_date'])
    idx = ((dates - dates.min()).dt.days.to_numpy() // 7).astype(int)
    x = np.full(idx.max() + 1, np.nan)
    x[idx] = d['value'].to_numpy(dtype=float)
    return x, d['truth_date'].min()


def choose_embedding(x):
    """
    Lag (first minimum of the mutual information) and embedding dimension (false nearest
    neighbours at that lag) of a series, chosen with the same two calls as in run_analysis.
    """
    np.random.seed(42)   # find_best_tau jitters its input with the global generator
    MI, tau, emb_len = efp.find_best_tau(x, 8, 3, mi_threshold=np.nan, mi_fraction=np.nan)
    dims, fnn_fracs, best_dim, n_pts = efp.fnn_from_sparse_embeddings(x, tau, max_dim=8)
    return int(best_dim), int(tau)


def quantile_submission(q, origin_date, target, target_end_dates, location, quantiles=HUB_QUANTILES):
    """
    Hub rows of one location from forecast quantiles.

    q : array, one row per horizon (1, 2, ...) and one column per entry of `quantiles`.
    Per horizon: the median row, then one row per quantile, as build_submission writes them.
    """
    quantiles = list(quantiles)
    rows = []
    for horizon, (ted, qh) in enumerate(zip(target_end_dates, q), start=1):
        common = {"origin_date": origin_date, "target": target, "target_end_date": ted,
                  "horizon": horizon, "location": str(location)}
        rows.append({**common, "output_type": "median", "output_type_id": None,
                     "value": float(qh[quantiles.index(0.5)])})
        for level, value in zip(quantiles, qh):
            rows.append({**common, "output_type": "quantile", "output_type_id": level, "value": float(value)})
    return pd.DataFrame(rows)


MAX_WEEKS_MISSING = 1   # method="direct": a location whose last observation is up to this many weeks
                        # older than the newest week is still forecast, with horizons shifted by that many


def run_analysis_direct(filename, df, locs, dfv, origin_date, target_incidence, max_weeks_missing=MAX_WEEKS_MISSING):
    """
    run_analysis with method="direct": for every location whose last observation is at most
    `max_weeks_missing` weeks older than the newest week of the round, the forecast of
    gpedm_direct.forecast, appended to `filename` in hub format.  A location k weeks behind is
    forecast from its last observation with horizons k+1..k+4, which are the hub's horizons 1..4.
    Returns one table per forecast location (the Gaussian of each horizon on the log scale).
    """
    newest = pd.Timestamp(dfv['target_end_date'].to_numpy()[1])
    target_end_dates = dfv['target_end_date'].to_numpy()[2:6]
    df_res = []
    for loc in locs:
        df_sel = df[(df['location'] == loc) & df['value'].notna()]

        # Weeks from the location's last observation to the newest week.  The latest date, not
        # the first row: some sources (UK nations, Swiss ILI) list their rows oldest first.
        last = df_sel['truth_date'].max()
        k = (newest - pd.Timestamp(last)).days // 7
        if k < 0 or k > max_weeks_missing:
            print(f"{loc}: no forecast, last observation {last} is {k} weeks from the newest week")
            continue

        x, first_date = series_by_date(df_sel)
        if np.isfinite(x).sum() < gd.MIN_ROWS:
            print(f"{loc}: no forecast, fewer than {gd.MIN_ROWS} observed weeks")
            continue
        best_dim, tau = choose_embedding(x)
        fc = gd.forecast(x, best_dim, first_date, weeks_missing=k)
        if not fc.ok:
            print(f"{loc}: no forecast, fewer than {gd.MIN_ROWS} complete training rows")
            continue
        if k > 0:
            print(f"{loc}: newest week missing, forecast from {last} with horizons {k + 1} to {k + 4} weeks after it")

        df_res.append(pd.DataFrame({
            "pop": loc, "horizon": list(gd.HORIZONS), "weeks_missing": k, "E": best_dim, "transform": fc.transform,
            "mean": fc.mean, "sd": fc.sd, "median": fc.quantiles[:, gd.HUB_QUANTILES.index(0.5)],
            "target_end_date": target_end_dates,
        }))
        submission = quantile_submission(fc.quantiles, origin_date, target_incidence, target_end_dates, loc)

        # Preview
        print(submission.head(10).to_string(index=False))
        print(f"\nTotal rows: {len(submission)}")
        print(f"  Median rows:   {(submission.output_type == 'median').sum()}")
        print(f"  Quantile rows: {(submission.output_type == 'quantile').sum()}")

        # Save
        submission.to_csv(filename, mode='a', index=False, header=not os.path.exists(filename))
        print(f"\nSaved → {filename}")

    return df_res


def run_analysis(filename,df,locs, dfv,origin_date = "2026-05-13", target_incidence="ARI incidence", backend="r", method="direct",
                 max_weeks_missing=MAX_WEEKS_MISSING):
    # method="direct" (from round 2026-10-21): one GP per horizon on the log scale, see gpedm_direct.py.
    #   Fitted with gpedm_py; backend is not used.  Locations up to max_weeks_missing weeks behind
    #   are forecast too (0 = only those with the newest week).
    # method="vt" (rounds up to 2026-10-14): one GP on variable-step lags, horizons as time gaps 1..4.
    #   backend="r" fits it with the R package GPEDM, backend="python" with its port gpedm_py.
    #   Only locations whose first row is the newest week; max_weeks_missing is not used.
    if method == "direct":
        return run_analysis_direct(filename, df, locs, dfv, origin_date, target_incidence, max_weeks_missing)
    if method != "vt":
        raise ValueError('method must be "direct" or "vt"')
        
    # os.remove(filename)
    
    if backend == "r":
        pandas2ri.activate()

        ro.r('''
        library(GPEDM)
        ''')
    
    n = 4  # 4 forecast horizons, one location
    NN = 8
    maxtau = 0
    df_res = []
    
    for ix in range(len(locs)):
        df_sel = df[df['location']==locs[ix]]
    
        # Only run for the data that has weeks defined
        if df_sel['truth_date'].to_numpy()[0] == dfv['target_end_date'].to_numpy()[1]:
        # seldf
            
            all_weeks = list(range(0,53))
            weeks, years, unique_weeks, unique_years, row_idx, col_idx, datatmp = extract_weekly_data_ari(df_sel,all_weeks[0],all_weeks[-1],2027)
            
            # ix_val = 50
    
            linear_index, row, col, all_years, nweeks = make_linear_index(years,weeks,maxtau)
            
            # np.shape(emb_vec),np.shape(y_data_embedded)
            testData = np.full((len(all_years),nweeks), np.nan) #extra week for nan's
            testData[row_idx, col_idx] = datatmp
            # example
            vector = testData.flatten()
            flat_idx = np.ravel_multi_index((row_idx, col_idx), testData.shape)
            
            MI, tau,emb_len = efp.find_best_tau(vector,NN,3,mi_threshold=np.nan, mi_fraction=np.nan)
            
            minIdx =  min(flat_idx)
            maxIdx = max(flat_idx)
            
            dims, fnn_fracs, best_dim, n_pts = efp.fnn_from_sparse_embeddings(vector, tau, max_dim=8)
        
            # print(maxIdx)
        
        
            # Wrap in DataFrame with a Time index
            backPts = 0
        
        
            df1 = pd.DataFrame({
                'Time': np.arange(len(vector[minIdx:maxIdx+1-backPts])),
                'X':    vector[minIdx:maxIdx+1-backPts]
            })
        
            df1.iloc[20,1] = np.nan # add a nan to have usable method
        
            
            if backend == "python":
                # same three steps as the R block below
                lags = gp.makelags(df1, y="X", time="Time", E=best_dim, tau=1, vtimestep=True)
                vtdemo = gp.fitGP(pd.concat([df1, lags], axis=1), y="X", x=list(lags.columns), time="Time")
                fore = gp.makelags(df1, y="X", time="Time", E=best_dim, tau=1, vtimestep=True,
                                   forecast=True, Tdiff_fore=[1, 2, 3, 4])
                df_pred = gp.predict(vtdemo, newdata=fore).outsampresults
            else:
                df2 = pd.DataFrame({
                    'Time': np.arange(len(datatmp)),
                    'X':    datatmp
                })
            
                # Push to R
                ro.globalenv['mydf1'] = pandas2ri.py2rpy(df1)
                ro.globalenv['mydf2'] = pandas2ri.py2rpy(df2)
            
                ro.globalenv['tau'] =  tau
                ro.globalenv['E'] = best_dim
        
                ro.r('''
                library(GPEDM)
        
            
                mydf1misslags=makelags(data=mydf1, y="X", time="Time", E=E, tau=1, vtimestep=T)
            
                HPmissdata=cbind(mydf1,mydf1misslags)
                vtdemo=fitGP(data=HPmissdata, y="X", x=colnames(mydf1misslags), time="Time")
        
            
                HPmissfore=makelags(data=mydf1, y="X", time="Time", E=E, tau=1,vtimestep=T,  
                                    forecast=T, Tdiff_fore=c(1,2,3,4))
        
                vtpred=predict(vtdemo, newdata = HPmissfore)
        
                ''')
        
                df_pred = pandas2ri.rpy2py(ro.r('vtpred$outsampresults'))
        
            # df_pred
            df_res.append(df_pred)
        
            # Simulate GP-EDM outsampresults (replace with your real output)
            np.random.seed(42)
        
            gp_results = pd.DataFrame({
                "pop":      [locs[ix]] * n,
                "horizon":  [1, 2, 3, 4],
                "predmean":  df_pred['predmean'].to_numpy(),
                "predsd":    df_pred['predsd'].to_numpy(),   # predsd, NOT predfsd
                "target_end_date": dfv['target_end_date'].to_numpy()[2:6],
            })
        
        
            submission = build_submission(
                gp_results   = gp_results,
                origin_date  = origin_date,
                target       = target_incidence,
                location_col = "pop",
                horizon_col  = "horizon",
                # target_end_date =  dfv['target_end_date'].to_numpy()[3:7],
            )
        
            # Preview
            print(submission.head(10).to_string(index=False))
            print(f"\nTotal rows: {len(submission)}")
            print(f"  Median rows:   {(submission.output_type == 'median').sum()}")
            print(f"  Quantile rows: {(submission.output_type == 'quantile').sum()}")
        
            # Save
            submission.to_csv(filename, mode='a', index=False,header=not os.path.exists(filename))
            print(f"\nSaved → {filename}")
    
    return df_res



def build_submission(
    gp_results: pd.DataFrame,
    origin_date: str,
    target: str,
    location_col: str = "pop",
    horizon_col: str = "horizon",
    quantiles: list = HUB_QUANTILES,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """
    Build a full submission DataFrame from a GP-EDM outsampresults table
    that may contain multiple horizons and locations.

    Parameters
    ----------
    gp_results : pd.DataFrame
        Must contain columns: predmean, predsd, and whatever
        location_col and horizon_col point to.
    origin_date : str
        Forecast reference date.
    target : str
        Target string for the hub.
    location_col : str
        Column in gp_results identifying the location/population.
    horizon_col : str
        Column in gp_results identifying the forecast horizon (weeks ahead).
    """
    all_rows = []

    for (loc, hor), group in gp_results.groupby([location_col, horizon_col]):
        valid = group.dropna(subset=["predmean", "predsd","target_end_date"])
        if valid.empty:
            continue

        chunk = gpedm_to_hub_format(
            predmean    = valid["predmean"].values,
            predsd      = valid["predsd"].values,
            origin_date = origin_date,
            target      = target,
            target_end_date      = valid["target_end_date"].values,
            location    = str(loc),
            horizon     = int(hor),
            quantiles   = quantiles,
            clip_negative = clip_negative,
        )
        all_rows.append(chunk)

    if not all_rows:
        raise RuntimeError("No valid predictions found after dropping NaNs.")

    submission = pd.concat(all_rows, ignore_index=True)
    return submission

