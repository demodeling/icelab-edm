"""
Other countries as covariates of a country's forecast (replay round 4).

A covariate is a series aligned with the target country's weeks: the value known at week s.
Each other country's series is put on the log scale (same relative offset as the forecast),
standardised over its own history, and carried forward over at most MAX_STALE missing weeks,
so a value never uses information from after week s.
"""
import numpy as np
import pandas as pd

import gpedm_direct as gd

MAX_STALE = 2        # weeks a neighbour's last value may be carried forward
MIN_COVER = 0.9      # "nb_each": share of the target's observed weeks a neighbour must cover

# Land borders, plus the nearest sea neighbour for islands.  Codes as in the hub.
BORDERS = {
    'AT': ['DE', 'CZ', 'SK', 'HU', 'SI', 'IT', 'CH', 'LI'],
    'BE': ['NL', 'DE', 'LU', 'FR'],
    'BG': ['RO', 'GR'],
    'CH': ['DE', 'FR', 'IT', 'AT', 'LI'],
    'CY': ['GR'],
    'CZ': ['DE', 'PL', 'SK', 'AT'],
    'DE': ['DK', 'PL', 'CZ', 'AT', 'CH', 'FR', 'LU', 'BE', 'NL'],
    'DK': ['DE', 'SE', 'NO'],
    'EE': ['LV', 'FI'],
    'ES': ['PT', 'FR'],
    'FI': ['SE', 'NO', 'EE'],
    'FR': ['BE', 'LU', 'DE', 'CH', 'IT', 'ES'],
    'GB-ENG': ['GB-WLS', 'GB-SCT', 'IE', 'FR'],
    'GB-NIR': ['IE', 'GB-SCT'],
    'GB-SCT': ['GB-ENG', 'GB-NIR'],
    'GB-WLS': ['GB-ENG', 'IE'],
    'GR': ['BG', 'CY', 'IT'],
    'HR': ['SI', 'HU', 'IT'],
    'HU': ['AT', 'SK', 'RO', 'HR', 'SI'],
    'IE': ['GB-NIR', 'GB-ENG', 'GB-WLS', 'GB-SCT'],
    'IS': ['NO', 'DK', 'IE'],
    'IT': ['FR', 'CH', 'AT', 'SI', 'MT', 'HR'],
    'LI': ['CH', 'AT'],
    'LT': ['LV', 'PL'],
    'LU': ['BE', 'DE', 'FR'],
    'LV': ['EE', 'LT'],
    'MT': ['IT'],
    'NL': ['BE', 'DE'],
    'NO': ['SE', 'FI', 'DK'],
    'PL': ['DE', 'CZ', 'SK', 'LT'],
    'PT': ['ES'],
    'RO': ['HU', 'BG'],
    'SE': ['NO', 'FI', 'DK'],
    'SI': ['IT', 'AT', 'HU', 'HR'],
    'SK': ['CZ', 'PL', 'HU', 'AT'],
}


def standardised(peer):
    """A peer's series on the log scale, standardised, with short gaps carried forward."""
    x = np.asarray(peer['x'], dtype=float)
    z = gd.forward(x, gd.log_transform(x))
    u = (z - np.nanmean(z)) / np.nanstd(z)
    out, last, age = np.full(len(u), np.nan), np.nan, 0
    for i, v in enumerate(u):
        if np.isnan(v):
            age += 1
            out[i] = last if age <= MAX_STALE else np.nan
        else:
            last, age = v, 0
            out[i] = v
    return out


def aligned(case, peer):
    """The peer's standardised series on the weeks of the target series (NaN outside its range)."""
    n = len(case['x'])
    shift = (pd.Timestamp(peer['first_week']) - pd.Timestamp(case['first_week'])).days // 7
    u = standardised(peer)
    out = np.full(n, np.nan)
    src = np.arange(n) - shift                      # index in the peer's series of each target week
    ok = (src >= 0) & (src < len(u))
    out[ok] = u[src[ok]]
    return out


def covariates(case, peers, kind):
    """(extra inputs for gpedm_direct.direct_forecast, number of countries used).
    kind: "nb_mean"   mean of the bordering countries, at week s and at week s - 1
          "nb_each"   one input per bordering country that covers the target's history
          "all_mean"  as nb_mean, over all other countries
          "nb_growth" change of the bordering countries' mean over the last week
          "nb_gap"    the bordering countries' mean minus the country's own standardised value
          "nb_growth_gap"  both
    Returns (None, 0) when no country is available: the forecast is then the model without covariates."""
    same = [p for p in peers if p['target'] == case['target'] and p['location'] != case['location']]
    if kind != 'all_mean':
        same = [p for p in same if p['location'] in BORDERS.get(case['location'], [])]
    cols = {p['location']: aligned(case, p) for p in same}
    cols = {k: v for k, v in cols.items() if np.isfinite(v[-1])}       # must be known at the forecast week
    if not cols:
        return None, 0
    if kind == 'nb_each':
        seen = ~np.isnan(np.asarray(case['x'], dtype=float))
        cols = {k: v for k, v in cols.items() if np.isfinite(v[seen]).mean() >= MIN_COVER}
        if not cols:
            return None, 0
        return {f'nb_{k}': v for k, v in sorted(cols.items())}, len(cols)
    with np.errstate(all='ignore'):
        m = np.nanmean(np.vstack(list(cols.values())), axis=0)
    prev = np.concatenate([[np.nan], m[:-1]])
    extra = {}
    if 'growth' in kind:
        extra['nb_growth'] = m - prev
    if 'gap' in kind:
        extra['nb_gap'] = m - standardised(case)
    return (extra or {'nb0': m, 'nb1': prev}), len(cols)
