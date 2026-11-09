"""Quantiles and scores for the replay forecasts written by run_replay.py."""
import numpy as np
from scipy.stats import norm

import variants as va

Q = np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99])
ZQ = norm.ppf(Q)


def quantiles(mean, sd, transform):
    """The 23 hub quantiles of a forecast that is Gaussian on the transformed scale, cut at 0.
    mean, sd: arrays of the same shape; transform: one name.  Result: shape + (23,)."""
    z = np.asarray(mean, dtype=float)[..., None] + np.asarray(sd, dtype=float)[..., None] * ZQ
    return np.clip(va.inverse(z, transform), 0, None)


def cdf(x, mean, sd, transform):
    """P(forecast <= x) for x >= 0, same broadcasting as quantiles()."""
    return norm.cdf((va.forward(x, transform) - mean) / sd)


def wis(q, y):
    """Weighted interval score as the hub computes it: mean over the 23 quantiles of twice the pinball loss."""
    y = np.asarray(y, dtype=float)[..., None]
    return np.mean(2 * np.where(y >= q, Q * (y - q), (1 - Q) * (q - y)), axis=-1)


def linear_pool(components, weights=None):
    """Quantiles of an equally weighted mixture.  components: list of (mean, sd, transform),
    each mean and sd a 1-d array (one entry per forecast)."""
    k = len(components)
    w = np.full(k, 1 / k) if weights is None else np.asarray(weights, dtype=float)
    qs = np.stack([quantiles(m, s, t) for m, s, t in components])        # k x n x 23
    lo, hi = qs.min(axis=0), qs.max(axis=0)
    for _ in range(60):                                                   # bisection on the mixture cdf
        mid = 0.5 * (lo + hi)
        F = sum(wi * cdf(mid, m[:, None], s[:, None], t) for wi, (m, s, t) in zip(w, components))
        below = F < Q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def quantile_average(components, weights=None):
    """Average of the components' quantiles (how hub ensembles are usually combined)."""
    k = len(components)
    w = np.full(k, 1 / k) if weights is None else np.asarray(weights, dtype=float)
    return sum(wi * quantiles(m, s, t) for wi, (m, s, t) in zip(w, components))
