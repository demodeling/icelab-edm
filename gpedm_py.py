"""
gpedm_py: NumPy port of the core of the R package GPEDM.

GPEDM (Munch & Rogers, https://github.com/tanyalrogers/GPEDM) fits a Gaussian
process to a delay-embedded time series.  This file ports, function by
function, the part of it that respi_edm.py calls through rpy2: makelags(),
fitGP() and predict().  The source is the installed version 0.0.0.9010
(commit 2e4bb3c), R/GPfunctions.R; every function below names the R function it
follows.  Where the R code does something unexpected the port does the same,
and a NOTE comment marks the spot.  The one deliberate difference: when no row
survives the removal of missing values, fitGP() here raises an error, while R
fits zero rows and its predict() returns NA.

The model
---------
    y_t = f(x_t) + e_t,                    e_t ~ N(0, ve)
    f ~ GP(0, C),   C(x, x') = sigma2 * exp(-sum_k phi_k (x_k - x'_k)^2) * r
    r = 1 within a population, rho between populations

x_t is the row of lagged values made by makelags().  y and every column of x
are standardised before the fit (scaling="global").  phi_1..phi_d, ve, sigma2
(and rho, if there are several populations) maximise the log posterior; the
optimiser is Rprop, started from the same point every time, so a fit is
deterministic.

Not ported: augmentation (makelags(augment=), fitGP(augdata=)), linprior,
rhomatrix, lags generated inside fitGP (its E and tau arguments), returnGPgrad,
the per-population R2 tables, and the fisheries and iterated-forecast functions.

Use, as in respi_edm.py::

    lags = makelags(df, y="X", time="Time", E=E, tau=1, vtimestep=True)
    fit = fitGP(pd.concat([df, lags], axis=1), y="X", x=list(lags.columns), time="Time")
    fore = makelags(df, y="X", time="Time", E=E, tau=1, vtimestep=True,
                    forecast=True, Tdiff_fore=[1, 2, 3, 4])
    pred = predict(fit, newdata=fore).outsampresults
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import lapack

# Bounds of the variance parameters, hard-coded in fitGP() and getlikegrad().
VEMIN, VEMAX = 1e-4, 4.9999
SIGMA2MIN, SIGMA2MAX = 1e-4, 4.9999
RHOMIN, RHOMAX = 1e-4, 1.0


# ----------------------------------------------------------------------------
# Lags
# ----------------------------------------------------------------------------

def makelags(data, y, E, tau, pop=None, time=None, forecast=False, vtimestep=False,
             Tdiff_max=None, Tdiff_fore=None):
    """
    Port of makelags() and makelags_subrt().

    data : DataFrame; y : column name or list of names; pop, time : column names.

    vtimestep=False: lag i of a variable is its value i*tau rows earlier.
    vtimestep=True: missing rows are skipped.  Lag 1 is the most recent row
    without missing values, lag 2 the one before that, and so on, and the
    columns Tdiff_1..Tdiff_E hold the time elapsed between consecutive lags.

    forecast=False returns one row of lags per row of data.  forecast=True
    returns the rows needed to predict beyond the end of the series: with
    vtimestep, one row per entry of Tdiff_fore (time steps ahead of the last
    row); without it, the next tau rows.
    """
    if E != int(E) or tau != int(tau) or E < 0 or tau < 0:
        raise ValueError("E and tau must be positive integers")
    E, tau = int(E), int(tau)

    data = pd.DataFrame(data)
    vnames = [y] if isinstance(y, str) else list(y)
    xy = data[vnames].to_numpy(dtype=float)
    n, nv = xy.shape
    popv = np.ones(n, dtype=int) if pop is None else data[pop].to_numpy()
    up = pd.unique(popv)
    pname = "pop" if pop is None else pop
    tname = "time" if time is None else time
    timev = None if time is None else data[time].to_numpy()

    # columns are grouped by variable: X_1 .. X_E, then the next variable
    lagcols = [f"{v}_{i * tau}" for v in vnames for i in range(1, E + 1)]

    if not vtimestep:
        out = np.full((n, nv * E), np.nan)
        popfore = np.repeat(up, tau)
        outfore = np.full((len(up) * tau, nv * E), np.nan)
        for p in up:
            ind = np.flatnonzero(popv == p)
            indfore = np.flatnonzero(popfore == p)
            if len(ind) <= E * tau:
                raise ValueError(f"time series length <= E*tau for pop {p}")
            for j in range(nv):
                ts = xy[ind, j]
                for i in range(1, E + 1):
                    col = j * E + i - 1
                    out[ind[tau * i:], col] = ts[:len(ts) - tau * i]
                    outfore[indfore, col] = ts[len(ts) - tau * i:len(ts) - tau * (i - 1)]
        if not forecast:
            return pd.DataFrame(out, columns=lagcols)
        res = pd.DataFrame(outfore, columns=lagcols)
        if len(up) > 1:
            res.insert(0, pname, popfore)
        if timev is not None:
            timefore = np.zeros(len(up) * tau)
            for p in up:
                timei = timev[popv == p]
                timefore[popfore == p] = timei[-1] + np.arange(1, tau + 1) * (timei[-1] - timei[-2])
            res.insert(0, tname, timefore)
        return res

    if timev is None:
        timev = np.zeros(n)
        for p in up:
            timev[popv == p] = np.arange(1, np.sum(popv == p) + 1)
    cols = lagcols + [f"Tdiff_{i * tau}" for i in range(1, E + 1)]

    outlist = []
    for p in up:
        pind = np.flatnonzero(popv == p)
        xyi, timei = xy[pind], timev[pind]
        nm = np.flatnonzero(~np.isnan(xyi).any(axis=1))   # rows with no missing value
        if forecast:
            if Tdiff_fore is None:
                # NOTE: set from the first population and reused for the others, as in R
                Tdiff_fore = np.min(np.diff(timei)) * tau
            time2 = np.concatenate([timei, timei[-1] + np.atleast_1d(Tdiff_fore)])
        else:
            time2 = timei

        out = np.full((len(time2), E * (nv + 1)), np.nan)
        for i in range(E * tau, len(time2)):   # the first E*tau rows get no lags
            prev = i                           # row the previous lag was taken from
            for Ei in range(1, E + 1):
                # lag Ei: the latest complete row that lies at least tau*Ei rows
                # before row i and at least tau rows before the previous lag
                k = np.searchsorted(nm, min(i - tau * Ei, prev - tau), side="right") - 1
                if k < 0:
                    break                      # no such row: this lag and all deeper ones stay NaN
                new = nm[k]
                out[i, np.arange(nv) * E + Ei - 1] = xyi[new]
                out[i, nv * E + Ei - 1] = time2[prev] - time2[new]
                prev = new

        if forecast:
            res = pd.DataFrame(out[len(timei):], columns=cols)
            res.insert(0, tname, time2[len(timei):])
            if len(up) > 1:
                res.insert(0, pname, p)
        else:
            res = pd.DataFrame(out, columns=cols)
        outlist.append(res)

    # NOTE: rows come out grouped by population, in order of first appearance.
    # They line up with `data` only if `data` is sorted the same way.
    output = pd.concat(outlist, ignore_index=True)
    if Tdiff_max is not None:
        tcols = [c for c in output.columns if "Tdiff_" in c]
        output[tcols] = output[tcols].where(~(output[tcols] > Tdiff_max))
    return output


# ----------------------------------------------------------------------------
# Likelihood, priors and gradient
# ----------------------------------------------------------------------------

def logit(x, lo=0.0, hi=1.0):
    return np.log((x - lo) / (hi - x))


def sqdist(X1, X2):
    """Squared Euclidean distance between the rows of X1 and of X2 (laGP::distance)."""
    X1 = np.asarray(X1, dtype=float).reshape(len(X1), -1)
    X2 = np.asarray(X2, dtype=float).reshape(len(X2), -1)
    D = np.zeros((len(X1), len(X2)))
    for k in range(X1.shape[1]):   # summed column by column, like laGP's C loop
        D += (X1[:, k, None] - X2[None, :, k]) ** 2
    return D


def getcov(phi, sigma2, rho, X, X2, pop, pop2):
    """Port of getcov(): covariance of the rows of X2 (rows) with the rows of X (columns)."""
    lC0 = -sqdist(X2 * np.sqrt(phi), X * np.sqrt(phi))
    popsame = (np.asarray(pop2)[:, None] == np.asarray(pop)[None, :]) * 1.0
    Cd = sigma2 * np.exp(lC0) * (popsame + rho * (1 - popsame))
    return popsame, Cd


def getcovinv(Sigma):
    """Port of getcovinv(): upper Cholesky factor L (Sigma = L'L) and the inverse of Sigma."""
    L, info = lapack.dpotrf(Sigma, lower=0)
    if info != 0:
        raise np.linalg.LinAlgError("covariance matrix is not positive definite")
    inv, info = lapack.dpotri(L, lower=0)
    if info != 0:
        raise np.linalg.LinAlgError("inversion of the covariance matrix failed")
    return L, np.triu(inv) + np.triu(inv, 1).T


def getpriors(phi, ve, sigma2, rho, modeprior, sigma2max, vemax):
    """
    Port of getpriors(): log prior and its derivative with respect to each parameter.

    phi_k    half-normal with variance lam_phi (modeprior=1 gives pi/2)
    ve/vemax, sigma2/sigma2max, rho    Beta(2, 2)
    """
    lam_phi = (2 ** (modeprior - 1)) ** 2 * np.pi / 2
    lp_phi = -0.5 * np.sum(phi ** 2) / lam_phi
    dlp_phi = -phi / lam_phi

    a, b = 2, 2
    lp_sigma2 = (a - 1) * np.log(sigma2 / sigma2max) + (b - 1) * np.log(1 - sigma2 / sigma2max)
    dlp_sigma2 = (a - 1) / sigma2 - (b - 1) / (sigma2max - sigma2)
    lp_ve = (a - 1) * np.log(ve / vemax) + (b - 1) * np.log(1 - ve / vemax)
    dlp_ve = (a - 1) / ve - (b - 1) / (vemax - ve)
    lp_rho = (a - 1) * np.log(rho) + (b - 1) * np.log(1 - rho)
    dlp_rho = (a - 1) / rho - (b - 1) / (1 - rho)

    lp = lp_phi + lp_ve + lp_sigma2 + lp_rho
    dlp = np.concatenate([dlp_phi, [dlp_ve, dlp_sigma2, dlp_rho]])
    return lp, dlp


def getlikegrad(parst, Y, X, pop, modeprior, fixedpars, rhofixed, D, returngradonly):
    """
    Port of getlikegrad(): negative log posterior and its gradient.

    parst holds the d + 3 parameters on the unconstrained scale the optimiser
    works on: log(phi_1..phi_d), then ve, sigma2 and rho mapped through a
    logistic curve onto their allowed intervals.  D[k] is the matrix of squared
    distances in column k of X.
    """
    d = X.shape[1]
    parst = np.array(parst, dtype=float)   # copy: fixed values are written back into it
    phi = np.exp(parst[:d])
    ve = (VEMAX - VEMIN) / (1 + np.exp(-parst[d])) + VEMIN
    sigma2 = (SIGMA2MAX - SIGMA2MIN) / (1 + np.exp(-parst[d + 1])) + SIGMA2MIN
    rho = (RHOMAX - RHOMIN) / (1 + np.exp(-parst[d + 2])) + RHOMIN

    # fixed parameters replace whatever the optimiser proposed
    if len(pd.unique(pop)) == 1:
        rhofixed = 0.5
    if rhofixed is not None:
        rho = rhofixed
        parst[d + 2] = logit(rho, RHOMIN, RHOMAX)
    phifixed = ~np.isnan(fixedpars[:d])
    if phifixed.any():
        phi[phifixed] = fixedpars[:d][phifixed]
        parst[:d] = np.log(phi)
    if not np.isnan(fixedpars[d]):
        ve = fixedpars[d]
        parst[d] = logit(ve, VEMIN, VEMAX)
    if not np.isnan(fixedpars[d + 1]):
        sigma2 = fixedpars[d + 1]
        parst[d + 1] = logit(sigma2, SIGMA2MIN, SIGMA2MAX)

    # derivative of each parameter with respect to its unconstrained version
    dpars = np.concatenate([phi, [
        (ve - VEMIN) * (1 - (ve - VEMIN) / (VEMAX - VEMIN)),
        (sigma2 - SIGMA2MIN) * (1 - (sigma2 - SIGMA2MIN) / (SIGMA2MAX - SIGMA2MIN)),
        (rho - RHOMIN) * (1 - (rho - RHOMIN) / (RHOMAX - RHOMIN)),
    ]])

    lp, dlp = getpriors(phi, ve, sigma2, rho, modeprior, SIGMA2MAX, VEMAX)

    popsame, Cd = getcov(phi, sigma2, rho, X, X, pop, pop)
    Sigma = Cd + ve * np.eye(len(Y))
    L, iKVs = getcovinv((Sigma + Sigma.T) / 2)
    a = iKVs @ Y
    like = -0.5 * (Y @ a) - np.sum(np.log(np.diag(L)))   # log likelihood, up to a constant

    # d(like)/d(theta) = sum(Q * dSigma/dtheta), with Q = (a a' - Sigma^-1) / 2
    Q = 0.5 * (np.outer(a, a) - iKVs)
    dl = np.zeros(d + 3)
    for i in range(d):
        if np.isnan(fixedpars[i]):
            # NOTE: dSigma/dphi_i = -D_i * Cd, so the derivative is sum(Q * (-D_i * Cd)).
            # R multiplies it by a further 0.5, and so does this port.  Finite
            # differences confirm the factor (gpedm_py_validation/check_gradient.py).
            # Rprop only uses the sign of each gradient entry, so the fit ends at the
            # maximum of the posterior with the phi prior counted twice, not at the
            # maximum of the nllpost reported here.
            dl[i] = 0.5 * np.sum(Q * (-D[i] * Cd))
    if np.isnan(fixedpars[d]):
        dl[d] = np.sum(np.diag(Q))                  # dSigma/dve = I
    if np.isnan(fixedpars[d + 1]):
        dl[d + 1] = np.sum(Q * (Cd / sigma2))
    if rhofixed is None:
        dl[d + 2] = np.sum(Q * (Cd * (1 - popsame) / rho))

    # NOTE: the prior gradient dlp is added for fixed parameters too, so their
    # entries of the gradient are not zero and count in the stopping rule.
    neglgrad = -(dl + dlp) * dpars
    neglpost = -(like + lp)
    if returngradonly:
        return {"nllpost": neglpost, "grad": neglgrad}

    names = [f"phi{i + 1}" for i in range(d)] + ["ve", "sigma2", "rho"]
    return {
        "pars": pd.Series(np.concatenate([phi, [ve, sigma2, rho]]), index=names),
        "parst": pd.Series(parst, index=names),
        "nllpost": neglpost,
        "grad": pd.Series(neglgrad, index=names),
        "lnL_LOO": 0.5 * np.sum(np.log(np.diag(iKVs))) - 0.5 * np.sum(a ** 2 / np.diag(iKVs)),
        "df": np.trace(Cd @ iKVs),
        "mean": Cd @ a,                    # in-sample posterior mean of f
        "cov": Cd - Cd @ iKVs @ Cd,        # in-sample posterior covariance of f
        "covm": {"popsame": popsame, "Cd": Cd, "Sigma": Sigma, "iKVs": iKVs},
        "SS": Y @ a,
        "logdet": np.sum(np.log(np.diag(L))),
        "lp": lp,
    }


def fmingrad_Rprop(initparst, Y, X, pop, modeprior, fixedpars, rhofixed, trace=False):
    """
    Port of fmingrad_Rprop(): minimise the negative log posterior with Rprop.

    Each parameter has its own step size.  A step moves every parameter against
    the sign of its gradient entry; the step size grows by 1.2 when that entry
    kept its sign and halves when it flipped.  The loop stops when the gradient
    norm falls below 1e-4, the relative change of the objective falls below
    1e-7, or after 200 iterations.
    """
    p = len(initparst)
    Delta0 = 0.1 * np.ones(p)
    Deltamin, Deltamax = 1e-6, 50
    eta_minus = 0.5 - 1
    eta_plus = 1.2 - 1
    maxiter = 200

    D = [sqdist(X[:, i], X[:, i]) for i in range(X.shape[1])]

    out = getlikegrad(initparst, Y, X, pop, modeprior, fixedpars, rhofixed, D, True)
    nllpost, grad = out["nllpost"], out["grad"]
    s = np.sqrt(grad @ grad)
    pa = np.array(initparst, dtype=float)
    it = 0
    delta = Delta0
    df = 10
    history = [dict(iter=0, nllpost=nllpost, gradnorm=s, relchange=np.nan, parst=pa.copy(), grad=grad)]

    while s > 1e-4 and it < maxiter and df > 1e-7:
        panew = pa - np.sign(grad) * delta
        out = getlikegrad(panew, Y, X, pop, modeprior, fixedpars, rhofixed, D, True)
        nllpost_new, grad_new = out["nllpost"], out["grad"]
        s = np.sqrt(grad_new @ grad_new)
        if not np.isfinite(s):
            raise FloatingPointError("the gradient of the log posterior is not finite")
        df = abs(nllpost_new / nllpost - 1)
        gc = grad * grad_new
        tdelta = delta * (1 + eta_plus * (gc > 0) + eta_minus * (gc < 0))
        delta = np.clip(tdelta, Deltamin, Deltamax)
        pa, grad, nllpost = panew, grad_new, nllpost_new
        it += 1
        history.append(dict(iter=it, nllpost=nllpost, gradnorm=s, relchange=df, parst=pa.copy(), grad=grad))

    out = getlikegrad(pa, Y, X, pop, modeprior, fixedpars, rhofixed, D, False)
    out["iter"] = it
    out["trace"] = history if trace else None
    return out


# ----------------------------------------------------------------------------
# Fit and predict
# ----------------------------------------------------------------------------

@dataclass
class GPFit:
    """What fitGP() returns; the fields carry the names of the R list."""
    pars: pd.Series            # phi1..phid, ve, sigma2, rho
    parst: pd.Series           # the same on the optimiser's unconstrained scale
    grad: pd.Series            # gradient of the negative log posterior at the end
    nllpost: float             # negative log posterior at the end
    iter: int                  # Rprop iterations used (200 = hit the limit)
    covm: dict                 # Cd, Sigma, iKVs (inverse of Sigma), popsame
    inputs: dict               # data as given, plus the standardised complete rows Y, X, Pop, Time
    scaling: dict              # scaling, ymeans, ysds, xmeans, xsds
    insampresults: pd.DataFrame
    insampfitstats: pd.Series
    trace: pd.DataFrame = None  # one row per Rprop iteration, if fitGP(trace=True)


@dataclass
class GPPred:
    """What predict() returns."""
    outsampresults: pd.DataFrame
    outsampfitstats: pd.Series = None


def getR2(obs, pred):
    ok = ~(np.isnan(obs) | np.isnan(pred))
    return 1 - np.sum((obs[ok] - pred[ok]) ** 2) / np.sum((obs[ok] - np.mean(obs[ok])) ** 2)


def _default_time(pop):
    time = np.zeros(len(pop))
    for p in pd.unique(pop):
        time[pop == p] = np.arange(1, np.sum(pop == p) + 1)
    return time


def _unscale(ypred, yfsd, ysd, pop, scaling):
    """Put standardised predictions back on the scale of the data."""
    if scaling["scaling"] == "global":
        ymeans, ysds = scaling["ymeans"], scaling["ysds"]
        return ypred * ysds + ymeans, np.sqrt(yfsd ** 2 * ysds ** 2), np.sqrt(ysd ** 2 * ysds ** 2)
    if scaling["scaling"] == "local":
        ypred, yfsd, ysd = ypred.copy(), yfsd.copy(), ysd.copy()
        for p in pd.unique(pop):
            m = pop == p
            locmean, locsd = scaling["ymeans"][p], scaling["ysds"][p]
            ypred[m] = ypred[m] * locsd + locmean
            yfsd[m] = np.sqrt(yfsd[m] ** 2 * locsd ** 2)
            ysd[m] = np.sqrt(ysd[m] ** 2 * locsd ** 2)
    return ypred, yfsd, ysd


def fitGP(data, y, x, pop=None, time=None, scaling="global", initpars=None, modeprior=1,
          fixedpars=None, rhofixed=None, trace=False):
    """
    Port of fitGP().

    data : DataFrame; y : response column; x : predictor column(s), normally the
    output of makelags(); pop, time : optional column names.
    scaling : "global" standardises y and each x over all rows, "local" within
        each population, "none" leaves them alone.
    initpars : starting values (phi_1..phi_d, ve, sigma2, rho);
        default 0.1 for every phi, ve=0.001, sigma2=0.999, rho=0.5.
    fixedpars : length d + 2 (phi_1..phi_d, ve, sigma2), NaN where the parameter is free.
    rhofixed : fix rho instead of estimating it.
    trace : keep the path of the optimiser in fit.trace.

    Rows with a missing y or x are left out of the fit.  A predictor that is
    constant has zero variance, so scaling turns it into NaN and every row is
    left out; that is an error here.  It is what happens to the Tdiff columns
    of makelags(vtimestep=True) when the series has no gap.
    """
    if scaling not in ("global", "local", "none"):
        raise ValueError("scaling must be 'global', 'local' or 'none'")
    if rhofixed is not None:
        rhofixed = min(max(rhofixed, 1e-4), 0.9999)

    data = pd.DataFrame(data)
    x_names = [x] if isinstance(x, str) else list(x)
    yd = data[y].to_numpy(dtype=float)
    xd = data[x_names].to_numpy(dtype=float)
    n, d = xd.shape
    popd = np.ones(n, dtype=int) if pop is None else data[pop].to_numpy()
    timed = _default_time(popd) if time is None else data[time].to_numpy()
    up = pd.unique(popd)

    # a column with zero variance turns into NaN here
    with np.errstate(invalid="ignore", divide="ignore"):
        if scaling == "local":
            ymeans = {p: np.nanmean(yd[popd == p]) for p in up}
            ysds = {p: np.nanstd(yd[popd == p], ddof=1) for p in up}
            xmeans = {p: np.nanmean(xd[popd == p], axis=0) for p in up}
            xsds = {p: np.nanstd(xd[popd == p], axis=0, ddof=1) for p in up}
            yds, xds = yd.copy(), xd.copy()
            for p in up:
                m = popd == p
                yds[m] = (yd[m] - ymeans[p]) / ysds[p]
                xds[m] = (xd[m] - xmeans[p]) / xsds[p]
        else:
            ymeans, ysds = np.nanmean(yd), np.nanstd(yd, ddof=1)
            xmeans, xsds = np.nanmean(xd, axis=0), np.nanstd(xd, axis=0, ddof=1)
            if scaling == "global":
                yds, xds = (yd - ymeans) / ysds, (xd - xmeans) / xsds
            else:
                yds, xds = yd, xd
                if abs(ymeans) > 0.1 or ysds > 1.1:
                    warnings.warn(f"y is not scaled properly, model may be unreliable "
                                  f"mean={ymeans:.3f} sd={ysds:.3f}")
                if np.any(np.abs(xmeans) > 0.1) or np.any(xsds > 1.1):
                    warnings.warn("one or more x is not scaled properly, model may be unreliable.")
    scaling = dict(scaling=scaling, ymeans=ymeans, ysds=ysds, xmeans=xmeans, xsds=xsds)

    if initpars is None:
        initpars = np.concatenate([np.full(d, 0.1), [0.001, 1 - 0.001, 0.5]])
    initpars = np.array(initpars, dtype=float)
    if fixedpars is None:
        fixedpars = np.full(d + 2, np.nan)
    fixedpars = np.array(fixedpars, dtype=float)
    if len(fixedpars) != d + 2:
        raise ValueError("Length of fixedpars must be number of predictors + 2 (phi, ve, sigma2)")
    isfixed = ~np.isnan(fixedpars)
    initpars[:d + 2][isfixed] = fixedpars[isfixed]

    initparst = np.concatenate([np.log(initpars[:d]), [
        logit(initpars[d], VEMIN, VEMAX),
        logit(initpars[d + 1], SIGMA2MIN, SIGMA2MAX),
        logit(initpars[d + 2], 0, 1),
    ]])
    if len(up) == 1:
        initparst[d + 2] = 0

    completerows = ~(np.isnan(yds) | np.isnan(xds).any(axis=1))
    if not completerows.any():
        allnan = [c for c, col in zip(x_names, xds.T) if np.isnan(col).all()]
        raise ValueError("no complete rows to fit; predictors that are missing everywhere after "
                         f"scaling (constant columns have zero variance): {allnan}")
    Y, X = yds[completerows], xds[completerows]
    Pop, Time = popd[completerows], timed[completerows]

    out = fmingrad_Rprop(initparst, Y, X, Pop, modeprior, fixedpars, rhofixed, trace)

    ypred, yfsd, ysd = (np.full(n, np.nan) for _ in range(3))
    ypred[completerows] = out["mean"]
    yfsd[completerows] = np.sqrt(np.diag(out["cov"]))
    ysd[completerows] = np.sqrt(np.diag(out["cov"]) + out["pars"]["ve"])
    ypred, yfsd, ysd = _unscale(ypred, yfsd, ysd, popd, scaling)

    insampresults = pd.DataFrame(dict(timestep=timed, pop=popd, predmean=ypred,
                                      predfsd=yfsd, predsd=ysd, obs=yd))
    insampfitstats = pd.Series(dict(
        R2=getR2(yd, ypred), rmse=np.sqrt(np.nanmean((yd - ypred) ** 2)),
        ln_post=-out["nllpost"], ln_prior=out["lp"], lnL_LOO=out["lnL_LOO"],
        df=out["df"], SS=out["SS"], logdet=out["logdet"]))

    if trace:
        names = list(out["pars"].index)
        trace = pd.DataFrame([
            {"iter": h["iter"], "nllpost": h["nllpost"], "gradnorm": h["gradnorm"],
             "relchange": h["relchange"],
             **{f"parst_{k}": v for k, v in zip(names, h["parst"])},
             **{f"grad_{k}": v for k, v in zip(names, h["grad"])}}
            for h in out["trace"]])
    else:
        trace = None

    inputs = dict(y_names=y, x_names=x_names, pop_names=pop, time_names=time,
                  y=yd, x=xd, pop=popd, time=timed, completerows=completerows,
                  Y=Y, X=X, Pop=Pop, Time=Time)
    return GPFit(pars=out["pars"], parst=out["parst"], grad=out["grad"], nllpost=out["nllpost"],
                 iter=out["iter"], covm=out["covm"], inputs=inputs, scaling=scaling,
                 insampresults=insampresults, insampfitstats=insampfitstats, trace=trace)


def predict(fit, newdata=None, predictmethod="loo", exclradius=0):
    """
    Port of predict.GP().

    newdata : DataFrame holding the predictor columns the model was fitted on
        (and its pop and time columns, if it had any), for instance the output
        of makelags(forecast=True).  Gives out-of-sample predictions.
    Without newdata, the training rows are predicted according to predictmethod:
        "loo"        leave one row out (and exclradius rows on either side of it)
        "lto"        leave one time point out, in all populations
        "sequential" predict each time point from the earlier ones

    predmean is the predicted mean, predfsd the sd of the function f alone and
    predsd the sd of a new observation (process noise ve included).
    """
    iKVs = fit.covm["iKVs"]
    d = fit.inputs["X"].shape[1]
    phi = fit.pars.to_numpy()[:d]
    ve, sigma2, rho = fit.pars["ve"], fit.pars["sigma2"], fit.pars["rho"]
    X, Y, Pop = fit.inputs["X"], fit.inputs["Y"], fit.inputs["Pop"]
    scaling = fit.scaling

    if newdata is not None:
        newdata = pd.DataFrame(newdata)
        xnew = newdata[fit.inputs["x_names"]].to_numpy(dtype=float)
        Tslp = len(xnew)
        y_names = fit.inputs["y_names"]
        ynew = newdata[y_names].to_numpy(dtype=float) if y_names in newdata.columns else None
        pop_names, time_names = fit.inputs["pop_names"], fit.inputs["time_names"]
        popnew = np.ones(Tslp, dtype=int) if pop_names is None else newdata[pop_names].to_numpy()
        timenew = _default_time(popnew) if time_names is None else newdata[time_names].to_numpy()

        # the new predictors get the means and sds of the training data
        xnews = xnew.copy()
        if scaling["scaling"] == "global":
            xnews = (xnew - scaling["xmeans"]) / scaling["xsds"]
        elif scaling["scaling"] == "local":
            for p in pd.unique(popnew):
                m = popnew == p
                xnews[m] = (xnew[m] - scaling["xmeans"][p]) / scaling["xsds"][p]

        ypred, yfsd, ysd = (np.full(Tslp, np.nan) for _ in range(3))
        completerowsnew = ~np.isnan(xnews).any(axis=1)
        if completerowsnew.any():
            _, Cs = getcov(phi, sigma2, rho, X, xnews[completerowsnew], Pop, popnew[completerowsnew])
            ymean = Cs @ (iKVs @ Y)
            yvar = sigma2 - np.sum((Cs @ iKVs) * Cs, axis=1)
            ypred[completerowsnew] = ymean
            yfsd[completerowsnew] = np.sqrt(yvar)
            ysd[completerowsnew] = np.sqrt(yvar + ve)

    else:
        if predictmethod not in ("loo", "lto", "sequential"):
            raise ValueError("predictmethod must be 'loo', 'lto' or 'sequential'")
        Cd, Sigma, Time = fit.covm["Cd"], fit.covm["Sigma"], fit.inputs["Time"]
        nd = len(Y)

        if predictmethod == "loo":
            ymean, yvar = np.zeros(nd), np.zeros(nd)
            for i in range(nd):
                # NOTE: exclradius counts rows of the fitted (complete) data, not time steps
                exclude = [j for j in range(i - exclradius, i + exclradius + 1)
                           if 0 <= j < nd and Pop[j] == Pop[i]]
                keep = np.ones(nd, dtype=bool)
                keep[exclude] = False
                S_noi = Sigma[np.ix_(keep, keep)]
                _, iKVs_noi = getcovinv((S_noi + S_noi.T) / 2)
                Cdi = Cd[i, keep]
                ymean[i] = Cdi @ iKVs_noi @ Y[keep]
                yvar[i] = sigma2 - Cdi @ iKVs_noi @ Cdi
            yf, ys = np.sqrt(yvar), np.sqrt(yvar + ve)

        elif predictmethod == "lto":
            timevals = pd.unique(Time)
            nt = len(timevals)
            ymean, yvar = np.full(nd, np.nan), np.full(nd, np.nan)
            for i in range(nt):
                # NOTE: exclradius counts positions in the list of time points, not time units
                exclude = [j for j in range(i - exclradius, i + exclradius + 1) if 0 <= j < nt]
                ind = Time == timevals[i]
                train = ~np.isin(Time, timevals[exclude])
                S_noi = Sigma[np.ix_(train, train)]
                _, iKVs_noi = getcovinv((S_noi + S_noi.T) / 2)
                Cdi = Cd[np.ix_(ind, train)]
                ymean[ind] = Cdi @ iKVs_noi @ Y[train]
                yvar[ind] = sigma2 - np.sum((Cdi @ iKVs_noi) * Cdi, axis=1)
            yf, ys = np.sqrt(yvar), np.sqrt(yvar + ve)

        else:
            timevals = pd.unique(Time)
            nt = len(timevals)
            S = Cd.copy()
            ymean1 = np.zeros((nd, nt))
            # NOTE: R starts this matrix of standard deviations at the variance
            # sigma2 + ve, so the sd reported for the first time point is a variance
            ysd1 = (sigma2 + ve) * np.ones((nd, nt))
            for i in range(nt - 1):
                ind = np.flatnonzero(Time == timevals[i])
                SS = S[np.ix_(ind, ind)] + ve * np.eye(len(ind))
                _, iK = getcovinv((SS + SS.T) / 2)
                k = iK @ (Y[ind] - ymean1[ind, i])
                ymean1[:, i + 1] = ymean1[:, i] + S[:, ind] @ k
                S = S - S[:, ind] @ iK @ S[ind, :]
                ysd1[:, i + 1] = np.sqrt(np.abs(np.diag(S)) + ve)
            ymean, ys = np.full(nd, np.nan), np.full(nd, sigma2 + ve)
            for i in range(nt):
                ind = Time == timevals[i]
                ymean[ind] = ymean1[ind, i]
                ys[ind] = ysd1[ind, i]
            yf = np.sqrt(ys ** 2 - ve)

        completerows = fit.inputs["completerows"]
        ynew, popnew, timenew = fit.inputs["y"], fit.inputs["pop"], fit.inputs["time"]
        ypred, yfsd, ysd = (np.full(len(ynew), np.nan) for _ in range(3))
        ypred[completerows], yfsd[completerows], ysd[completerows] = ymean, yf, ys

    ypred, yfsd, ysd = _unscale(ypred, yfsd, ysd, popnew, scaling)
    out = pd.DataFrame(dict(timestep=timenew, pop=popnew, predmean=ypred, predfsd=yfsd, predsd=ysd))
    if ynew is None:
        return GPPred(outsampresults=out)
    out["obs"] = ynew
    stats = pd.Series(dict(R2=getR2(ynew, ypred), rmse=np.sqrt(np.nanmean((ynew - ypred) ** 2))))
    return GPPred(outsampresults=out, outsampfitstats=stats)
