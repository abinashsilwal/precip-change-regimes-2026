"""
Pixel-wise trend statistics, vectorised over pixels. X is always (n_years, n_pixels).

mk_sen      Mann-Kendall S with the tie-corrected variance, z, two-sided p, and the Sen slope
hamed_rao   Mann-Kendall p after the Hamed and Rao (1998) correction for autocorrelation
bh_fdr      Benjamini-Hochberg false discovery rate
pettitt     Pettitt (1979) change-point test
ols_slope   ordinary least squares slope
"""
import numpy as np
from scipy.special import erfc
from scipy.stats import rankdata


def _chunks(P, size):
    for a in range(0, P, size):
        yield a, min(P, a + size)


def mk_sen(X, chunk=15000):
    n, P = X.shape
    iu, ju = np.triu_indices(n, 1)
    dt = (ju - iu).astype("float32")[:, None]
    S = np.zeros(P, "float32"); sen = np.zeros(P, "float32"); tie = np.zeros(P, "float32")
    for a, b in _chunks(P, chunk):
        x = X[:, a:b]
        d = x[ju] - x[iu]
        S[a:b] = np.sign(d).sum(0)
        sen[a:b] = np.median(d / dt, axis=0)
        e = (x[:, None, :] == x[None, :, :]).sum(1).astype("float32")     # size of the tie group of each value
        tie[a:b] = ((e - 1) * (2 * e + 5)).sum(0)
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    z = np.where(S > 0, (S - 1), np.where(S < 0, (S + 1), 0)) / np.sqrt(var)
    p = erfc(np.abs(z) / np.sqrt(2))
    return dict(S=S, var=var, z=z, p=p, sen=sen)


def hamed_rao(X, S, var_tied, sen, chunk=15000):
    """p-value after multiplying the (tie-corrected) variance by the Hamed-Rao factor n/n*."""
    n, P = X.shape
    t = np.arange(n, dtype="float32")[:, None]
    factor = np.ones(P, "float32")
    for a, b in _chunks(P, chunk):
        y = X[:, a:b] - sen[a:b][None, :] * t                              # detrend with Sen slope
        r = np.apply_along_axis(rankdata, 0, y).astype("float32")
        r -= r.mean(0)
        den = (r ** 2).sum(0)
        acc = np.zeros(b - a, "float32")
        for k in range(1, n - 1):
            rho = (r[:-k] * r[k:]).sum(0) / den
            sig = np.abs(rho) > 1.96 / np.sqrt(n)                          # only significant lags enter the correction
            acc += np.where(sig, (n - k) * (n - k - 1) * (n - k - 2) * rho, 0)
        factor[a:b] = 1 + 2.0 / (n * (n - 1) * (n - 2)) * acc
    factor = np.maximum(factor, 1.0)                                       # only ever inflate the variance (conservative)
    var = var_tied * factor
    z = np.where(S > 0, (S - 1), np.where(S < 0, (S + 1), 0)) / np.sqrt(var)
    return erfc(np.abs(z) / np.sqrt(2)), factor


def bh_fdr(p, alpha=0.10):
    """Benjamini-Hochberg: boolean array of rejected hypotheses (NaN p-values are never rejected)."""
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    passed = pv[order] <= (np.arange(1, m + 1) / m) * alpha
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(p.shape, bool)
    idx = np.where(ok)[0][order[:k]]
    rej[idx] = True
    pcrit = pv[order[k - 1]] if k else 0.0
    return rej, pcrit


def pettitt(X, chunk=4000):
    """returns K (max |U|), change index (last index of the first segment) and approximate two-sided p"""
    n, P = X.shape
    K = np.zeros(P, "float32"); tau = np.zeros(P, "int32")
    for a, b in _chunks(P, chunk):
        x = X[:, a:b]
        s = np.sign(x[None, :, :] - x[:, None, :]).astype("float32")      # s[i, j] = sign(x_j - x_i)
        U = np.stack([s[:t + 1, t + 1:].sum((0, 1)) for t in range(n - 1)])
        K[a:b] = np.abs(U).max(0)
        tau[a:b] = np.abs(U).argmax(0)
    p = np.minimum(1.0, 2 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2)))
    return K, tau, p


def ols_slope(X):
    n = X.shape[0]
    t = np.arange(n, dtype="float64")
    t -= t.mean()
    return (t[:, None] * (X - np.nanmean(X, 0))).sum(0) / (t ** 2).sum()
