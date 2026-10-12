"""
Joint exceedance of the median thresholds by the relative annual and Rx1day trends, and their rank correlation (Section 3.4).
Output: outputs/09b_joint_exceedance.csv
"""
import numpy as np, pandas as pd, rasterio
from scipy.stats import rankdata
import config as C

v = C.valid_mask()
W = C.lat_weights()[v]
rd = lambda n: rasterio.open(C.RASTERS / n).read(1)[v].astype("float64")
A = np.load(C.DATA / "cube_annual.npy").mean(0)
R = np.load(C.DATA / "cube_rx1.npy").mean(0)
a = 1000 * rd("annual_sen_1981_2025.tif") / A
r = 1000 * rd("rx1_sen_1981_2025.tif") / R
ok = np.isfinite(a) & np.isfinite(r)
a, r, W = a[ok], r[ok], W[ok]


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, q * c[-1])]


ta, tr = wq(np.abs(a), W, 0.5), wq(np.abs(r), W, 0.5)
ea, er = np.abs(a) > ta, np.abs(r) > tr
pw = lambda m: 100 * W[m].sum() / W.sum()
print("thresholds %.2f %.2f" % (ta, tr))
print("P(annual exceeds) %.1f  P(rx1 exceeds) %.1f  both %.1f  expected if independent %.1f" % (pw(ea), pw(er), pw(ea & er), pw(ea) * pw(er) / 100))
print("Spearman (unweighted) %.3f" % pd.Series(a).corr(pd.Series(r), method="spearman"))
ra, rr = rankdata(a), rankdata(r)
ma, mr = np.average(ra, weights=W), np.average(rr, weights=W)
cw = np.average((ra - ma) * (rr - mr), weights=W) / np.sqrt(np.average((ra - ma) ** 2, weights=W) * np.average((rr - mr) ** 2, weights=W))
print("Spearman (area-weighted ranks) %.3f" % cw)

pd.DataFrame([dict(pct_annual_exceeds=pw(ea), pct_rx1_exceeds=pw(er), pct_both=pw(ea & er), pct_expected_independent=pw(ea) * pw(er) / 100, spearman=pd.Series(a).corr(pd.Series(r), method="spearman"))]).to_csv(C.OUT / "09b_joint_exceedance.csv", index=False)
