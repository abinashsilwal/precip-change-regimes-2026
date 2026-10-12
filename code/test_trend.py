"""Sanity tests for trend.py against independent implementations. Run: python test_trend.py"""
import numpy as np
from scipy.stats import theilslopes, kendalltau
import trend as T

rng = np.random.default_rng(1)
n, P = 45, 300
X = (rng.gamma(2, 20, (n, P)) + np.arange(n)[:, None] * rng.normal(0, 0.5, P)).astype("float32")
X[:, :50] = np.round(X[:, :50] / 10) * 10          # many ties
t = np.arange(n)
r = T.mk_sen(X)
sen_ref = np.array([theilslopes(X[:, i], t)[0] for i in range(P)])
print("Sen vs scipy theilslopes max abs diff:", np.abs(r["sen"] - sen_ref).max())
S_ref = np.array([np.sign(X[j, i] - X[k, i]) for i in range(P) for j in range(n) for k in range(j)]).reshape(P, -1).sum(1)
print("S vs brute force max abs diff:", np.abs(r["S"] - S_ref).max())
tau_p = np.array([kendalltau(t, X[:, i], variant="b")[1] for i in range(P)])
print("p vs scipy kendalltau-b (asymptotic, tie aware): median |diff| = %.4f, corr = %.4f" %
      (np.median(np.abs(r["p"] - tau_p)), np.corrcoef(r["p"], tau_p)[0, 1]))
# Pettitt: brute force on a few columns
K, tau, p = T.pettitt(X[:, :5])
for i in range(5):
    U = [sum(np.sign(X[j, i] - X[k, i]) for k in range(tt + 1) for j in range(tt + 1, n)) for tt in range(n - 1)]
    assert abs(max(abs(np.array(U))) - K[i]) < 1e-4 and np.argmax(np.abs(U)) == tau[i]
print("Pettitt matches brute force")
rej, pc = T.bh_fdr(np.r_[np.full(10, 1e-4), np.full(90, 0.5)], 0.1)
print("BH rejects", rej.sum(), "of 100 (expect 10)")
p_hr, f = T.hamed_rao(X, r["S"], r["var"], r["sen"])
print("Hamed-Rao factor range %.2f-%.2f (white noise should be near 1)" % (f.min(), f.max()), "median", np.median(f))
