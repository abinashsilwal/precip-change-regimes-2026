"""
Compares PRISM with nClimGrid-Daily on the same cells: trend agreement, Pettitt change-point tests on the national
series and on the difference PRISM minus nClimGrid, trends before and after 2002, east and west of 100W.
"""
import numpy as np, pandas as pd
import config as C, trend as T

valid = C.valid_mask()
n = len(C.YEARS)
yrs = np.array(C.YEARS)
W2 = C.lat_weights()
lon = C.TRANSFORM.c + (np.arange(C.SHAPE[1]) + 0.5) * C.TRANSFORM.a
EAST = np.repeat((lon > -100)[None, :], C.SHAPE[0], axis=0)

# ---- load PRISM and nClimGrid (n, H, W) for the indices compared ----
names = {"annual": None, "rx1": 0, "rx5": 1, "r10": 2, "wet": 4}   # index into the basic extremes file
prism = {k: np.empty((n,) + C.SHAPE, "float32") for k in names}
ncl = {k: np.empty((n,) + C.SHAPE, "float32") for k in names}
for i, y in enumerate(C.YEARS):
    mon = np.load(C.DATA / f"monthly_{y}.npy")
    prism["annual"][i] = mon.sum(0)
    ex = np.load(C.DATA / f"extremes_basic_{y}.npy")
    nc = np.load(C.DATA / f"nclim_{y}.npz")
    for k, b in names.items():
        if b is not None:
            prism[k][i] = ex[b]
        ncl[k][i] = nc[k]
common = valid & np.isfinite(ncl["annual"]).all(0) & np.isfinite(prism["annual"]).all(0)
Wc = W2[common]
C.log("common cells", int(common.sum()), "of", int(valid.sum()))
flat = lambda a: a[:, common]
pct = lambda m, w=Wc: 100 * w[m].sum() / w.sum()

# ---- 1. trend agreement ----
rows = []
trends = {}
for k in ("annual", "rx1", "rx5"):
    out = {}
    for src, d in (("prism", prism), ("nclim", ncl)):
        X = flat(d[k])
        r = T.mk_sen(X)
        fdr, _ = T.bh_fdr(r["p"], 0.10)
        out[src] = dict(sen=r["sen"], p=r["p"], fdr=fdr)
        C.write_tif(C.RASTERS / f"cmp_{src}_{k}_sen.tif", C.to_grid(r["sen"], common))
    a, b = out["prism"], out["nclim"]
    rows.append(dict(index=k, corr_sen=np.corrcoef(a["sen"], b["sen"])[0, 1],
                     sign_agree_pct=100 * np.mean(np.sign(a["sen"]) == np.sign(b["sen"])),
                     conus_mean_sen_prism=float((Wc * a["sen"]).sum() / Wc.sum()),
                     conus_mean_sen_nclim=float((Wc * b["sen"]).sum() / Wc.sum()),
                     pct_area_fdr_prism=pct(a["fdr"]), pct_area_fdr_nclim=pct(b["fdr"]), pct_area_fdr_both=pct(a["fdr"] & b["fdr"]),
                     pct_area_fdr_either_opposite_sign=pct((a["fdr"] | b["fdr"]) & (np.sign(a["sen"]) != np.sign(b["sen"])))))
    trends[k] = out
t1 = pd.DataFrame(rows)
t1.to_csv(C.OUT / "07_trend_agreement_prism_vs_nclimgrid.csv", index=False)
print(t1.round(3).to_string(index=False))

# ---- 2/4. national and regional series ----
def national(a, mask):
    w = W2 * mask
    return np.array([(a[i] * w)[np.isfinite(a[i])].sum() / w[np.isfinite(a[i])].sum() for i in range(n)])

regions = {"CONUS": common, "East of 100W": common & EAST, "West of 100W": common & ~EAST}
series, seg_rows, cp_rows = {}, [], []
for reg, m in regions.items():
    for k in ("annual", "rx1", "rx5"):
        sp, sn = national(prism[k], m), national(ncl[k], m)
        series[(reg, k)] = (sp, sn)
        for lab, s in (("PRISM", sp), ("nClimGrid", sn), ("PRISM - nClimGrid", sp - sn)):
            K, tau, p = T.pettitt(s[:, None])
            r = T.mk_sen(s[:, None])
            pre, post = yrs <= 2001, yrs >= 2002
            sl = lambda sel: float(T.mk_sen(s[sel][:, None])["sen"][0])
            cp_rows.append(dict(region=reg, index=k, series=lab, pettitt_p=float(p[0]), change_after_year=int(yrs[tau[0]]),
                                mean_1981_2001=float(s[pre].mean()), mean_2002_2025=float(s[post].mean()),
                                pct_step_2002=100 * float(s[post].mean() / s[pre].mean() - 1) if lab != "PRISM - nClimGrid" else np.nan,
                                sen_full=float(r["sen"][0]), sen_pre2002=sl(pre), sen_post2002=sl(post), mk_p_full=float(r["p"][0])))
cp = pd.DataFrame(cp_rows)
cp.to_csv(C.OUT / "07_national_series_changepoints.csv", index=False)
print(cp[cp.region == "CONUS"].round(3).to_string(index=False))

# ---- 3. pixel-wise Pettitt on the difference series ----
rows = []
hist = {}
for k in ("annual", "rx1"):
    D = flat(prism[k]) - flat(ncl[k])
    K, tau, p = T.pettitt(D)
    sig, _ = T.bh_fdr(p, 0.10)
    cy = yrs[tau]
    hist[k] = np.array([100 * Wc[sig & (cy == y)].sum() / Wc.sum() for y in yrs])
    rows.append(dict(index=k, pct_area_pettitt_p05=pct(p < 0.05), pct_area_pettitt_fdr=pct(sig),
                     pct_area_sig_change_2000_2004=pct(sig & (cy >= 2000) & (cy <= 2004)),
                     pct_area_sig_change_2019_2022=pct(sig & (cy >= 2019) & (cy <= 2022)),
                     pct_area_sig_east=100 * Wc[sig & EAST[common]].sum() / Wc[EAST[common]].sum(),
                     pct_area_sig_west=100 * Wc[sig & ~EAST[common]].sum() / Wc[~EAST[common]].sum()))
t3 = pd.DataFrame(rows)
t3.to_csv(C.OUT / "07_pixelwise_pettitt_on_difference.csv", index=False)
print(t3.round(2).to_string(index=False))
C.log("done")
