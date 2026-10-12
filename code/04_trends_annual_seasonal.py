"""
Annual and seasonal trends from the monthly cube: tie-corrected Mann-Kendall test, Sen slope, Hamed-Rao check and
Benjamini-Hochberg FDR. Also splits the annual trend into seasonal contributions (DJF taken as January, February and December of the
same year so that the seasons add up to the annual total).
"""
import numpy as np, pandas as pd, rasterio
import config as C, trend as T

valid = C.valid_mask()
P = int(valid.sum())
W = C.lat_weights()[valid]
n = len(C.YEARS)

C.log("loading monthly cube")
M = np.empty((n, 12, P), "float32")
for i, y in enumerate(C.YEARS):
    M[i] = np.load(C.DATA / f"monthly_{y}.npy")[:, valid]
assert np.isfinite(M).all(), "NaN inside valid mask"

annual = M.sum(1)
mam, jja, son = M[:, 2:5].sum(1), M[:, 5:8].sum(1), M[:, 8:11].sum(1)
djf_cal = M[:, 0] + M[:, 1] + M[:, 11]
djf_paper = M[:, 0] + M[:, 1]
djf_paper[1:] += M[:-1, 11]
series = {"annual": annual, "mam": mam, "jja": jja, "son": son, "djf_paper": djf_paper, "djf_cal": djf_cal,
          "djf_complete": djf_paper[1:]}
np.save(C.DATA / "cube_annual.npy", annual)                      # reused by later steps

rows, res = [], {}
for name, X in series.items():
    C.log("trend", name, X.shape)
    r = T.mk_sen(X)
    p_hr, fac = T.hamed_rao(X, r["S"], r["var"], r["sen"])
    f_mk, pc_mk = T.bh_fdr(r["p"], 0.10)
    f_hr, pc_hr = T.bh_fdr(p_hr, 0.10)
    res[name] = dict(sen=r["sen"], p=r["p"], p_hr=p_hr, fdr=f_mk, fdr_hr=f_hr, fac=fac)
    for k in ("sen", "p", "p_hr"):
        C.write_tif(C.RASTERS / f"{name}_{k}_1981_2025.tif", C.to_grid(res[name][k], valid))
    C.write_tif(C.RASTERS / f"{name}_fdr_1981_2025.tif", C.to_grid(f_mk.astype("float32"), valid))
    pct = lambda m: 100 * W[m].sum() / W.sum()
    rows.append(dict(series=name, n_years=X.shape[0],
                     pct_area_p05_uncorrected=pct(r["p"] < 0.05), pct_area_fdr=pct(f_mk), pct_area_fdr_hamed_rao=pct(f_hr),
                     pct_area_fdr_pos=pct(f_mk & (r["sen"] > 0)), pct_area_fdr_neg=pct(f_mk & (r["sen"] < 0)),
                     median_hamed_rao_factor=float(np.median(fac)), pct_pixels_factor_gt_1p5=100 * np.mean(fac > 1.5),
                     p_crit_fdr=pc_mk))
tab = pd.DataFrame(rows)
tab.to_csv(C.OUT / "04_trend_significance_summary.csv", index=False)
print(tab.round(2).to_string(index=False))

# ---- seasonal contribution to the annual change ----
# OLS slopes of the four calendar-year-partitioning seasons add up to the annual OLS slope exactly.
seas = {"DJF": djf_cal, "MAM": mam, "JJA": jja, "SON": son}
ols = {k: T.ols_slope(v) for k, v in seas.items()}
ols_ann = T.ols_slope(annual)
print("\nmax |sum of seasonal OLS slopes - annual OLS slope| =", float(np.abs(sum(ols.values()) - ols_ann).max()))
mean_ann = annual.mean(0)
dec = {k: 10 * v for k, v in ols.items()}                         # mm per decade
dec["annual"] = 10 * ols_ann
out = []
for k, v in dec.items():
    out.append(dict(season=k, conus_mean_trend_mm_per_decade=float((W * v).sum() / W.sum()),
                    conus_mean_trend_pct_of_annual_mean_per_decade=float((W * 100 * v / mean_ann).sum() / W.sum())))
cs = pd.DataFrame(out)
cs.to_csv(C.OUT / "04_seasonal_contribution_conus.csv", index=False)
print(cs.round(3).to_string(index=False))
# dominant season among pixels with a significant annual trend (FDR): which season carries most of the annual change
st = np.stack([ols[k] for k in ("DJF", "MAM", "JJA", "SON")])
names = np.array(["DJF", "MAM", "JJA", "SON"])
sgn = np.sign(ols_ann)
dom = names[np.argmax(st * sgn, axis=0)]                          # season with the largest contribution in the direction of change
rec = []
for grp, m in [("annual FDR-significant wetting", res["annual"]["fdr"] & (ols_ann > 0)),
               ("annual FDR-significant drying", res["annual"]["fdr"] & (ols_ann < 0)), ("all CONUS", np.ones(P, bool))]:
    for k in names:
        rec.append(dict(group=grp, dominant_season=k, pct_area=100 * W[m & (dom == k)].sum() / W[m].sum()))
pd.DataFrame(rec).to_csv(C.OUT / "04_dominant_season.csv", index=False)
print(pd.DataFrame(rec).round(1).pivot(index="group", columns="dominant_season", values="pct_area").to_string())
C.log("done")
