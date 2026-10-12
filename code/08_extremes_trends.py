"""
Mann-Kendall / Sen / FDR trends for all extreme indices, agreement between OLS and Sen slopes, and a look at what drives
the R95p trend (days above the threshold, wet-day frequency, R10mm and R20mm).
"""
import numpy as np, pandas as pd
import config as C, trend as T

BANDS = ["rx1", "rx5", "r10", "r20", "wet", "r95p_wet", "n95_wet", "r95p_all", "n95_all"]
BANDS = [b for b in BANDS if b in C.load_extremes(C.YEARS[0]) and all(b in C.load_extremes(y) for y in C.YEARS)]   # only what is downloaded
print("bands available:", BANDS)
valid = C.valid_mask()
P = int(valid.sum())
W = C.lat_weights()[valid]
n = len(C.YEARS)

C.log("loading extremes")
cube = np.empty((n, len(BANDS), P), "float32")
for i, y in enumerate(C.YEARS):
    ex = C.load_extremes(y, BANDS)
    cube[i] = np.stack([ex[b][valid] for b in BANDS])
for b, name in enumerate(BANDS):
    np.save(C.DATA / f"cube_{name}.npy", cube[:, b])
print("cells with NaN:", int((~np.isfinite(cube)).sum()))

pct = lambda m: 100 * W[m].sum() / W.sum()
rows, res = [], {}
for b, name in enumerate(BANDS):
    X = cube[:, b]
    C.log("trend", name)
    r = T.mk_sen(X)
    p_hr, fac = T.hamed_rao(X, r["S"], r["var"], r["sen"])
    fdr, pc = T.bh_fdr(r["p"], 0.10)
    fdr_hr, _ = T.bh_fdr(p_hr, 0.10)
    ols = T.ols_slope(X)
    res[name] = dict(sen=r["sen"], p=r["p"], fdr=fdr, ols=ols)
    for k, v in (("sen", r["sen"]), ("p", r["p"]), ("p_hr", p_hr), ("fdr", fdr.astype("float32"))):
        C.write_tif(C.RASTERS / f"{name}_{k}_1981_2025.tif", C.to_grid(v, valid))
    rows.append(dict(index=name, conus_mean=float((W * X.mean(0)).sum() / W.sum()),
                     pct_area_p05_uncorrected=pct(r["p"] < 0.05), pct_area_fdr=pct(fdr), pct_area_fdr_pos=pct(fdr & (r["sen"] > 0)),
                     pct_area_fdr_neg=pct(fdr & (r["sen"] < 0)), pct_area_fdr_hamed_rao=pct(fdr_hr),
                     pct_area_sen_positive=pct(r["sen"] > 0), corr_ols_vs_sen=float(np.corrcoef(ols, r["sen"])[0, 1]),
                     sign_agree_ols_sen_pct=100 * float(np.mean(np.sign(ols) == np.sign(r["sen"]))),
                     conus_mean_pct_change_per_decade=float((W * 1000 * r["sen"] / np.maximum(X.mean(0), 1e-6)).sum() / W.sum())))
tab = pd.DataFrame(rows)
tab.to_csv(C.OUT / "08_extremes_trend_summary.csv", index=False)
print(tab.round(3).to_string(index=False))

if not {"r95p_wet", "r95p_all", "n95_wet"} <= set(BANDS):
    C.log("done (threshold-based bands missing; download them with 05_download_extremes.py q95 and rerun)")
    raise SystemExit

# ---- R95p with the 95th percentile of wet days vs of all days ----
a, b = res["r95p_wet"]["sen"], res["r95p_all"]["sen"]
print("\nR95p wet-day q95 vs all-day q95: Sen slope corr %.3f, sign agreement %.1f%%" %
      (np.corrcoef(a, b)[0, 1], 100 * np.mean(np.sign(a) == np.sign(b))))

# ---- where R95p falls while Rx1day rises: which other indices fall there? ----
rx, r95 = res["rx1"]["sen"], res["r95p_wet"]["sen"]
dec = (r95 < 0) & (rx > 0)
print("area %% with Sen R95p<0 and Sen Rx1day>0: %.1f" % pct(dec))
rec = []
for nm in ("n95_wet", "wet", "r10", "r20"):
    s = res[nm]["sen"]
    rec.append(dict(subset="R95p<0 & Rx1day>0", index=nm, pct_area_sen_neg=100 * W[dec & (s < 0)].sum() / W[dec].sum(),
                    mean_sen=float((W[dec] * s[dec]).sum() / W[dec].sum())))
    rec.append(dict(subset="all CONUS", index=nm, pct_area_sen_neg=pct(s < 0), mean_sen=float((W * s).sum() / W.sum())))
rec = pd.DataFrame(rec)
rec.to_csv(C.OUT / "08_r95p_decomposition.csv", index=False)
print(rec.round(3).to_string(index=False))
C.log("done")
