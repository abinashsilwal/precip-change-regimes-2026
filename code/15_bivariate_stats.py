"""
Rx1day trend class (FDR-significant decrease, none, increase) by population tertile: area and population of each combination.
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
rd = lambda n: rasterio.open(C.RASTERS / n).read(1)
sen, fdr = rd("rx1_sen_1981_2025.tif"), rd("rx1_fdr_1981_2025.tif") > 0
pop = np.load(C.DATA / "pop_on_prism_grid.npy").astype("float64")
tr = np.where(fdr & (sen < 0), 0, np.where(fdr & (sen > 0), 2, 1))             # 0 decrease, 1 none, 2 increase
pos = valid & (pop > 0)
q = np.quantile(np.log10(pop[pos]), [1 / 3, 2 / 3])
pb = np.where(~pos, 0, np.where(np.log10(np.where(pop > 0, pop, 1)) <= q[0], 0, np.where(np.log10(np.where(pop > 0, pop, 1)) <= q[1], 1, 2)))
code = (3 * pb + tr + 1).astype("float32")
code[~valid] = np.nan

rows = []
W = C.lat_weights()
for i, p in enumerate(["Low", "Medium", "High"]):
    for j, t in enumerate(["Decrease", "No significant trend", "Increase"]):
        m = valid & (code == 3 * i + j + 1)
        rows.append(dict(population_bin=p, rx1_trend=t, area_pct=100 * W[m].sum() / W[valid].sum(), population_million=pop[m].sum() / 1e6,
                         population_pct=100 * pop[m].sum() / pop[valid].sum()))
pd.DataFrame(rows).to_csv(C.OUT / "15_bivariate_stats.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
