"""
Same population reference as the within-state one in step 20, but with larger regions: the nine NOAA climate regions and the 18 HUC2 regions.
Each region's population is spread over its area in proportion to the area of each class. Comparing that with the observed share shows how much of
the excess of population over area is left once you stay inside a region.
Output: outputs/25_region_null.csv
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
W = C.lat_weights()[valid]
pop = np.load(C.DATA / "pop_on_prism_grid.npy")[valid].astype("float64")
huc = np.load(C.DATA / "huc2.npy")[valid]
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
ids = pd.read_csv(C.OUT / "state_ids.csv", index_col=0)["state_id"]
rd = lambda name: rasterio.open(C.RASTERS / name).read(1)[valid]
CLIMATE = {"Northeast": "CT DE ME MD MA NH NJ NY PA RI VT DC", "Southeast": "AL FL GA NC SC VA", "South": "AR KS LA MS OK TX",
           "Southwest": "AZ CO NM UT", "West": "CA NV", "Northwest": "ID OR WA", "Northern Rockies and Plains": "MT NE ND SD WY",
           "Upper Midwest": "IA MI MN WI", "Ohio Valley": "IL IN KY MO OH TN WV"}
noaa = np.zeros(valid.sum(), int)
for i, (k, v) in enumerate(CLIMATE.items(), 1):
    noaa[np.isin(state, [ids[s] for s in v.split() if s in ids.index])] = i

rows = []
for scheme, lab in (("rel", "relative"), ("abs", "absolute")):
    cl = rd(f"class5_{scheme}_p50.tif")
    for gname, g in (("state", state), ("NOAA climate region", noaa), ("HUC2 region", huc)):
        for c, cname in ((4, "Wetting + stronger extremes"), (1, "Drying + weaker extremes"), (2, "Drying + stronger extremes")):
            m = cl == c
            null = 0.0
            for k in np.unique(g):
                if k == 0:
                    continue
                s = g == k
                null += pop[s].sum() * W[s & m].sum() / W[s].sum()
            rows.append(dict(thresholds=lab, grouping=gname, cls=cname, area_pct=100 * W[m].sum() / W.sum(), pop_pct=100 * pop[m].sum() / pop.sum(),
                             null_pct=100 * null / pop[g > 0].sum()))
df = pd.DataFrame(rows)
df["excess_pts"] = df.pop_pct - df.area_pct
df["explained_by_grouping_pts"] = df.null_pct - df.area_pct
df.to_csv(C.OUT / "25_region_null.csv", index=False)
pd.set_option("display.width", 220)
print(df.round(2).to_string(index=False))
