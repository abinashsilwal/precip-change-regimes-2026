"""
Regime areas, population and median trends by HUC2 region and NOAA climate region.
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
cls_abs, cls_rel = rd("class5_abs_p50.tif"), rd("class5_rel_p50.tif")
senA, senR = rd("annual_sen_1981_2025.tif"), rd("rx1_sen_1981_2025.tif")
fdrA, fdrR = rd("annual_fdr_1981_2025.tif") > 0, rd("rx1_fdr_1981_2025.tif") > 0
A, R = np.load(C.DATA / "cube_annual.npy").mean(0), np.load(C.DATA / "cube_rx1.npy").mean(0)
relA, relR = 1000 * senA / A, 1000 * senR / R

CLIMATE = {"Northeast": "CT DE ME MD MA NH NJ NY PA RI VT DC", "Southeast": "AL FL GA NC SC VA", "South": "AR KS LA MS OK TX",
           "Southwest": "AZ CO NM UT", "West": "CA NV", "Northwest": "ID OR WA", "Northern Rockies and Plains": "MT NE ND SD WY",
           "Upper Midwest": "IA MI MN WI", "Ohio Valley": "IL IN KY MO OH TN WV"}
HUC2 = {1: "New England", 2: "Mid-Atlantic", 3: "South Atlantic-Gulf", 4: "Great Lakes", 5: "Ohio", 6: "Tennessee", 7: "Upper Mississippi",
        8: "Lower Mississippi", 9: "Souris-Red-Rainy", 10: "Missouri", 11: "Arkansas-White-Red", 12: "Texas-Gulf", 13: "Rio Grande",
        14: "Upper Colorado", 15: "Lower Colorado", 16: "Great Basin", 17: "Pacific Northwest", 18: "California"}

def wmed(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])

def summarise(label_masks, out):
    rows = []
    for name, m in label_masks.items():
        if m.sum() == 0:
            continue
        r = dict(region=name, pct_conus_area=100 * W[m].sum() / W.sum(), pop_million=pop[m].sum() / 1e6, pct_conus_pop=100 * pop[m].sum() / pop.sum(),
                 median_annual_change_pct_per_decade=wmed(relA[m], W[m]), median_rx1_change_pct_per_decade=wmed(relR[m], W[m]),
                 pct_area_fdr_annual=100 * W[m & fdrA].sum() / W[m].sum(), pct_area_fdr_rx1=100 * W[m & fdrR].sum() / W[m].sum())
        for tag, cl in (("abs", cls_abs), ("rel", cls_rel)):
            for c, nm in ((4, "wet_strong"), (2, "dry_strong"), (1, "dry_weak"), (3, "wet_weak"), (5, "nochange")):
                r[f"{tag}_{nm}_area_pct"] = 100 * W[m & (cl == c)].sum() / W[m].sum()
            r[f"{tag}_wet_strong_pop_pct"] = 100 * pop[m & (cl == 4)].sum() / max(pop[m].sum(), 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / out, index=False)
    return df

hm = {f"{k:02d} {v}": huc == k for k, v in HUC2.items()}
d1 = summarise(hm, "11b_regions_huc2.csv")
cm = {k: np.isin(state, [ids[s] for s in v.split() if s in ids.index]) for k, v in CLIMATE.items()}
d2 = summarise(cm, "11b_regions_noaa_climate.csv")
pd.set_option("display.width", 250)
cols = ["region", "pct_conus_area", "pct_conus_pop", "median_annual_change_pct_per_decade", "median_rx1_change_pct_per_decade",
        "abs_wet_strong_area_pct", "rel_wet_strong_area_pct", "abs_dry_strong_area_pct"]
print(d1[cols].round(1).to_string(index=False))
print(d2[cols].round(1).to_string(index=False))
