"""
Downloads the HUC2 region raster and the CONUS mean annual temperature series from Earth Engine.
"""
import numpy as np, pandas as pd, ee
import config as C, ee_io

ee_io.init()
huc = ee.FeatureCollection("USGS/WBD/2017/HUC02").map(lambda f: f.set("hn", ee.Number.parse(f.get("huc2"))))
img = huc.reduceToImage(["hn"], ee.Reducer.first()).rename("huc2")
a = ee_io.fetch(img, nr=1, nc=3)[0]
a = np.nan_to_num(a, nan=0).astype("int16")
np.save(C.DATA / "huc2.npy", a)
print("HUC2 ids:", np.unique(a), " cells assigned:", int((a > 0).sum()))

conus = ee_io.conus()
tm = ee.ImageCollection(C.PRISM_ASSET.replace("ANd", "ANd")).select("tmean")
rows = []
for y in C.YEARS:
    s = ee.Date.fromYMD(y, 1, 1)
    m = tm.filterDate(s, s.advance(1, "year")).mean()
    v = m.reduceRegion(ee.Reducer.mean(), conus, 4638, maxPixels=1e13, bestEffort=True).get("tmean").getInfo()
    rows.append(dict(year=y, tmean_c=v))
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "11_conus_tmean_series.csv", index=False)
b = np.polyfit(df.year, df.tmean_c, 1)[0]
print("CONUS tmean trend 1981-2025: %.3f C per decade (OLS)" % (b * 10))
