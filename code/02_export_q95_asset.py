"""
Computes the exact 95th percentile of daily precipitation per pixel over 1981-2025 in Earth Engine and exports it
as an asset. Two bands: q95_wet (wet days of at least 1 mm) and q95_all (all days). Needed for R95p.
Usage: python 02_export_q95_asset.py
"""
import time, ee
import config as C, ee_io

ASSET = f"projects/{C.EE_PROJECT}/assets/prism_q95_1981_2025"

if __name__ == "__main__":
    ee_io.init()
    daily = ee_io.prism().filterDate("1981-01-01", "2026-01-01")
    n = daily.size().getInfo(); C.log("daily images", n)
    red = ee.Reducer.percentile([95], maxBuckets=1000, minBucketWidth=0.001, maxRaw=n + 10)
    wet = daily.map(lambda im: im.updateMask(im.gte(C.WET_DAY_MM))).reduce(red).rename("q95_wet")
    allq = daily.reduce(red).rename("q95_all")
    img = wet.addBands(allq)
    proj = daily.first().projection()
    C.log("native projection", proj.crs().getInfo(), proj.nominalScale().getInfo())
    task = ee.batch.Export.image.toAsset(image=img, description="prism_q95_1981_2025", assetId=ASSET,
                                         region=ee_io.conus().bounds(), crs=proj.crs().getInfo(),
                                         scale=proj.nominalScale().getInfo(), maxPixels=1e13)
    task.start(); C.log("task started", task.id)
    while True:
        st = task.status(); C.log(st["state"], st.get("error_message", ""))
        if st["state"] in ("COMPLETED", "FAILED", "CANCELLED"): break
        time.sleep(60)
