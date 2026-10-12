"""
Downloads monthly PRISM precipitation totals (mm) for 1981-2025 from Google Earth Engine onto the analysis grid.
The source is the PRISM AN daily collection OREGONSTATE/PRISM/ANd (set as PRISM_ASSET in config.py). ee_io.prism() opens it,
the daily values are summed by month on the Earth Engine side, and ee_io.fetch() downloads the result as a raster.
You need an Earth Engine account and a project: run `earthengine authenticate` once and set EE_PROJECT (see README).
One file per year in the data cache, shape (12, rows, cols). Existing years are skipped.
Usage: python 01_download_monthly.py [workers]
"""
import sys, numpy as np, ee
from concurrent.futures import ThreadPoolExecutor
import config as C, ee_io

def monthly_image(y):
    p = ee_io.prism()
    ims = []
    for m in range(1, 13):
        s = ee.Date.fromYMD(y, m, 1)
        ims.append(p.filterDate(s, s.advance(1, "month")).sum().rename(f"m{m:02d}"))
    return ee.Image.cat(ims)

def one(y):
    f = C.DATA / f"monthly_{y}.npy"
    if f.exists(): return y
    C.log("start", y)
    try:
        a = ee_io.fetch(monthly_image(y), nr=1, nc=3)
        np.save(f, a); C.log("saved", y, a.shape)
    except Exception as e:                      # log and carry on; rerun the script to retry failed years
        C.log("FAILED", y, repr(e)[:300])
    return y

if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ee_io.init()
    with ThreadPoolExecutor(nw) as ex:
        for _ in ex.map(one, C.YEARS): pass
    C.log("all monthly done")
