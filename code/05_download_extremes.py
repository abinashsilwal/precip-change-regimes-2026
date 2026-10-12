"""
Yearly extreme-precipitation indices from PRISM in Earth Engine, on the analysis grid.
  basic: rx1, rx5, r10, r20, wet (days >= 1 mm)
  q95:   r95p_wet, n95_wet, r95p_all, n95_all (uses the asset from 02_export_q95_asset.py)
  doy:   rx1_doy, day of the year of the annual maximum
Usage: python 05_download_extremes.py basic|q95|doy [workers] [asset_name]
"""
import sys, numpy as np, ee
from concurrent.futures import ThreadPoolExecutor
import config as C, ee_io



def basic_image(y):
    start = ee.Date.fromYMD(y, 1, 1)
    yc = ee_io.prism().filterDate(start, start.advance(1, "year"))
    nd = yc.size()
    wins = ee.List.sequence(0, nd.subtract(5)).map(
        lambda i: yc.filterDate(start.advance(i, "day"), start.advance(ee.Number(i).add(5), "day")).sum())
    cnt = lambda f: yc.map(lambda im: f(im)).sum()
    return ee.Image.cat([yc.max().rename("rx1"), ee.ImageCollection(wins).max().rename("rx5"),
                         cnt(lambda im: im.gte(10)).rename("r10"), cnt(lambda im: im.gte(20)).rename("r20"),
                         cnt(lambda im: im.gte(C.WET_DAY_MM)).rename("wet")])


def doy_image(y):
    start = ee.Date.fromYMD(y, 1, 1)
    yc = ee_io.prism().filterDate(start, start.advance(1, "year"))
    def add_doy(im):
        d = ee.Date(im.get("system:time_start")).getRelative("day", "year").add(1)
        return im.addBands(ee.Image.constant(d).toFloat().rename("doy"))
    return yc.map(add_doy).reduce(ee.Reducer.max(2)).select(["max1"]).rename("rx1_doy")


def q95_image(y, asset):
    q = ee.Image(asset)
    qw, qa = q.select("q95_wet"), q.select("q95_all")
    start = ee.Date.fromYMD(y, 1, 1)
    yc = ee_io.prism().filterDate(start, start.advance(1, "year"))
    cnt = lambda f: yc.map(lambda im: f(im)).sum()
    return ee.Image.cat([yc.map(lambda im: im.multiply(im.gt(qw))).sum().rename("r95p_wet"), cnt(lambda im: im.gt(qw)).rename("n95_wet"),
                         yc.map(lambda im: im.multiply(im.gt(qa))).sum().rename("r95p_all"), cnt(lambda im: im.gt(qa)).rename("n95_all")])


def one(y, mode, asset):
    f = C.DATA / f"extremes_{mode}_{y}.npy"
    if f.exists():
        return
    C.log("extremes", mode, "start", y)
    try:
        img = {"basic": lambda: basic_image(y), "doy": lambda: doy_image(y), "q95": lambda: q95_image(y, asset)}[mode]()
        a = ee_io.fetch(img, nr=1, nc=3)
        np.save(f, a)
        C.log("extremes", mode, "saved", y, a.shape)
    except Exception as e:
        C.log("extremes", mode, "FAILED", y, repr(e)[:300])


if __name__ == "__main__":
    mode = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    asset = f"projects/{C.EE_PROJECT}/assets/" + (sys.argv[3] if len(sys.argv) > 3 else "prism_q95_1981_2025")
    ee_io.init()
    with ThreadPoolExecutor(nw) as ex:
        list(ex.map(lambda y: one(y, mode, asset), C.YEARS))
    C.log("all extremes", mode, "done")
