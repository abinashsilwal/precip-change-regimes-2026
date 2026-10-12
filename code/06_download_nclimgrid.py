"""
Streams the monthly nClimGrid-Daily files from the NOAA open-data bucket, builds the annual indices for each year
(monthly totals, annual total, Rx1day, Rx5day, wet days, R10mm, R20mm) and samples them at the centres of the analysis grid.
One npz per year in the data cache. Downloaded files are deleted after use.
Usage: python 06_download_nclimgrid.py [workers]
"""
import sys, os
import numpy as np, xarray as xr, s3fs
from concurrent.futures import ThreadPoolExecutor
import config as C

BUCKET = "noaa-nclimgrid-daily-pds/access/grids"
TMP = C.DATA / "tmp_nclim"
TMP.mkdir(exist_ok=True)
fs = s3fs.S3FileSystem(anon=True)


def index_maps(lat, lon):
    """nearest native-cell index of every paper-grid cell centre; -1 outside the native grid"""
    H, W = C.SHAPE
    t = C.TRANSFORM
    lat_c = t.f + (np.arange(H) + 0.5) * t.e
    lon_c = t.c + (np.arange(W) + 0.5) * t.a
    d = 1 / 24
    ri = np.rint((lat_c - lat[0]) / d).astype(int)
    ci = np.rint((lon_c - lon[0]) / d).astype(int)
    ri[(ri < 0) | (ri >= len(lat))] = -1
    ci[(ci < 0) | (ci >= len(lon))] = -1
    return ri, ci


def rolling5_max(x):
    """max over all 5-day windows inside the year; x shape (days, lat, lon) with NaN allowed"""
    s = np.zeros_like(x[:len(x) - 4])
    for k in range(5):
        s += x[k:len(x) - 4 + k]
    return np.nanmax(s, axis=0)


def one_year(y):
    f = C.DATA / f"nclim_{y}.npz"
    if f.exists():
        return
    C.log("nclim start", y)
    try:
        days, monthly = [], []
        for m in range(1, 13):
            p = TMP / f"ncdd-{y}{m:02d}.nc"
            fs.get(f"{BUCKET}/{y}/ncdd-{y}{m:02d}-grd-scaled.nc", str(p))
            with xr.open_dataset(p) as ds:
                pr = ds["prcp"].values.astype("float32")
                lat, lon = ds["lat"].values, ds["lon"].values
            os.remove(p)
            days.append(pr)
            monthly.append(np.nansum(pr, axis=0))
        x = np.concatenate(days)
        valid = np.isfinite(x).any(axis=0)
        ri, ci = index_maps(lat, lon)

        def samp(a):
            out = np.full(C.SHAPE, np.nan, "float32")
            ok_r, ok_c = ri >= 0, ci >= 0
            sub = a[np.ix_(np.where(ok_r, ri, 0), np.where(ok_c, ci, 0))]
            sub[~ok_r, :] = np.nan
            sub[:, ~ok_c] = np.nan
            return sub.astype("float32")

        masked = lambda a: np.where(valid, a, np.nan)
        res = dict(monthly=np.stack([samp(masked(m)) for m in monthly]),
                   annual=samp(masked(np.nansum(x, axis=0))),
                   rx1=samp(np.nanmax(x, axis=0)),
                   rx5=samp(rolling5_max(np.nan_to_num(x, nan=0.0)).astype("float32") * np.where(valid, 1, np.nan)),
                   wet=samp(masked((x >= 1).sum(0).astype("float32"))),
                   r10=samp(masked((x >= 10).sum(0).astype("float32"))),
                   r20=samp(masked((x >= 20).sum(0).astype("float32"))))
        np.savez(f, **res)
        C.log("nclim saved", y)
    except Exception as e:
        C.log("nclim FAILED", y, repr(e)[:300])


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with ThreadPoolExecutor(nw) as ex:
        list(ex.map(one_year, C.YEARS))
    C.log("all nclim done")
