"""
Splits September-November precipitation in nClimGrid-Daily into days within 500 km of a tropical cyclone (IBTrACS, +-1.5 days)
and all other days, and compares their Sen trends by region.
"""
import os, sys
import numpy as np, pandas as pd, xarray as xr, s3fs, rasterio
from concurrent.futures import ThreadPoolExecutor
import config as C, trend as T
import importlib
dl = importlib.import_module("06_download_nclimgrid")

BUCKET = dl.BUCKET
TMP = C.DATA / "tmp_nclim_tc"
TMP.mkdir(exist_ok=True)
fs = s3fs.S3FileSystem(anon=True)
RADIUS = 500.0
tc = pd.read_pickle(C.DATA / "ibtracs_NA_EP.pkl")
tc = tc.assign(day=(tc.time - pd.Timestamp("1980-01-01")).dt.total_seconds() / 86400.0)


def window_mask(lat2d, lon2d, lat0, lon0, shape):
    """cells within RADIUS km of one point, evaluated only in a bounding window"""
    dlat = RADIUS / 111.0 + 0.1
    dlon = RADIUS / (111.0 * max(np.cos(np.radians(lat0)), 0.2)) + 0.1
    sel = (np.abs(lat2d - lat0) < dlat) & (np.abs(lon2d - lon0) < dlon)
    out = np.zeros(shape, bool)
    if not sel.any():
        return out
    la, lo = np.radians(lat2d[sel]), np.radians(lon2d[sel])
    p0, l0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((la - p0) / 2) ** 2 + np.cos(p0) * np.cos(la) * np.sin((lo - l0) / 2) ** 2
    out[sel] = 2 * 6371.0 * np.arcsin(np.sqrt(a)) <= RADIUS
    return out


def one_year(y):
    f = C.DATA / f"nclim_son_tc_{y}.npz"
    if f.exists():
        return
    try:
        tot = tcs = None
        for m in (9, 10, 11):
            p = TMP / f"ncdd-{y}{m:02d}.nc"
            fs.get(f"{BUCKET}/{y}/ncdd-{y}{m:02d}-grd-scaled.nc", str(p))
            with xr.open_dataset(p) as ds:
                pr = ds["prcp"].values.astype("float32")
                lat, lon = ds["lat"].values, ds["lon"].values
                dates = pd.to_datetime(ds["time"].values)
            os.remove(p)
            lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
            if tot is None:
                tot = np.zeros(pr.shape[1:], "float32")
                tcs = np.zeros(pr.shape[1:], "float32")
            for k, d in enumerate(dates):
                x = (d - pd.Timestamp("1980-01-01")).total_seconds() / 86400.0
                fx = tc[(tc.day >= x - 1.5) & (tc.day <= x + 1.5)]
                day = np.nan_to_num(pr[k], nan=0.0)
                tot += day
                if len(fx):
                    mask = np.zeros(day.shape, bool)
                    for la0, lo0 in zip(fx.LAT.values, fx.LON.values):
                        mask |= window_mask(lat2d, lon2d, la0, lo0, day.shape)
                    tcs += np.where(mask, day, 0.0)
        valid = np.isfinite(pr).any(0)
        ri, ci = dl.index_maps(lat, lon)

        def samp(a):
            ok_r, ok_c = ri >= 0, ci >= 0
            sub = a[np.ix_(np.where(ok_r, ri, 0), np.where(ok_c, ci, 0))].astype("float32")
            sub[~ok_r, :] = np.nan
            sub[:, ~ok_c] = np.nan
            return sub

        np.savez(f, total=samp(np.where(valid, tot, np.nan)), tc=samp(np.where(valid, tcs, np.nan)))
        C.log("saved", y)
    except Exception as e:
        C.log("FAILED", y, repr(e)[:200])


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ThreadPoolExecutor(nw) as ex:
        list(ex.map(one_year, C.YEARS))

    # ---------------- analysis ----------------
    valid = C.valid_mask()
    W = C.lat_weights()
    tot = np.stack([np.load(C.DATA / f"nclim_son_tc_{y}.npz")["total"] for y in C.YEARS])
    tcp = np.stack([np.load(C.DATA / f"nclim_son_tc_{y}.npz")["tc"] for y in C.YEARS])
    ok = valid & np.isfinite(tot).all(0)
    A, B = tot[:, ok], tcp[:, ok]
    N = A - B
    res = {k: T.mk_sen(X) for k, X in (("total", A), ("tc", B), ("nontc", N))}
    ols = {"total": T.ols_slope(A), "tc": T.ols_slope(B), "nontc": T.ols_slope(N)}   # OLS slopes add up exactly: total = tc + non-tc
    for k, r in res.items():
        C.write_tif(C.RASTERS / f"tc_son_{k}_sen.tif", C.to_grid(r["sen"], ok))
    state = np.load(C.DATA / "state_on_prism_grid.npy")[ok]
    ids = pd.read_csv(C.OUT / "state_ids.csv", index_col=0)["state_id"]
    lon = (C.TRANSFORM.c + (np.arange(C.SHAPE[1]) + 0.5) * C.TRANSFORM.a)[None, :].repeat(C.SHAPE[0], 0)[ok]
    groups = {"Atlantic coast states": "FL GA SC NC VA MD DE NJ NY CT RI MA NH ME".split(),
              "Gulf coast states": "TX LA MS AL".split()}
    gm = {k: np.isin(state, [ids[s] for s in v]) for k, v in groups.items()}
    gm["Inland East (east of 100W)"] = (lon > -100) & ~gm["Atlantic coast states"] & ~gm["Gulf coast states"]
    gm["West (west of 100W)"] = lon <= -100
    w = W[ok]
    rows = []
    for name, m in gm.items():
        wm = lambda v: float((w[m] * v[m]).sum() / w[m].sum())
        rows.append(dict(region=name, pct_area=100 * w[m].sum() / w.sum(),
                         mean_son_mm=wm(A.mean(0)), mean_tc_share_pct=100 * wm(B.mean(0)) / wm(A.mean(0)),
                         ols_total_mm_per_decade=10 * wm(ols["total"]), ols_tc_mm_per_decade=10 * wm(ols["tc"]),
                         ols_nontc_mm_per_decade=10 * wm(ols["nontc"]),
                         sen_total_mm_per_decade=10 * wm(res["total"]["sen"]), sen_tc_mm_per_decade=10 * wm(res["tc"]["sen"]),
                         sen_nontc_mm_per_decade=10 * wm(res["nontc"]["sen"]),
                         pct_area_total_trend_positive=100 * w[m & (res["total"]["sen"] > 0)].sum() / w[m].sum(),
                         pct_area_nontc_trend_positive=100 * w[m & (res["nontc"]["sen"] > 0)].sum() / w[m].sum()))
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "21_tc_autumn_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    with rasterio.open(C.RASTERS / "son_sen_1981_2025.tif") as s:
        pr_son = s.read(1)[ok]
    print("corr of nClimGrid SON Sen with PRISM SON Sen: %.2f" % np.corrcoef(res["total"]["sen"], pr_son)[0, 1])
    C.log("done")
