"""
Matches the day of the annual Rx1day maximum of each pixel with IBTrACS storm fixes (within 500 km, +-1 day).
"""
import io, requests
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
import config as C, trend as T

RADII = (300, 500)
URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.{b}.list.v04r01.csv"

def load_ibtracs():
    f = C.DATA / "ibtracs_NA_EP.pkl"
    if f.exists():
        return pd.read_pickle(f)
    frames = []
    for b in ("NA", "EP"):
        txt = requests.get(URL.format(b=b), timeout=300).text
        d = pd.read_csv(io.StringIO(txt), skiprows=[1], usecols=["SID", "ISO_TIME", "NATURE", "LAT", "LON"], low_memory=False)
        d["basin"] = b
        frames.append(d)
    d = pd.concat(frames)
    d["time"] = pd.to_datetime(d["ISO_TIME"])
    d = d[["SID", "time", "NATURE", "LAT", "LON", "basin"]].dropna()
    d["LAT"], d["LON"] = d.LAT.astype(float), d.LON.astype(float)
    d.to_pickle(f)
    return d

valid = C.valid_mask()
rr, cc = np.where(valid)
lat = C.TRANSFORM.f + (rr + 0.5) * C.TRANSFORM.e
lon = C.TRANSFORM.c + (cc + 0.5) * C.TRANSFORM.a
W = C.lat_weights()[valid]
n = len(C.YEARS)
tc = load_ibtracs()
print("IBTrACS fixes:", len(tc), "storms:", tc.SID.nunique())
doy = np.stack([np.load(C.DATA / f"extremes_doy_{y}.npy")[0][valid] for y in C.YEARS]).astype(int)   # (n, P)

flag = {R: np.zeros((n, valid.sum()), bool) for R in RADII}
pix = np.radians(np.c_[lat, lon])
for i, y in enumerate(C.YEARS):
    sub = tc[(tc.time >= f"{y - 1}-12-30") & (tc.time <= f"{y + 1}-01-02")]
    t_days = (sub.time - pd.Timestamp(f"{y}-01-01")).dt.total_seconds().values / 86400.0          # days since Jan 1 00 UTC
    xy = np.radians(sub[["LAT", "LON"]].values)
    for d in np.unique(doy[i]):
        m = doy[i] == d
        # PRISM day d covers 12 UTC on day d-1 to 12 UTC on day d  ->  days since Jan 1: [d-1.5, d-0.5]; add +-1 day tolerance
        sel = (t_days >= d - 2.5) & (t_days <= d + 0.5)
        if not sel.any():
            continue
        bt = BallTree(xy[sel], metric="haversine")
        dist = bt.query(pix[m], k=1)[0][:, 0] * 6371.0
        for R in RADII:
            flag[R][i, m] = dist <= R
    print(y, {R: round(100 * flag[R][i].mean(), 1) for R in RADII}, flush=True)
for R in RADII:
    np.save(C.DATA / f"tc_flag_{R}km.npy", flag[R])
    C.write_tif(C.RASTERS / f"tc_fraction_{R}km.tif", C.to_grid(flag[R].mean(0), valid))

# ---- summaries ----
yrs = np.array(C.YEARS)
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
ids = pd.read_csv(C.OUT / "state_ids.csv", index_col=0)["state_id"]
GULF_ATL = ["TX", "LA", "MS", "AL", "FL", "GA", "SC", "NC", "VA"]
rows = []
for R in RADII:
    f = flag[R]
    share = lambda m, sel=slice(None): 100 * (W[m] * f[sel][:, m].mean(0)).sum() / W[m].sum()
    allm = np.ones(valid.sum(), bool)
    coast = np.isin(state, [ids[s] for s in GULF_ATL if s in ids.index])
    for name, m in (("CONUS", allm), ("Gulf+South Atlantic states", coast), ("rest of CONUS", ~coast)):
        rows.append(dict(radius_km=R, region=name, pct_events_tc_all=share(m),
                         pct_events_tc_1981_2002=share(m, yrs <= 2002), pct_events_tc_2003_2025=share(m, yrs >= 2003)))
    # trend in the TC-associated share (area-weighted mean flag per year)
    ser = np.array([(W * f[i]).sum() / W.sum() for i in range(n)])
    r = T.mk_sen(ser[:, None])
    rows.append(dict(radius_km=R, region="CONUS: trend of TC share per decade (percentage points)", pct_events_tc_all=1000 * r["sen"][0], pct_events_tc_1981_2002=r["p"][0]))
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "12_tc_attribution_summary.csv", index=False)
print(df.round(2).to_string(index=False))

# Rx1day trend with and without TC-associated years: replace TC years by the series median of non-TC years (a rough but easy-to-check approach)
R1 = np.load(C.DATA / "cube_rx1.npy")
f = flag[500]
sen_all = T.mk_sen(R1)["sen"]
Rx_fill = np.where(f, np.nan, R1)
# pixels with at least 20 non-TC years: Sen slope of the remaining years (n varies, handled by pairwise slopes)
def sen_masked(X):
    n_, P = X.shape
    iu, ju = np.triu_indices(n_, 1)
    out = np.full(P, np.nan, "float32")
    for a in range(0, P, 10000):
        x = X[:, a:a + 10000]
        d = (x[ju] - x[iu]) / (ju - iu)[:, None]
        out[a:a + 10000] = np.nanmedian(d, axis=0)
    return out
sen_ex = sen_masked(Rx_fill)
ok = np.isfinite(sen_ex) & (np.isfinite(Rx_fill).sum(0) >= 20)
tcfrac = f.mean(0)
rows = []
for name, m in (("TC share of events >= 20%", tcfrac >= 0.20), ("TC share < 20%", tcfrac < 0.20), ("all", np.ones_like(ok))):
    mm = ok & m
    rows.append(dict(group=name, pct_area=100 * W[mm].sum() / W.sum(), mean_sen_all_years=float((W[mm] * sen_all[mm]).sum() / W[mm].sum()),
                     mean_sen_excluding_tc_years=float((W[mm] * sen_ex[mm]).sum() / W[mm].sum())))
d2 = pd.DataFrame(rows)
d2.to_csv(C.OUT / "12_rx1_trend_with_and_without_tc_years.csv", index=False)
print(d2.round(3).to_string(index=False))
C.log("done")
