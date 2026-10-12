"""
Compares PRISM and nClimGrid-Daily with long-record COOP stations from GHCN-Daily.
A station is used if it has at least 38 years with 330 or more valid days (at least 17 before and 17 from 2002).
For each station the annual total and Rx1day are compared with the values of the grid cell that contains it.
"""
import gzip, io, concurrent.futures as cf
import numpy as np, pandas as pd, s3fs
from scipy.stats import mannwhitneyu
import config as C, trend as T

fs = s3fs.S3FileSystem(anon=True)
B = "noaa-ghcn-pds"
CACHE = C.DATA / "ghcn"
CACHE.mkdir(exist_ok=True)
yrs = np.array(C.YEARS)
n = len(yrs)
H, W = C.SHAPE
t = C.TRANSFORM

# ---------------- station selection ----------------
inv_f = CACHE / "ghcnd-inventory.txt"
if not inv_f.exists():
    fs.get(f"{B}/ghcnd-inventory.txt", str(inv_f))
inv = pd.read_fwf(inv_f, colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
                  names=["id", "lat", "lon", "el", "y0", "y1"])
inv = inv[(inv.el == "PRCP") & inv.id.str.startswith("USC") & (inv.y0 <= 1981) & (inv.y1 >= 2025)]
north, south = t.f, t.f + H * t.e
west, east = t.c, t.c + W * t.a
inv = inv[(inv.lat < north) & (inv.lat > south) & (inv.lon > west) & (inv.lon < east)].reset_index(drop=True)
print("candidate COOP stations:", len(inv), flush=True)


def load(sid):
    f = CACHE / f"{sid}.npz"
    if f.exists():
        z = np.load(f)
        return sid, z["ann"], z["rx1"], z["ok"]
    try:
        raw = fs.cat(f"{B}/csv.gz/by_station/{sid}.csv.gz")
        d = pd.read_csv(io.BytesIO(gzip.decompress(raw)), header=None, usecols=[1, 2, 3, 5],
                        names=["date", "el", "v", "q"], dtype={"date": str, "el": str, "q": str})
        d = d[(d.el == "PRCP") & d.q.isna() & d.date.str[:4].astype(int).between(1981, 2025)]
        d["y"] = d.date.str[:4].astype(int)
        d["mm"] = d.v / 10.0
        g = d.groupby("y").mm
        cnt, tot, mx = g.count().reindex(yrs), g.sum().reindex(yrs), g.max().reindex(yrs)
        ok = (cnt >= 330).values
        ann = np.where(ok, tot.values, np.nan)
        rx1 = np.where(ok, mx.values, np.nan)
    except Exception:
        ann = rx1 = np.full(n, np.nan)
        ok = np.zeros(n, bool)
    np.savez(f, ann=ann, rx1=rx1, ok=ok)
    return sid, ann, rx1, ok


rows = []
with cf.ThreadPoolExecutor(16) as ex:
    for i, (sid, ann, rx1, ok) in enumerate(ex.map(load, inv.id)):
        rows.append((sid, ann, rx1, ok))
        if i % 500 == 0:
            print("loaded", i, flush=True)
keep = []
for sid, ann, rx1, ok in rows:
    if ok.sum() >= 38 and ok[yrs < 2002].sum() >= 17 and ok[yrs >= 2002].sum() >= 17:
        keep.append((sid, ann, rx1))
print("stations passing completeness:", len(keep), flush=True)
meta = inv.set_index("id")

# ---------------- gridded values at station cells ----------------
valid = C.valid_mask()
pidx = -np.ones((H, W), int)
pidx[valid] = np.arange(valid.sum())
A_p = np.load(C.DATA / "cube_annual.npy")          # (n, P)
R_p = np.load(C.DATA / "cube_rx1.npy")
A_n = np.stack([np.load(C.DATA / f"nclim_{y}.npz")["annual"] for y in C.YEARS])
R_n = np.stack([np.load(C.DATA / f"nclim_{y}.npz")["rx1"] for y in C.YEARS])
recs = []
for sid, ann, rx1 in keep:
    la, lo = meta.loc[sid, "lat"], meta.loc[sid, "lon"]
    r = int((la - t.f) / t.e)
    c = int((lo - t.c) / t.a)
    if not (0 <= r < H and 0 <= c < W) or pidx[r, c] < 0 or not np.isfinite(A_n[:, r, c]).all():
        continue
    recs.append(dict(id=sid, lat=la, lon=lo, r=r, c=c, ann=ann, rx1=rx1, pa=A_p[:, pidx[r, c]], pr=R_p[:, pidx[r, c]], na=A_n[:, r, c], nr=R_n[:, r, c]))
print("stations with valid PRISM and nClimGrid cells:", len(recs), flush=True)

pre = yrs <= 2001
post = yrs >= 2002
out = []
for s in recs:
    for var, st, ps, ns in (("annual", s["ann"], s["pa"], s["na"]), ("rx1", s["rx1"], s["pr"], s["nr"])):
        ok = np.isfinite(st)
        if ok.sum() < 38:
            continue
        dp, dn = ps - st, ns - st
        mean_st = np.nanmean(st)
        rec = dict(id=s["id"], lat=s["lat"], lon=s["lon"], east=s["lon"] > -100, var=var, mean_station=mean_st)
        for lab, d in (("prism", dp), ("nclim", dn)):
            a, b = d[ok & pre], d[ok & post]
            rec[f"step_{lab}"] = b.mean() - a.mean()
            rec[f"step_{lab}_pct"] = 100 * (b.mean() - a.mean()) / mean_st
            rec[f"mwu_p_{lab}"] = mannwhitneyu(a, b).pvalue
        yy = np.where(ok)[0]
        X = np.column_stack([st[ok], ps[ok], ns[ok]])
        sen = [T.mk_sen(X[:, [k]])["sen"][0] for k in range(3)]
        rec.update(sen_station=sen[0], sen_prism=sen[1], sen_nclim=sen[2])
        out.append(rec)
df = pd.DataFrame(out)
df.to_csv(C.OUT / "19_station_level.csv", index=False)

rows = []
for var in ("annual", "rx1"):
    for reg in ("CONUS", "East of 100W", "West of 100W"):
        d = df[(df["var"] == var) & (df.east if reg.startswith("East") else (~df.east if reg.startswith("West") else True))]
        rows.append(dict(var=var, region=reg, n_stations=len(d),
                         median_step_prism_pct=d.step_prism_pct.median(), median_step_nclim_pct=d.step_nclim_pct.median(),
                         pct_stations_sig_step_prism=100 * (d.mwu_p_prism < 0.05).mean(), pct_stations_sig_step_nclim=100 * (d.mwu_p_nclim < 0.05).mean(),
                         median_sen_station=d.sen_station.median(), median_sen_prism=d.sen_prism.median(), median_sen_nclim=d.sen_nclim.median(),
                         corr_sen_prism_station=np.corrcoef(d.sen_station, d.sen_prism)[0, 1], corr_sen_nclim_station=np.corrcoef(d.sen_station, d.sen_nclim)[0, 1]))
sm = pd.DataFrame(rows)
sm.to_csv(C.OUT / "19_station_summary.csv", index=False)
pd.set_option("display.width", 250)
print(sm.round(3).to_string(index=False))

C.log("done")

# ---- trends in percent of the station mean per decade, at the station cells ----
df["pct_dec_station"] = 1000 * df.sen_station / df.mean_station
df["pct_dec_prism"] = 1000 * df.sen_prism / df.mean_station
df["pct_dec_nclim"] = 1000 * df.sen_nclim / df.mean_station
rows = []
for var in ("annual", "rx1"):
    for reg in ("CONUS", "East of 100W", "West of 100W"):
        d = df[(df["var"] == var) & (df.east if reg.startswith("East") else (~df.east if reg.startswith("West") else True))]
        rows.append(dict(var=var, region=reg, n=len(d), median_pct_per_decade_station=d.pct_dec_station.median(), median_pct_per_decade_prism=d.pct_dec_prism.median(),
                         median_pct_per_decade_nclim=d.pct_dec_nclim.median(), mean_pct_per_decade_station=d.pct_dec_station.mean(),
                         mean_pct_per_decade_prism=d.pct_dec_prism.mean(), mean_pct_per_decade_nclim=d.pct_dec_nclim.mean()))
pd.DataFrame(rows).to_csv(C.OUT / "19_station_pct_per_decade.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
