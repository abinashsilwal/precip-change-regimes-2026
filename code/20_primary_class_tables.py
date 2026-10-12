"""
Class tables for the relative and absolute classifications (area, population, within-state null, FDR significance), and
the agreement between PRISM and nClimGrid-Daily on wetting with stronger extremes.
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
W = C.lat_weights()[valid]
pop = np.load(C.DATA / "pop_on_prism_grid.npy")[valid].astype("float64")
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
rd = lambda n: rasterio.open(C.RASTERS / n).read(1)[valid]
fa, fr = rd("annual_fdr_1981_2025.tif") > 0, rd("rx1_fdr_1981_2025.tif") > 0
sigcat = np.where(fa & fr, "both", np.where(fa | fr, "one", "neither"))
NAMES = {1: "Drying + weaker extremes", 2: "Drying + stronger extremes", 3: "Wetting + weaker extremes",
         4: "Wetting + stronger extremes", 5: "Weak or mixed signal"}
sid = np.unique(state[state > 0])


def null_share(m):
    return 100 * sum(pop[state == s].sum() * W[(state == s) & m].sum() / W[state == s].sum() for s in sid) / pop.sum()


rows = []
for tag in ("rel", "abs"):
    cl = rd(f"class5_{tag}_p50.tif")
    for c in range(1, 6):
        m = cl == c
        r = dict(classification=tag, cls=NAMES[c], area_pct=100 * W[m].sum() / W.sum(), pop_M=pop[m].sum() / 1e6,
                 pop_pct=100 * pop[m].sum() / pop.sum(), null_pct=null_share(m))
        for s in ("both", "one", "neither"):
            r[f"sig_{s}_pct_conus"] = 100 * W[m & (sigcat == s)].sum() / W.sum()
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "20_class_tables.csv", index=False)
print(df.round(2).to_string(index=False))

# ---- agreement with nClimGrid-Daily (relative classification, own medians) ----
cm = np.zeros(C.SHAPE, bool)
nc_a = np.stack([np.load(C.DATA / f"nclim_{y}.npz")["annual"] for y in C.YEARS])
nc_r = np.stack([np.load(C.DATA / f"nclim_{y}.npz")["rx1"] for y in C.YEARS])
common = np.isfinite(nc_a).all(0) & np.isfinite(nc_r).all(0) & C.valid_mask()
sa = rasterio.open(C.RASTERS / "cmp_nclim_annual_sen.tif").read(1)
sr = rasterio.open(C.RASTERS / "cmp_nclim_rx1_sen.tif").read(1)
ra, rr = 1000 * sa / nc_a.mean(0), 1000 * sr / nc_r.mean(0)
ta, tr = np.nanpercentile(np.abs(ra[common]), 50), np.nanpercentile(np.abs(rr[common]), 50)
nclim_head = common & (ra > ta) & (rr > tr)
nclim_dry = common & (ra < -ta)
# PRISM relative classification on the same cells, thresholds from the same cells for a fair comparison
pa = np.full(C.SHAPE, np.nan, "float32"); pr = pa.copy()
pa[valid] = 1000 * rd("annual_sen_1981_2025.tif") / np.load(C.DATA / "cube_annual.npy").mean(0)
pr[valid] = 1000 * rd("rx1_sen_1981_2025.tif") / np.load(C.DATA / "cube_rx1.npy").mean(0)
tpa, tpr = np.nanpercentile(np.abs(pa[common]), 50), np.nanpercentile(np.abs(pr[common]), 50)
prism_head = common & (pa > tpa) & (pr > tpr)
Wg = C.lat_weights()
popg = np.load(C.DATA / "pop_on_prism_grid.npy")
st = np.load(C.DATA / "state_on_prism_grid.npy")
rows = []
for name, m in (("PRISM wetting + stronger (relative, common cells)", prism_head), ("nClimGrid-Daily wetting + stronger (relative)", nclim_head),
                ("Both products", prism_head & nclim_head), ("PRISM only", prism_head & ~nclim_head), ("nClimGrid only", nclim_head & ~prism_head)):
    nul = 100 * sum(popg[(st == s) & common].sum() * Wg[(st == s) & m].sum() / Wg[(st == s) & common].sum() for s in sid if (common & (st == s)).any()) / popg[common].sum()
    rows.append(dict(set=name, area_pct=100 * Wg[m].sum() / Wg[common].sum(), pop_pct=100 * popg[m].sum() / popg[common].sum(), null_pct=nul))
ag = pd.DataFrame(rows)
ag["share_of_prism_class_retained_pct"] = 100 * ag.area_pct / ag.area_pct.iloc[0]
ag.to_csv(C.OUT / "20_agreement.csv", index=False)
print(ag.round(2).to_string(index=False))
print("thresholds (relative %/dec): PRISM", tpa, tpr, " nClimGrid", ta, tr)

ag_r = np.full(C.SHAPE, np.nan, "float32")
ag_r[common] = 0
ag_r[prism_head & ~nclim_head] = 1
ag_r[nclim_head & ~prism_head] = 2
ag_r[prism_head & nclim_head] = 3
C.write_tif(C.RASTERS / "agree_wet_strong_rel.tif", ag_r)
