"""
Compares the exposure results for two population layers: the 2020 Census block counts (primary) and GlobPOP 2022 (modelled, global).
For each regime classification (relative, absolute) it gives the area, the population share and the within-state area null of
every class, and the share of CONUS population in the main class under each layer.
Output: outputs/23_population_sensitivity.csv
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
W = C.lat_weights()[valid]
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
sid = np.unique(state[state > 0])
layers = {"census2020": np.load(C.DATA / "pop_on_prism_grid.npy")[valid].astype("float64"),
          "globpop2022": np.load(C.DATA / "pop_globpop2022_grid.npy")[valid].astype("float64")}
NAMES = {1: "Drying + weaker extremes", 2: "Drying + stronger extremes", 3: "Wetting + weaker extremes",
         4: "Wetting + stronger extremes", 5: "Weak or mixed signal"}
rows = []
for tag in ("rel", "abs"):
    cl = rasterio.open(C.RASTERS / f"class5_{tag}_p50.tif").read(1)[valid]
    for lname, pop in layers.items():
        for c in range(1, 6):
            m = cl == c
            null = sum(pop[state == s].sum() * W[(state == s) & m].sum() / W[state == s].sum() for s in sid)
            rows.append(dict(classification=tag, population=lname, cls=NAMES[c], area_pct=100 * W[m].sum() / W.sum(),
                             pop_million=pop[m].sum() / 1e6, pop_pct=100 * pop[m].sum() / pop.sum(), null_pct=100 * null / pop.sum(),
                             ratio_pop_to_area=(pop[m].sum() / pop.sum()) / (W[m].sum() / W.sum())))
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "23_population_sensitivity.csv", index=False)
pd.set_option("display.width", 200)
print(df[df.cls == "Wetting + stronger extremes"].round(2).to_string(index=False))
print("totals (million):", {k: round(v.sum() / 1e6, 2) for k, v in layers.items()})
# correlation of the two layers at the cell level and by state
a, b = layers["census2020"], layers["globpop2022"]
print("cell-level correlation (log1p): %.3f" % np.corrcoef(np.log1p(a), np.log1p(b))[0, 1])
