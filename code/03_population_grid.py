"""
Rasterizes the states on the analysis grid and sums the GlobPOP 2022 population onto it. GlobPOP is the sensitivity layer (the main population layer is built by 03b_census_population.py). Also compares state totals with Census Vintage 2022 estimates. Inputs in the input folder: GlobPOP_Count_30arc_2022_I32.tiff, tl_2022_us_county_conus.gpkg, conus_grid_mask.tif.
"""
import io
import numpy as np, pandas as pd, rasterio, geopandas as gpd, requests
from rasterio.features import rasterize
from scipy import ndimage
import config as C

R = C.INPUT
valid = C.valid_mask()
H, Wd = C.SHAPE
t = C.TRANSFORM

# ---------- states on the PRISM grid and on the population grid ----------
cnt = gpd.read_file(R / "tl_2022_us_county_conus.gpkg")[["STATEFP", "geometry"]].to_crs("EPSG:4326")
st = cnt.dissolve(by="STATEFP").reset_index()
FIPS = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV",
        "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR",
        "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA",
        "53": "WA", "54": "WV", "55": "WI", "56": "WY"}
st["ST"] = st["STATEFP"].map(FIPS)
ids = {s: i + 1 for i, s in enumerate(st["ST"])}
shapes = [(g, ids[s]) for g, s in zip(st.geometry, st["ST"])]
state_prism = rasterize(shapes, out_shape=C.SHAPE, transform=t, fill=0, dtype="int16")

# ---------- population (GlobPOP 2022 counts) ----------
with rasterio.open(R / "GlobPOP_Count_30arc_2022_I32.tiff") as ps:
    left, top = t.c, t.f
    right, bottom = left + Wd * t.a, top + H * t.e
    win = rasterio.windows.from_bounds(left, bottom, right, top, ps.transform).round_offsets().round_lengths()
    pop = ps.read(1, window=win).astype("float64")
    wt = ps.window_transform(win)
    nd = ps.nodata
pop[(pop == nd) | ~np.isfinite(pop) | (pop < 0)] = 0
fh, fw = pop.shape
state_pop_grid = rasterize(shapes, out_shape=(fh, fw), transform=wt, fill=0, dtype="int16")
rows = (np.arange(fh) + 0.5) * wt.e + wt.f
cols = (np.arange(fw) + 0.5) * wt.a + wt.c
rr = np.floor((rows - t.f) / t.e).astype(int)       # PRISM row of each population row
cc = np.floor((cols - t.c) / t.a).astype(int)       # PRISM column of each population column
RR, CC = np.meshgrid(rr, cc, indexing="ij")
inside = (RR >= 0) & (RR < H) & (CC >= 0) & (CC < Wd)
us = state_pop_grid > 0                              # people located inside a CONUS state polygon
RRc, CCc = np.clip(RR, 0, H - 1), np.clip(CC, 0, Wd - 1)


def to_prism(select, redirect=None):
    """Sum population of the selected 30" pixels onto PRISM cells; optionally redirect to other cells."""
    sel = select & inside
    r, c = RR[sel], CC[sel]
    if redirect is not None:
        r, c = redirect(r, c)
    out = np.zeros(C.SHAPE)
    np.add.at(out, (r, c), pop[sel])
    return out


strict = to_prism(us & valid[RRc, CCc])
_, (ir, ic) = ndimage.distance_transform_edt(~valid, return_indices=True)       # nearest valid PRISM cell
nearest = to_prism(us, lambda r, c: (ir[r, c], ic[r, c]))
tot_us = pop[us].sum()
np.save(C.DATA / "pop_globpop2022_grid.npy", nearest.astype("float32"))      # used for the sensitivity comparison
np.save(C.DATA / "state_on_prism_grid.npy", state_prism)
pd.Series(ids).rename("state_id").to_csv(C.OUT / "state_ids.csv")

tot = pd.DataFrame([("People inside CONUS state polygons (GlobPOP 2022)", tot_us / 1e6),
                    ("Strict: people whose cell falls in a valid PRISM cell", strict.sum() / 1e6),
                    ("Nearest: edge people reassigned to nearest valid cell", nearest.sum() / 1e6)], columns=["quantity", "million"])
print(tot.round(2).to_string(index=False))
tot.to_csv(C.OUT / "03_population_totals.csv", index=False)

# ---------- Census Vintage 2022 comparison ----------
by_state = pd.DataFrame({"ST": st["ST"],
                         "globpop_polygon": [pop[state_pop_grid == ids[s]].sum() for s in st["ST"]],
                         "globpop_prism_nearest": [nearest[state_prism == ids[s]].sum() for s in st["ST"]]})
NAME2ST = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
           "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL",
           "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
           "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
           "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
           "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
           "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
           "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
           "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
try:
    u = "https://www2.census.gov/programs-surveys/popest/tables/2020-2022/state/totals/NST-EST2022-POP.xlsx"
    raw = pd.read_excel(io.BytesIO(requests.get(u, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).content), header=None)
    raw = raw[raw[0].astype(str).str.startswith(".")].copy()           # state rows are written ".Alabama"
    cen = pd.DataFrame({"NAME": raw[0].str.lstrip("."), "POPESTIMATE2022": raw.iloc[:, -1].astype(float)})
    cen["ST"] = cen["NAME"].map(NAME2ST)
    cen = cen.dropna(subset=["ST"])
    by_state = by_state.merge(cen[["ST", "POPESTIMATE2022"]], on="ST", how="left")
    by_state["pct_diff_vs_census"] = 100 * (by_state["globpop_polygon"] / by_state["POPESTIMATE2022"] - 1)
    print("Census 2022 CONUS+DC total (M): %.2f" % (by_state["POPESTIMATE2022"].sum() / 1e6))
    print("GlobPOP polygon total (M): %.2f" % (by_state["globpop_polygon"].sum() / 1e6))
    print(by_state.sort_values("pct_diff_vs_census")[["ST", "pct_diff_vs_census"]].round(1).iloc[[0, 1, 2, -3, -2, -1]].to_string(index=False))
except Exception as e:
    print("Census download failed:", repr(e)[:150])
by_state.round(1).to_csv(C.OUT / "03_state_population_vs_census.csv", index=False)

