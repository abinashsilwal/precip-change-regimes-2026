"""
Population from the 2020 Decennial Census (PL 94-171 redistricting files, census block level), summed onto the analysis grid.
Each block's resident population (POP100) is placed at the block's internal point. People are conserved: 329,260,619 for the 48
states and DC. Internal points that fall in a cell outside the CONUS mask (coast, border) are given to the nearest CONUS cell.
Outputs: data_cache/pop_on_prism_grid.npy (this becomes the population layer used by all later scripts),
         outputs/03b_census_population_by_state.csv
The GlobPOP layer from 03_population_grid.py is kept as data_cache/pop_globpop2022_grid.npy for the sensitivity comparison.
"""
import io, zipfile, concurrent.futures as cf
import numpy as np, pandas as pd, requests
from scipy import ndimage
import config as C

BASE = "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/"
STATES = {"AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
          "DC": "District_of_Columbia", "FL": "Florida", "GA": "Georgia", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
          "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
          "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New_Hampshire",
          "NJ": "New_Jersey", "NM": "New_Mexico", "NY": "New_York", "NC": "North_Carolina", "ND": "North_Dakota", "OH": "Ohio",
          "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode_Island", "SC": "South_Carolina", "SD": "South_Dakota",
          "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West_Virginia",
          "WI": "Wisconsin", "WY": "Wyoming"}
CACHE = C.DATA / "census_pl"
CACHE.mkdir(exist_ok=True)
SUMLEV, POP100, LAT, LON = 2, 90, 92, 93      # columns of the geographic header file


def one(st):
    f = CACHE / f"{st}.npz"
    if f.exists():
        z = np.load(f)
        return st, z["lat"], z["lon"], z["pop"]
    url = f"{BASE}{STATES[st]}/{st.lower()}2020.pl.zip"
    for k in range(4):
        try:
            r = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            break
        except Exception:
            if k == 3:
                raise
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = [n for n in z.namelist() if "geo" in n][0]
    rows = [l.split("|") for l in z.read(name).decode("latin1").split("\n") if l[:7] == f"PLST|{st}"]
    rows = [x for x in rows if x[SUMLEV] == "750"]
    lat = np.array([float(x[LAT]) for x in rows])
    lon = np.array([float(x[LON]) for x in rows])
    pop = np.array([int(x[POP100]) for x in rows])
    np.savez(f, lat=lat, lon=lon, pop=pop)
    return st, lat, lon, pop


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, STATES))
    valid = C.valid_mask()
    H, W = C.SHAPE
    t = C.TRANSFORM
    _, (ir, ic) = ndimage.distance_transform_edt(~valid, return_indices=True)
    grid = np.zeros(C.SHAPE)
    rows = []
    for st, lat, lon, pop in res:
        r = np.floor((lat - t.f) / t.e).astype(int)
        c = np.floor((lon - t.c) / t.a).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c, p = r[ok], c[ok], pop[ok]
        r2, c2 = ir[r, c], ic[r, c]
        np.add.at(grid, (r2, c2), p)
        rows.append(dict(state=st, blocks=len(pop), population_2020=int(pop.sum()), placed_on_grid=int(p.sum()),
                         placed_outside_mask_then_moved=int(p[~valid[r, c]].sum())))
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "03b_census_population_by_state.csv", index=False)
    print("blocks:", int(df.blocks.sum()), " total 2020 population (48 states + DC): %d" % df.population_2020.sum())
    print("placed on grid: %d, of which %d fell outside the CONUS mask and were moved to the nearest cell" %
          (df.placed_on_grid.sum(), df.placed_outside_mask_then_moved.sum()))
    np.save(C.DATA / "pop_on_prism_grid.npy", grid.astype("float32"))
    C.log("done")
