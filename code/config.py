"""Paths and constants shared by all scripts."""
import os
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine

HERE = Path(__file__).resolve().parent
# Large intermediate files (monthly and extreme-index cubes, nClimGrid yearly files). Set PRECIP_DATA to put them elsewhere.
DATA = Path(os.environ.get("PRECIP_DATA", HERE.parent / "data_cache"))
# Source files that are downloaded by hand (see README): GlobPOP raster, county boundaries, grid mask.
INPUT = Path(os.environ.get("PRECIP_INPUT", HERE.parent / "input"))
OUT = HERE.parent / "outputs"
RASTERS = DATA / "rasters"
for p in (DATA, OUT, RASTERS):
    p.mkdir(parents=True, exist_ok=True)

EE_PROJECT = os.environ.get("EE_PROJECT", "your-earth-engine-project")
PRISM_ASSET = "OREGONSTATE/PRISM/ANd"
YEARS = list(range(1981, 2026))
WET_DAY_MM = 1.0

# Analysis grid: EPSG:4326, 4000 m expressed in degrees, 1614 columns by 697 rows
TRANSFORM = Affine(0.03593261136478086, 0.0, -124.8658244926135, 0.0, -0.03593261136478086, 49.407340626573685)
SHAPE = (697, 1614)
CRS = rasterio.crs.CRS.from_epsg(4326)
PROFILE = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=1, dtype="float32", crs=CRS, transform=TRANSFORM,
               nodata=np.nan, compress="lzw")


def lat_weights():
    """cos(latitude) weights on the grid, shape SHAPE"""
    lats = TRANSFORM.f + (np.arange(SHAPE[0]) + 0.5) * TRANSFORM.e
    return np.repeat(np.cos(np.deg2rad(lats))[:, None], SHAPE[1], axis=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def valid_mask():
    """Boolean mask of CONUS cells (645,556 cells), read from input/conus_grid_mask.tif"""
    with rasterio.open(INPUT / "conus_grid_mask.tif") as s:
        return s.read(1) > 0


def to_grid(vec, valid):
    out = np.full(SHAPE, np.nan, "float32")
    out[valid] = vec
    return out


def write_tif(path, arr):
    with rasterio.open(path, "w", **PROFILE) as d:
        d.write(arr.astype("float32"), 1)


EXTREME_BANDS = {"basic": ["rx1", "rx5", "r10", "r20", "wet"], "q95": ["r95p_wet", "n95_wet", "r95p_all", "n95_all"], "doy": ["rx1_doy"]}


def load_extremes(y, names=None):
    """dict band name -> (rows, cols) array for year y, merging the 'basic', 'q95' and 'doy' files that exist"""
    out = {}
    for mode, bands in EXTREME_BANDS.items():
        f = DATA / f"extremes_{mode}_{y}.npy"
        if f.exists():
            a = np.load(f)
            out.update({b: a[i] for i, b in enumerate(bands)})
    return out if names is None else {k: out[k] for k in names}
