# Input files

Put these files here before running the scripts.

| file | source |
|---|---|
| `conus_grid_mask.tif` | grid mask of the 645,556 CONUS cells (1 = cell used). Included in the Zenodo archive under `grid/`. |
| `GlobPOP_Count_30arc_2022_I32.tiff` | GlobPOP 2022 population count, release 2.0, https://doi.org/10.5281/zenodo.11179644 (only for the sensitivity layer) |
| `tl_2022_us_county_conus.gpkg` | 2022 TIGER/Line county boundaries for the lower 48 states and DC (U.S. Census Bureau), saved as one GeoPackage |

The other inputs are read directly from their sources: PRISM through Earth Engine (`OREGONSTATE/PRISM/ANd`), nClimGrid-Daily and
the 2020 Census PL 94-171 files from census.gov (downloaded by `03b_census_population.py`), GHCN-Daily from the NOAA open-data buckets on AWS, IBTrACS v04r01 from NOAA NCEI, the Census Vintage 2022 state estimates from census.gov,
and the HUC2 regions from Earth Engine (`USGS/WBD/2017/HUC02`).
