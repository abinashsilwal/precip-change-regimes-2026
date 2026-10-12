# Joint trends in annual and extreme precipitation across the conterminous United States, 1981-2025

Analysis code for "Hydroclimatic change regimes from joint annual and extreme precipitation trends and population exposure across the
conterminous United States" (Silwal, Gautam, Parajuli, Regmi, Tamrakar, Thapa, Zhran, Dahal).
The derived data (trend rasters, regime classifications, population layers, result tables) are on Zenodo: DOI to be added.
The repository has the processing and statistics only. Figures and tables for the paper are made from the derived data and are not included.

Data used: PRISM daily precipitation (Google Earth Engine, `OREGONSTATE/PRISM/ANd`), nClimGrid-Daily and GHCN-Daily (NOAA open data on AWS),
2020 Census block populations (PL 94-171), GlobPOP 2022 (sensitivity layer), IBTrACS storm tracks, and USGS HUC2 regions.

## Setup

```
conda env create -f environment.yml
conda activate precip_trends
earthengine authenticate
export EE_PROJECT=<your Earth Engine project>      # on Windows: setx EE_PROJECT <project>
```

Three folders are used. Each can be moved with an environment variable.

| folder | default | content |
|---|---|---|
| `input/` (`PRECIP_INPUT`) | `./input` | files you download once, listed in `input/README.md` |
| `data_cache/` (`PRECIP_DATA`) | `./data_cache` | large intermediate files written by the download scripts, about 9 GB |
| `outputs/` | `./outputs` | csv tables |

Run the scripts from inside `code/`.

## Order of the scripts

| script | what it does |
|---|---|
| `01_download_monthly.py` | monthly PRISM totals for each year (Earth Engine) |
| `01b_prism_metadata.py` | asset version, dataset type and creation date by year |
| `02_export_q95_asset.py` | exact 95th percentile of daily precipitation as an Earth Engine asset (about an hour) |
| `03_population_grid.py` | state raster; GlobPOP summed onto the analysis grid (sensitivity layer); comparison with Census estimates |
| `03b_census_population.py` | 2020 Census block populations summed onto the analysis grid (main population layer) |
| `04_trends_annual_seasonal.py` | annual and seasonal trends; seasonal contribution to the annual change |
| `05_download_extremes.py basic`, `q95`, `doy` | Rx1day, Rx5day, R10mm, R20mm, wet days, R95p, day of the Rx1day maximum |
| `06_download_nclimgrid.py` | annual indices from nClimGrid-Daily on the same cells |
| `07_prism_vs_nclimgrid.py` | PRISM against nClimGrid-Daily: trend agreement, Pettitt tests, before and after 2002 |
| `08_extremes_trends.py` | trends of all extreme indices (run once with the basic bands and again after the q95 bands) |
| `09_classification_variants.py` | regime classification: absolute, relative, signal-to-noise, three by three, significance-gated |
| `10_period_sensitivity.py` | other start and end years |
| `11a_download_ancillary.py`, `11b_regional_summaries.py` | HUC2 regions, temperature series, regional tables |
| `12_tropical_cyclone_attribution.py` | storm match for the annual Rx1day maximum |
| `15_bivariate_stats.py` | Rx1day trend class by population tertile |
| `16_autocorrelation.py` | lag-1 autocorrelation before and after detrending |
| `19_station_benchmark.py` | PRISM and nClimGrid-Daily against COOP stations |
| `20_primary_class_tables.py` | class tables for the relative and absolute classifications; agreement between products |
| `21_tc_autumn.py` | September to November precipitation on and off tropical-cyclone days |
| `23_population_sensitivity.py` | exposure with Census 2020 against GlobPOP 2022 |
| `25_region_null.py` | population reference within NOAA climate regions and HUC2 regions |
| `22_export_archive.py` | collects the derived data into the folder that is archived on Zenodo |

`config.py` holds the grid and paths, `trend.py` the trend statistics (vectorized Mann-Kendall with ties, Sen slope, Hamed-Rao correction,
Benjamini-Hochberg, Pettitt), and `ee_io.py` the Earth Engine download helper. `python test_trend.py` compares `trend.py` with scipy.

## Notes

- All gridded results are on one grid: EPSG:4326, 4000 m expressed in degrees (0.03593 degrees), 1614 by 697 cells, 645,556 CONUS cells.
- `02_export_q95_asset.py` writes an asset to your Earth Engine project, and `05_download_extremes.py q95` reads it.
- nClimGrid-Daily and GHCN-Daily files are streamed from the NOAA buckets and are not stored.
- PRISM is re-made about six months after each month, so a download made later than October 2026 can differ slightly in 2025.
- Trend statistics are computed in Python. Earth Engine is used for daily aggregation only.

## License

Code: MIT (see `LICENSE`). Data on Zenodo: CC BY 4.0.
