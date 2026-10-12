"""
Collects the derived data into one folder for archiving (Zenodo): multi-band GeoTIFFs for trends and yearly indices, regime
classifications, population and region grids, csv tables, and a checksum list.
Usage: python 22_export_archive.py [target_folder]       (default: ../archive)
"""
import hashlib, shutil, sys
import numpy as np, pandas as pd, rasterio
import config as C

target = C.Path(sys.argv[1]) if len(sys.argv) > 1 else C.OUT.parent / "archive"
for sub in ("grid", "trends", "indices", "classification", "population", "regions", "tables"):
    (target / sub).mkdir(parents=True, exist_ok=True)
valid = C.valid_mask()
H, W = C.SHAPE
base = dict(driver="GTiff", height=H, width=W, crs=C.CRS, transform=C.TRANSFORM, compress="deflate", tiled=True, blockxsize=256, blockysize=256)


def write_stack(path, arrays, names, dtype="float32", nodata=np.nan, predictor=3):
    prof = dict(base, count=len(arrays), dtype=dtype, nodata=nodata, predictor=predictor)
    with rasterio.open(path, "w", **prof) as d:
        for i, (a, n) in enumerate(zip(arrays, names), 1):
            d.write(a.astype(dtype), i)
            d.set_band_description(i, str(n))


def vec_to_grid(v):
    out = np.full((H, W), np.nan, "float32")
    out[valid] = v
    return out


# grid mask
write_stack(target / "grid" / "conus_grid_mask.tif", [valid.astype("uint8")], ["CONUS cell"], dtype="uint8", nodata=255, predictor=2)

# trends: sen, p, p_hr, fdr per index
rdr = lambda n: rasterio.open(C.RASTERS / n).read(1)
names = ["annual", "mam", "jja", "son", "djf_complete", "djf_cal", "rx1", "rx5", "r10", "r20", "wet", "r95p_wet", "n95_wet", "r95p_all", "n95_all"]
for n in names:
    write_stack(target / "trends" / f"trend_{n}_1981_2025.tif", [rdr(f"{n}_{k}_1981_2025.tif") for k in ("sen", "p", "p_hr", "fdr")],
                ["sen_slope_per_year", "mann_kendall_p", "mann_kendall_p_hamed_rao", "fdr_significant_0p10"])
for n in ("annual", "rx1", "rx5"):
    write_stack(target / "trends" / f"trend_nclimgrid_{n}_sen_1981_2025.tif", [rdr(f"cmp_nclim_{n}_sen.tif")], ["sen_slope_per_year"])

# yearly indices (one band per year)
years = C.YEARS
cube = lambda n: np.load(C.DATA / f"cube_{n}.npy")
stacks = {"annual_precip_mm": cube("annual"), "rx1day_mm": cube("rx1"), "rx5day_mm": cube("rx5"), "r10mm_days": cube("r10"), "r20mm_days": cube("r20"),
          "wet_days": cube("wet"), "r95p_wet_mm": cube("r95p_wet"), "n95_wet_days": cube("n95_wet")}
M = np.stack([np.load(C.DATA / f"monthly_{y}.npy")[:, valid] for y in years])
stacks["djf_calendar_precip_mm"] = M[:, 0] + M[:, 1] + M[:, 11]
stacks["mam_precip_mm"] = M[:, 2:5].sum(1)
stacks["jja_precip_mm"] = M[:, 5:8].sum(1)
stacks["son_precip_mm"] = M[:, 8:11].sum(1)
for k, v in stacks.items():
    write_stack(target / "indices" / f"prism_{k}_1981_2025.tif", [vec_to_grid(v[i]) for i in range(len(years))], years)
for k, key in (("annual_precip_mm", "annual"), ("rx1day_mm", "rx1"), ("rx5day_mm", "rx5")):
    arr = [np.load(C.DATA / f"nclim_{y}.npz")[key] for y in years]
    write_stack(target / "indices" / f"nclimgrid_{k}_1981_2025.tif", arr, years)

# classifications
for n in ("class5_rel_p50", "class5_abs_p50", "class5_snr_p50", "class9_rel_p50", "class9_abs_p50", "class9_snr_p50", "agree_wet_strong_rel"):
    a = rdr(f"{n}.tif")
    write_stack(target / "classification" / f"{n}.tif", [np.where(np.isfinite(a), a, 255)], [n], dtype="uint8", nodata=255, predictor=2)

# population, states, regions, storm fraction
for fname, out_name in (("pop_on_prism_grid.npy", "census2020_on_analysis_grid.tif"), ("pop_globpop2022_grid.npy", "globpop2022_on_analysis_grid.tif")):
    pop = np.load(C.DATA / fname).astype("float32")
    pop[~valid] = np.nan
    write_stack(target / "population" / out_name, [pop], ["persons per cell"])
st = np.load(C.DATA / "state_on_prism_grid.npy").astype("uint8")
write_stack(target / "population" / "state_id_grid.tif", [st], ["state id"], dtype="uint8", nodata=0, predictor=2)
shutil.copy(C.OUT / "state_ids.csv", target / "population" / "state_ids.csv")
write_stack(target / "regions" / "huc2_grid.tif", [np.load(C.DATA / "huc2.npy").astype("uint8")], ["HUC2 id"], dtype="uint8", nodata=0, predictor=2)
for R in (300, 500):
    write_stack(target / "regions" / f"rx1day_tropical_cyclone_fraction_{R}km.tif", [rdr(f"tc_fraction_{R}km.tif")], ["share of years"])
for n in ("total", "tc", "nontc"):
    f = C.RASTERS / f"tc_son_{n}_sen.tif"
    if f.exists():
        write_stack(target / "trends" / f"trend_nclimgrid_son_{n}_sen_1981_2025.tif", [rdr(f"tc_son_{n}_sen.tif")], ["sen_slope_mm_per_year"])

# national series
w = C.lat_weights()
wv = w[valid]
nat = {"year": years, "prism_annual_precip_mm": (stacks["annual_precip_mm"] * wv).sum(1) / wv.sum(),
       "prism_rx1day_mm": (stacks["rx1day_mm"] * wv).sum(1) / wv.sum()}
for k, key in (("nclimgrid_annual_precip_mm", "annual"), ("nclimgrid_rx1day_mm", "rx1")):
    s = []
    for y in years:
        a = np.load(C.DATA / f"nclim_{y}.npz")[key]
        ok = np.isfinite(a)
        s.append((a[ok] * w[ok]).sum() / w[ok].sum())
    nat[k] = s
t = pd.read_csv(C.OUT / "11_conus_tmean_series.csv")
nat["prism_tmean_c"] = t.tmean_c.values
pd.DataFrame(nat).round(3).to_csv(target / "tables" / "conus_mean_series_1981_2025.csv", index=False)

# tables
skip = ("log_",)
for f in sorted(C.OUT.glob("*.csv")):
    shutil.copy(f, target / "tables" / f.name)

# checksums
lines = []
for f in sorted(target.rglob("*")):
    if f.is_file() and f.name != "checksums.sha256":
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {f.relative_to(target).as_posix()}")
(target / "checksums.sha256").write_text("\n".join(lines) + "\n")
C.log("archive written to", target, len(lines), "files")
