"""
Reads the metadata of every daily image in the PRISM ANd asset (dataset type, version, creation date) and
summarizes it by year and, for 2025, by month. Used to document the AN81 to AN91 change in 2021 and the status of recent grids.
"""
import ee, pandas as pd
import config as C, ee_io

ee_io.init()
c = ee.ImageCollection(C.PRISM_ASSET)

def first(v):
    """properties are plain strings on some days and lists on others"""
    is_list = ee.Algorithms.ObjectType(v).compareTo("List").eq(0)
    return ee.Algorithms.If(is_list, ee.List(v).get(0), v)

def feat(im):
    return ee.Feature(None, {"idx": im.get("system:index"), "type": first(im.get("dataset_type")),
                             "ver": first(im.get("PRISM_DATASET_VERSION")), "code": first(im.get("PRISM_CODE_VERSION")),
                             "created": first(im.get("PRISM_DATASET_CREATE_DATE")), "fmt": first(im.get("PRISM_FORMAT_VERSION"))})
rows = []
for y in range(1981, 2027):
    s = ee.Date.fromYMD(y, 1, 1)
    fc = c.filterDate(s, s.advance(1, "year")).map(feat)
    rows += [f["properties"] for f in fc.getInfo()["features"]]
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["idx"], format="%Y%m%d")
df["year"], df["month"] = df.date.dt.year, df.date.dt.month
by_year = df.groupby("year").agg(days=("idx", "size"), types=("type", lambda s: ",".join(sorted(set(s)))),
                                 versions=("ver", lambda s: ",".join(sorted(set(s)))), code=("code", lambda s: ",".join(sorted(set(s)))),
                                 created_min=("created", "min"), created_max=("created", "max"))
by_year.to_csv(C.OUT / "01_prism_asset_metadata_by_year.csv")
pd.set_option("display.width", 250)
print(by_year.to_string())
m25 = df[df.year == 2025].groupby("month").agg(types=("type", lambda s: ",".join(sorted(set(s)))),
                                                created_min=("created", "min"), created_max=("created", "max"))
m25.to_csv(C.OUT / "01_prism_2025_by_month.csv")
print(m25.to_string())
