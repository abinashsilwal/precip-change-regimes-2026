"""
Lag-1 autocorrelation of annual precipitation and Rx1day, before and after removing the Sen trend.
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
W = C.lat_weights()[valid]
n = len(C.YEARS)
t = np.arange(n, dtype="float32")[:, None]
rows = []
for k, f in (("annual", "annual_sen_1981_2025.tif"), ("rx1", "rx1_sen_1981_2025.tif")):
    X = np.load(C.DATA / f"cube_{k}.npy")
    sen = rasterio.open(C.RASTERS / f).read(1)[valid]
    for lab, Y in (("raw", X), ("detrended", X - sen[None, :] * t)):
        Y = Y - Y.mean(0)
        r1 = (Y[:-1] * Y[1:]).sum(0) / (Y ** 2).sum(0)
        rows.append(dict(series=k, version=lab, median_r1=float(np.median(r1)), pct_area_abs_r1_gt_0p3=100 * W[np.abs(r1) > 0.3].sum() / W.sum(),
                         pct_area_r1_gt_0p3=100 * W[r1 > 0.3].sum() / W.sum()))
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "16_lag1_detrended.csv", index=False)
print(df.round(3).to_string(index=False))
