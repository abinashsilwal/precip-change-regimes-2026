"""Earth Engine helpers: initialise, and pull an ee.Image onto the reference grid as a numpy array."""
import io, time, zipfile
import ee, numpy as np, rasterio
import requests
import config as C

def init():
    ee.Initialize(project=C.EE_PROJECT)

def conus():
    st = ee.FeatureCollection("TIGER/2018/States")
    ex = ["AK", "HI", "PR", "AS", "GU", "MP", "VI"]
    return st.filter(ee.Filter.inList("STUSPS", ex).Not()).geometry().dissolve()

def prism():
    return ee.ImageCollection(C.PRISM_ASSET).select("ppt")

def _tiles(nr, nc):
    H, W = C.SHAPE
    rs = np.linspace(0, H, nr + 1).astype(int); cs = np.linspace(0, W, nc + 1).astype(int)
    for i in range(nr):
        for j in range(nc):
            yield rs[i], rs[i + 1], cs[j], cs[j + 1]

def fetch(image, nr=1, nc=1, retries=4, nodata=-9999.0):
    """Download `image` (clipped to CONUS) on the reference grid. Returns array (bands, H, W) float32, NaN outside.
    Tiles (nr x nc) keep each request below Earth Engine size and time limits."""
    t = C.TRANSFORM
    img = image.clip(conus()).unmask(nodata).toFloat()
    out = None
    for r0, r1, c0, c1 in _tiles(nr, nc):
        x0, x1 = t.c + c0 * t.a, t.c + c1 * t.a
        y1, y0 = t.f + r0 * t.e, t.f + r1 * t.e
        region = ee.Geometry.Rectangle([x0, y0, x1, y1], "EPSG:4326", False)
        params = dict(region=region, crs="EPSG:4326", crs_transform=[t.a, 0, t.c, 0, t.e, t.f], format="GEO_TIFF")
        for k in range(retries):
            try:
                url = img.getDownloadURL(params)
                resp = requests.get(url, timeout=600); resp.raise_for_status()
                break
            except Exception as e:
                if k == retries - 1: raise
                time.sleep(10 * (k + 1))
        data = resp.content
        arrs, tr = [], None
        if data[:2] == b"PK":                      # zipped multi-band download: one tif per band
            z = zipfile.ZipFile(io.BytesIO(data))
            for n in sorted(n for n in z.namelist() if n.endswith(".tif")):
                with rasterio.MemoryFile(z.read(n)) as m, m.open() as s: arrs.append(s.read(1)); tr = s.transform
            arr = np.stack(arrs)
        else:
            with rasterio.MemoryFile(data) as m, m.open() as s: arr = s.read(); tr = s.transform
        # EE trims a tile to the extent of valid data, so place it by its own origin on the reference grid
        c_off = int(round((tr.c - t.c) / t.a)); r_off = int(round((tr.f - t.f) / t.e))
        assert abs(tr.a - t.a) < 1e-9 and abs(tr.e - t.e) < 1e-9, "pixel size differs from reference grid"
        if out is None: out = np.full((arr.shape[0],) + C.SHAPE, np.nan, "float32")
        h, w = arr.shape[1:]
        out[:, r_off:r_off + h, c_off:c_off + w] = arr
    out[~np.isfinite(out) | (out == nodata)] = np.nan   # EE writes -inf outside the clip
    return out
