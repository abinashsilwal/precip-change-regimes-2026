"""
Regime classification from the Sen slopes of annual precipitation and Rx1day. Variants: absolute, relative (percent of
mean per decade) and signal-to-noise trends; thresholds at the 33rd, 50th and 67th percentile; with and without the requirement
that trends are FDR-significant; five-class and three-by-three schemes. Reports area, population and the within-state area null.
"""
import numpy as np, pandas as pd, rasterio
import config as C

valid = C.valid_mask()
P = int(valid.sum())
W = C.lat_weights()[valid]
n = len(C.YEARS)
pop = np.load(C.DATA / "pop_on_prism_grid.npy")[valid].astype("float64")
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
rd = lambda name: rasterio.open(C.RASTERS / name).read(1)[valid]

A = np.load(C.DATA / "cube_annual.npy")
R = np.load(C.DATA / "cube_rx1.npy")
senA, senR = rd("annual_sen_1981_2025.tif"), rd("rx1_sen_1981_2025.tif")
fdrA, fdrR = rd("annual_fdr_1981_2025.tif") > 0, rd("rx1_fdr_1981_2025.tif") > 0

t = np.arange(n, dtype="float32")[:, None]
def detrended_sd(X, sen):
    return (X - sen[None, :] * t - np.median(X - sen[None, :] * t, axis=0)).std(0, ddof=1)

metrics = {"abs": (senA, senR),
           "rel": (1000 * senA / A.mean(0), 1000 * senR / R.mean(0)),
           "snr": (senA * (n - 1) / detrended_sd(A, senA), senR * (n - 1) / detrended_sd(R, senR))}

NAMES5 = {1: "Drying + weaker extremes", 2: "Drying + stronger extremes", 3: "Wetting + weaker extremes",
          4: "Wetting + stronger extremes", 5: "No major change / weak mixed"}
def classify5(a, r, ta, tr, ga=None, gr=None):
    wet, dry, strong, weak = a > ta, a < -ta, r > tr, r < -tr
    if ga is not None:
        wet, dry, strong, weak = wet & ga, dry & ga, strong & gr, weak & gr
    out = np.full(P, 5, "int8")
    out[dry & weak] = 1; out[dry & strong] = 2; out[wet & weak] = 3; out[wet & strong] = 4
    return out

def classify9(a, r, ta, tr, ga=None, gr=None):
    """rows: A drying/neutral/wetting = 0/1/2 ; cols: R weaker/neutral/stronger = 0/1/2 ; code = 3*row + col + 1"""
    ra = np.where(a > ta, 2, np.where(a < -ta, 0, 1)); rr = np.where(r > tr, 2, np.where(r < -tr, 0, 1))
    if ga is not None:
        ra = np.where(ga, ra, 1); rr = np.where(gr, rr, 1)
    return (3 * ra + rr + 1).astype("int8")

# state-level area and population totals for the null
sid = np.unique(state[state > 0])
def shares(cl, codes):
    area = {c: 100 * W[cl == c].sum() / W.sum() for c in codes}
    popp = {c: 100 * pop[cl == c].sum() / pop.sum() for c in codes}
    null = {c: 0.0 for c in codes}
    for s in sid:
        m = state == s
        sp, sa = pop[m].sum(), W[m].sum()
        for c in codes:
            null[c] += 100 * sp * W[m & (cl == c)].sum() / sa / pop.sum()
    return area, popp, null

rows = []
for mname, (a, r) in metrics.items():
    for q in (33, 50, 67):
        ta, tr = np.percentile(np.abs(a), q), np.percentile(np.abs(r), q)
        for gate in ("none", "both_fdr", "either_fdr"):
            if gate == "none":
                g = (None, None)
            elif gate == "both_fdr":
                g = (fdrA, fdrR)
            else:
                g = (fdrA | fdrR, fdrA | fdrR)
            cl = classify5(a, r, ta, tr, *g)
            area, popp, null = shares(cl, range(1, 6))
            for c in range(1, 6):
                rows.append(dict(metric=mname, pct=q, gate=gate, thr_annual=ta, thr_rx1=tr, scheme="five", cls=NAMES5[c],
                                 area_pct=area[c], pop_pct=popp[c], pop_pct_null=null[c]))
            if q == 50 and gate == "none":
                C.write_tif(C.RASTERS / f"class5_{mname}_p50.tif", C.to_grid(cl.astype("float32"), valid))
        # 3x3 (no gating)
        cl9 = classify9(a, r, ta, tr)
        area, popp, null = shares(cl9, range(1, 10))
        lab = ["drying/weaker", "drying/neutral", "drying/stronger", "neutral/weaker", "neutral/neutral", "neutral/stronger",
               "wetting/weaker", "wetting/neutral", "wetting/stronger"]
        for c in range(1, 10):
            rows.append(dict(metric=mname, pct=q, gate="none", thr_annual=ta, thr_rx1=tr, scheme="nine", cls=lab[c - 1],
                             area_pct=area[c], pop_pct=popp[c], pop_pct_null=null[c]))
        if q == 50:
            C.write_tif(C.RASTERS / f"class9_{mname}_p50.tif", C.to_grid(cl9.astype("float32"), valid))
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "09_class_variants.csv", index=False)

# headline class across variants
head = df[(df.scheme == "five") & (df.cls == NAMES5[4])].copy()
head["pop_over_area"] = head.pop_pct / head.area_pct
head["excess_pts"] = head.pop_pct - head.area_pct
head["excess_explained_by_state_placement_pts"] = head.pop_pct_null - head.area_pct
head.to_csv(C.OUT / "09_summary_headline_class.csv", index=False)
print(head.round(2)[["metric", "pct", "gate", "thr_annual", "thr_rx1", "area_pct", "pop_pct", "pop_pct_null", "pop_over_area"]]
      .to_string(index=False))

# 3x3 at the 50th percentile for abs and rel
for mname in ("abs", "rel", "snr"):
    d = df[(df.scheme == "nine") & (df.metric == mname) & (df.pct == 50)][["cls", "area_pct", "pop_pct", "pop_pct_null"]]
    d.to_csv(C.OUT / f"09_nine_class_{mname}_p50.csv", index=False)
    print("\n3x3,", mname, "50th pct (rows: annual, cols: Rx1day)\n", d.round(1).to_string(index=False))

# 2-D histogram of the two trends in % per decade
a, r = metrics["rel"]
H, xe, ye = np.histogram2d(a, r, bins=[np.linspace(-8, 8, 65), np.linspace(-25, 25, 65)], weights=W)
np.savez_compressed(C.OUT / "09_hist2d_rel_per_decade.npz", H=H, xe=xe, ye=ye,
                    thr_a=np.percentile(np.abs(a), 50), thr_r=np.percentile(np.abs(r), 50))
print("\nCONUS area-mean change (%% per decade): annual %.2f, Rx1day %.2f" % ((W * a).sum() / W.sum(), (W * r).sum() / W.sum()))
C.log("done")
