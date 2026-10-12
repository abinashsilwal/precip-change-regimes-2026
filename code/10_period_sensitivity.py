"""
Repeats the trend and classification analysis for other start and end years.
"""
import numpy as np, pandas as pd
import config as C, trend as T

valid = C.valid_mask()
W = C.lat_weights()[valid]
pop = np.load(C.DATA / "pop_on_prism_grid.npy")[valid].astype("float64")
state = np.load(C.DATA / "state_on_prism_grid.npy")[valid]
A = np.load(C.DATA / "cube_annual.npy")
R = np.load(C.DATA / "cube_rx1.npy")
yrs = np.array(C.YEARS)
sid = np.unique(state[state > 0])


def null_share(m):
    out = sum(pop[state == s].sum() * W[(state == s) & m].sum() / W[state == s].sum() for s in sid)
    return 100 * out / pop.sum()


rows = []
for y0, y1 in [(1981, 2025), (1982, 2025), (1981, 2020), (1986, 2025), (1991, 2025), (1981, 2010), (1996, 2025), (2002, 2025)]:
    sel = (yrs >= y0) & (yrs <= y1)
    a, r = A[sel], R[sel]
    ra, rr = T.mk_sen(a), T.mk_sen(r)
    fa, _ = T.bh_fdr(ra["p"], 0.10)
    fr, _ = T.bh_fdr(rr["p"], 0.10)
    rel_a, rel_r = 1000 * ra["sen"] / a.mean(0), 1000 * rr["sen"] / r.mean(0)
    row = dict(window=f"{y0}-{y1}", n_years=int(sel.sum()), pct_area_fdr_annual=100 * W[fa].sum() / W.sum(),
               pct_area_fdr_rx1=100 * W[fr].sum() / W.sum(), conus_mean_annual_pct_per_decade=float((W * rel_a).sum() / W.sum()),
               conus_mean_rx1_pct_per_decade=float((W * rel_r).sum() / W.sum()))
    for tag, (x, z) in (("rel", (rel_a, rel_r)), ("abs", (ra["sen"], rr["sen"]))):
        ta, tr = np.percentile(np.abs(x), 50), np.percentile(np.abs(z), 50)
        head = (x > ta) & (z > tr)
        row.update({f"{tag}_area_pct": 100 * W[head].sum() / W.sum(), f"{tag}_pop_pct": 100 * pop[head].sum() / pop.sum(),
                    f"{tag}_pop_pct_null": null_share(head)})
    rows.append(row)
    print(row, flush=True)
df = pd.DataFrame(rows)
df.to_csv(C.OUT / "10_period_sensitivity.csv", index=False)
print(df.round(2).to_string(index=False))
