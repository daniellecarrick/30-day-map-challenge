import numpy as np, rasterio, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
R = Path("data/rasters"); years = np.arange(1981, 2026)
A = []
for y in years:
    with rasterio.open(R / f"lilac_bloom_{y}.tif") as r:
        a = r.read(1).astype(float); a[a == r.nodata] = np.nan; A.append(a); tr = r.transform
A = np.stack(A)
nvalid = (~np.isnan(A)).sum(0); print("valid-years counts", np.unique(nvalid, return_counts=True))
full = nvalid == 45
normal = np.nanmean(A[(years >= 1991) & (years <= 2020)], 0)
chg = np.nanmean(A[years >= 2011], 0) - np.nanmean(A[years <= 1995], 0)
rows, cols = np.indices(normal.shape); lat = tr.f + (rows + .5) * tr.e; lon = tr.c + (cols + .5) * tr.a
for cut in (60, 75, 90):
    m = full & (normal >= cut)
    print(f"mask<{cut}: keep {m.sum()/full.sum():.2%}  normal p5/p50/p95", np.percentile(normal[m], [5, 50, 95]).round(),
          " chg median", np.median(chg[m]).round(1), "share earlier", (chg[m] < 0).mean().round(2), "share >=5 earlier", (chg[m] <= -5).mean().round(2))
m = full & (normal >= 75)
e = m & (lon > -100)
k = np.polyfit(lat[e], normal[e], 1); print("east: days per degree lat", k[0].round(2), "-> miles/day", (69 / k[0]).round(1))
for name, la, lo in [("Atlanta", 33.75, -84.39), ("DC", 38.9, -77.04), ("NYC", 40.71, -74.0), ("Chicago", 41.88, -87.63), ("Boston", 42.36, -71.06), ("Minneapolis", 44.98, -93.27), ("Denver", 39.74, -104.99), ("Bangor", 44.8, -68.77), ("Fargo", 46.88, -96.79), ("Seattle", 47.6, -122.33), ("Leadville", 39.25, -106.29)]:
    r_, c_ = int((la - tr.f) / tr.e), int((lo - tr.c) / tr.a); print(f"{name:12s} normal {normal[r_, c_]:.0f}  chg {chg[r_, c_]:+.1f}")
print("latest normal (masked) top", np.sort(normal[m])[-5:])
