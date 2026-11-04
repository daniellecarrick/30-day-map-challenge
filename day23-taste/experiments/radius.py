import sys, numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
g = pd.read_csv(sys.argv[1])
g = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.longitude, g.latitude), crs=4326).to_crs(32618)
xy = np.c_[g.geometry.x, g.geometry.y]; t = cKDTree(xy)
chains = g.dba.str.upper().str.contains("STARBUCKS|DUNKIN|PRET A|BLUESTONE|JOE & THE JUICE|GREGORYS|BLUE BOTTLE|LA COLOMBE|JOE COFFEE|TIM HORTONS")
print("coffee chains share:", chains[g.cuisine_description=="Coffee/Tea"].mean().round(2))
R = 500
for c, mask in [("Chinese", g.cuisine_description=="Chinese"), ("Coffee/Tea", g.cuisine_description=="Coffee/Tea"), ("indie coffee", (g.cuisine_description=="Coffee/Tea") & ~chains)]:
    nb = t.query_ball_point(xy, R)
    tot = np.array([len(n) for n in nb]); hit = np.array([mask.values[n].sum() for n in nb])
    df = pd.DataFrame({"nta": g.ntaname, "dba": g.dba, "n": hit, "tot": tot, "share": hit/np.maximum(tot,1)})
    print(f"\n== {c} within {R} m of a restaurant")
    print(" densest (count):", df.sort_values("n", ascending=False).drop_duplicates("nta").head(5)[["nta","n","tot"]].to_string(index=False, header=False))
    print(" highest share (>=60 places):", df[df.tot>=60].sort_values("share", ascending=False).drop_duplicates("nta").head(5)[["nta","n","tot","share"]].round(2).to_string(index=False, header=False))
# Cobble Hill proper (Court St & Atlantic-ish center)
ch = gpd.GeoSeries(gpd.points_from_xy([-73.9955],[40.6870]), crs=4326).to_crs(32618)[0]
n = t.query_ball_point([ch.x, ch.y], R); sub = g.iloc[n]
cof = sub.cuisine_description=="Coffee/Tea"
print(f"\nCobble Hill center, {R} m: {len(sub)} places, coffee {cof.sum()} ({cof.mean():.0%}), indie {(cof & ~chains.iloc[n]).sum()}; citywide coffee share {(g.cuisine_description=='Coffee/Tea').mean():.1%}")
