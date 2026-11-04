import sys, pandas as pd, geopandas as gpd
d = sys.argv[1]
r = pd.read_csv(f"{d}/restaurants.csv", dtype=str)
r["last_insp"] = pd.to_datetime(r.last_insp)
r[["latitude","longitude"]] = r[["latitude","longitude"]].astype(float)
print("rows", len(r), "unique camis", r.camis.nunique())
r = r.sort_values("last_insp").drop_duplicates("camis", keep="last")
print("never inspected (1900):", (r.last_insp.dt.year < 2000).sum())
print("last insp before 2023:", (r.last_insp < "2023-01-01").sum())
r = r[(r.latitude > 40) & (r.last_insp >= "2023-01-01")]
print("kept", len(r))
print(r.cuisine_description.value_counts().head(30).to_string())
nta = gpd.read_file(f"{d}/nta2020.geojson")[["nta2020","ntaname","boroname","geometry"]]
g = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.longitude, r.latitude), crs=4326)
g = gpd.sjoin(g, nta, predicate="within", how="inner")
g.drop(columns="geometry").to_csv(f"{d}/_joined.csv", index=False)
tot = g.groupby("ntaname").size()
city = g.cuisine_description.value_counts(normalize=True)
for c in ["Chinese","Coffee/Tea","Italian","Pizza","Mexican","Japanese","Korean","Caribbean","Bakery Products/Desserts","Indian","Thai","Jewish/Kosher","Latin American","Spanish"]:
    s = g[g.cuisine_description==c].groupby("ntaname").size()
    df = pd.DataFrame({"n":s, "total":tot}).dropna(); df["share"]=df.n/df.total; df["LQ"]=df.share/city[c]
    print(f"\n== {c}  (citywide {city[c]:.1%}, n={int(s.sum())})")
    print(" by count:", ", ".join(f"{i} {int(v)}" for i,v in s.sort_values(ascending=False).head(5).items()))
    top = df[df.total>=40].sort_values("share", ascending=False).head(5)
    print(" by share (>=40 rest.):", ", ".join(f"{i} {r.share:.0%} ({int(r.n)}/{int(r.total)})" for i,r in top.iterrows()))
for name in ["Chinatown-Two Bridges","Carroll Gardens-Cobble Hill-Gowanus-Red Hook"]:
    sub = g[g.ntaname==name]
    print(f"\n{name}: {len(sub)} places; top:", sub.cuisine_description.value_counts().head(6).to_dict())
