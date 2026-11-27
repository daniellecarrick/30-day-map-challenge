"""Day 28 · Senses: Feeling — two candidate directions, drafted as PNGs to compare.

  out/explore_lonely.png   NYC census tracts: share of adults who often feel lonely (CDC PLACES 2025 release)
  out/explore_names.png    Lower-48 place names containing a feeling word (USGS GNIS)

Run: sh fetch.sh, then ../.venv/bin/python explore.py
"""
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
DAY = "Day 28: Senses: Feeling"

# ---------------------------------------------------------------- loneliness
NYC = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}


def lonely():
    d = pd.DataFrame(json.loads((RAW / "places_nyc.json").read_text()))
    d["v"] = d.data_value.astype(float)
    w = d.pivot(index="locationname", columns="measureid", values="v")
    pop = d.drop_duplicates("locationname").set_index("locationname").totalpopulation.astype(float)
    w["pop"] = pop
    t = gpd.read_file(RAW / "tracts_ny.zip")
    t = t[t.COUNTYFP.isin(NYC)].to_crs("EPSG:2263")
    nta = pd.DataFrame(json.loads((RAW / "nyc_tracts_nta.json").read_text())).set_index("geoid")
    t = t.join(w, on="GEOID").join(nta, on="GEOID")
    have = t[t.LONELINESS.notna()].copy()
    avg = (have.LONELINESS * have["pop"]).sum() / have["pop"].sum()
    color = S.make_scale(S.diverging_stops(round(avg), 10))
    have["color"] = have.LONELINESS.apply(color)
    return t, have, avg


def png_lonely():
    t, have, avg = lonely()
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.05)
    t[t.LONELINESS.isna()].plot(ax=ax, color=S.C["other"], edgecolor=S.EDGE, linewidth=0.2)
    have.plot(ax=ax, color=have.color, edgecolor=S.EDGE, linewidth=0.2)
    t.dissolve("COUNTYFP").boundary.plot(ax=ax, color=S.INK, linewidth=0.5)
    ax.set_axis_off()
    ax.set_aspect("equal")
    hi = have.loc[have.LONELINESS.idxmax()]
    lo = have.loc[have.LONELINESS.idxmin()]
    S.callout(ax, (hi.geometry.centroid.x, hi.geometry.centroid.y), (0.04, 0.80),
              f"{hi.ntaname}, {hi.boroname}", f"{hi.LONELINESS:.0f}% often feel lonely", outline=hi.geometry)
    S.callout(ax, (lo.geometry.centroid.x, lo.geometry.centroid.y), (0.62, 0.12),
              f"{lo.ntaname}, {lo.boroname}", f"{lo.LONELINESS:.0f}% often feel lonely", outline=lo.geometry)
    share_hi = (have["pop"][have.LONELINESS >= 45].sum() / have["pop"].sum()) * 100
    S.title(fig, f"About {avg:.0f}% of adult New Yorkers often feel lonely", y=0.965)
    fig.text(0.5, 0.915, f"Share of adults who say they always, usually or sometimes feel lonely, by census tract. "
             f"Blue: below the city's {avg:.0f}%. Brown: above.", ha="center", fontsize=S.SIZE["subtitle"], color=S.MUTED)
    fig.text(0.5, 0.895, f"Range {have.LONELINESS.min():.0f}% to {have.LONELINESS.max():.0f}% · "
             f"{share_hi:.0f}% of New Yorkers live in tracts at 45% or more", ha="center",
             fontsize=S.SIZE["subtitle"], color=S.MUTED)
    S.credit(fig, f"Modeled tract estimates · Data: CDC PLACES 2025 release (BRFSS 2023)  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "explore_lonely.png", dpi=200)
    print(f"lonely: avg {avg:.1f}%, max {hi.LONELINESS} {hi.ntaname}, min {lo.LONELINESS} {lo.ntaname}")
    print(have.groupby("boroname").apply(lambda g: (g.LONELINESS * g['pop']).sum() / g['pop'].sum()).round(1))


# ---------------------------------------------------------------- place names
GOOD = ["Pleasant", "Happy", "Hope", "Hopeful", "Joy", "Joyful", "Lovely", "Peaceful", "Bliss", "Comfort",
        "Delight", "Merry", "Jolly", "Glad", "Pleasure", "Calm", "Tranquil", "Tranquility", "Serene",
        "Cheerful", "Contentment", "Content", "Lucky", "Wonderful", "Sweetheart", "Proud"]
BAD = ["Lonesome", "Lonely", "Misery", "Miserable", "Disappointment", "Grief", "Despair", "Dismal", "Sorrow",
       "Sad", "Melancholy", "Gloomy", "Forlorn", "Desolation", "Hopeless", "Weary", "Hungry", "Mad", "Angry",
       "Hate", "Fear", "Terror", "Horror", "Dread", "Fright", "Scared", "Scary", "Panic", "Worry",
       "Trouble", "Troublesome", "Confusion", "Restless", "Grim", "Woe"]
WORD = {w.lower(): ("good" if w in GOOD else "bad") for w in GOOD + BAD}
PAT = re.compile(r"\b(" + "|".join(GOOD + BAD) + r")\b", re.I)


def names():
    g = pd.read_csv(RAW / "Text" / "DomesticNames_National.txt", sep="|", encoding="utf-8-sig",
                    usecols=["feature_name", "feature_class", "state_name", "prim_lat_dec", "prim_long_dec"])
    g = g.dropna(subset=["feature_name"])
    g = g[~g.feature_class.isin(["Civil", "Census", "Military"])]          # admin duplicates of real places
    g = g[~g.feature_name.str.contains(r"\(historical\)", regex=True)]
    g["word"] = g.feature_name.str.extract(PAT, expand=False).str.lower()
    g = g[g.word.notna() & (g.prim_lat_dec != 0)].copy()
    g["mood"] = g.word.map(WORD)
    return g


def png_names():
    g = names()
    states = gpd.read_file(RAW / "states.zip")
    states = states[~states.STUSPS.isin(["AK", "HI", "PR"])].to_crs("EPSG:5070")
    pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.prim_long_dec, g.prim_lat_dec), crs=4326).to_crs(5070)
    pts = gpd.sjoin(pts, states[["geometry"]], predicate="within").drop(columns="index_right")
    good, bad = pts[pts.mood == "good"], pts[pts.mood == "bad"]
    blue, brown = S.DIVERGING[0], S.DIVERGING[3]
    fig, ax = plt.subplots(figsize=(12, 8.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04)
    states.plot(ax=ax, color=S.C["ctl"], edgecolor=S.BG, linewidth=0.6)
    good.plot(ax=ax, color=blue, markersize=2.2, alpha=0.7, linewidth=0)
    bad.plot(ax=ax, color=brown, markersize=3.5, alpha=0.9, linewidth=0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ratio = len(good) / len(bad)
    S.title(fig, f"America named {ratio:.0f} places for a good feeling for every one it named for a bad one", y=0.965)
    top_g = good.word.value_counts().head(5)
    top_b = bad.word.value_counts().head(5)
    fig.text(0.5, 0.915, f"Lakes, creeks, peaks and towns with a feeling in the name. "
             f"Blue: good ({len(good):,}), led by " + ", ".join(f"{w.title()} {n:,}" for w, n in top_g.items()) + ".",
             ha="center", fontsize=S.SIZE["subtitle"], color=S.MUTED)
    fig.text(0.5, 0.893, f"Brown: bad ({len(bad):,}), led by " + ", ".join(f"{w.title()} {n:,}" for w, n in top_b.items()) + ".",
             ha="center", fontsize=S.SIZE["subtitle"], color=S.MUTED)
    S.credit(fig, f"Whole-word match on {len(WORD)} feeling words, lower 48 · Data: USGS Geographic Names Information System  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "explore_names.png", dpi=200)
    print(f"names: {len(good)} good, {len(bad)} bad (lower 48); all US {len(g)}")
    print("good:", good.word.value_counts().head(12).to_dict())
    print("bad:", bad.word.value_counts().head(15).to_dict())
    by = pts.groupby(["state_name", "mood"]).size().unstack(fill_value=0)
    by["bad_share"] = (by.bad / (by.good + by.bad) * 100).round(0)
    print(by.sort_values("bad_share").to_string())


if __name__ == "__main__":
    S.apply()
    png_lonely()
    png_names()
