"""Day 28 · Senses: Feeling — US places named for a feeling, good (Pleasant, Hope) or bad
(Lonesome, Misery).

Writes:
  - out/feeling_places.png    static map (lower 48)
  - out/feeling_places.html   interactive map (all 50 states)

Data: USGS Geographic Names Information System, domestic names (Nov 2026). sh fetch.sh first.
Run:  ../.venv/bin/python build.py
"""
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
SOURCE_URL = "https://www.usgs.gov/tools/geographic-names-information-system-gnis"

# Each view sorts its words into two sides: "lo" draws in blue, "hi" in amber/brown
# (the series' diverging ends). Whole-word, case-insensitive match on the feature name.
VIEWS = {
    "emotion": {
        "lo": ("Good feeling", ["Pleasant", "Happy", "Hope", "Hopeful", "Joy", "Joyful", "Lovely", "Peaceful",
                                "Bliss", "Comfort", "Delight", "Merry", "Jolly", "Glad", "Pleasure", "Calm",
                                "Tranquil", "Tranquility", "Serene", "Cheerful", "Contentment", "Content",
                                "Wonderful", "Sweetheart", "Proud"]),
        "hi": ("Bad feeling", ["Lonesome", "Lonely", "Misery", "Miserable", "Disappointment", "Grief", "Despair",
                               "Dismal", "Sorrow", "Sad", "Melancholy", "Gloomy", "Forlorn", "Desolation",
                               "Hopeless", "Weary", "Mad", "Angry", "Hate", "Fear", "Terror", "Horror", "Dread",
                               "Fright", "Scared", "Scary", "Panic", "Worry", "Trouble", "Troublesome",
                               "Confusion", "Restless", "Grim", "Woe"]),
    },
}
WEST = {"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"}   # Census West region
LOWER48_DROP = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}


def word_table():
    rows = []
    for view, sides in VIEWS.items():
        for side, (_, words) in sides.items():
            rows += [(w.lower(), view, side) for w in words]
    return pd.DataFrame(rows, columns=["word", "view", "side"])


def load() -> pd.DataFrame:
    g = pd.read_csv(RAW / "Text" / "DomesticNames_National.txt", sep="|", encoding="utf-8-sig",
                    usecols=["feature_name", "feature_class", "state_name", "state_numeric", "county_name",
                             "prim_lat_dec", "prim_long_dec"], dtype={"state_numeric": str})
    g = g.dropna(subset=["feature_name"])
    g = g[~g.feature_class.isin(["Civil", "Census", "Military"])]        # admin twins of real places
    g = g[~g.feature_name.str.contains(r"\(historical\)", regex=True)]   # places that no longer exist
    g = g[(g.prim_lat_dec != 0) & g.prim_lat_dec.notna()]
    words = word_table()
    out = []
    for view in VIEWS:
        w = words[words.view == view]
        pat = re.compile(r"\b(" + "|".join(w.word) + r")\b", re.I)
        m = g[g.feature_name.str.contains(pat.pattern, case=False, regex=True)].copy()
        m["word"] = m.feature_name.str.extract(pat, expand=False).str.lower()
        out.append(m.merge(w, on="word"))
    d = pd.concat(out, ignore_index=True)
    st = gpd.read_file(RAW / "states.zip")[["STATEFP", "STUSPS", "NAME"]]
    d = d.merge(st, left_on="state_numeric", right_on="STATEFP", how="inner")   # 50 states + DC + PR
    return d[d.STUSPS != "PR"]


def stats(d: pd.DataFrame) -> dict:
    s = {}
    for view in VIEWS:
        v = d[d.view == view]
        s[view] = {
            "lo": int((v.side == "lo").sum()), "hi": int((v.side == "hi").sum()),
            # share of each side's names that are in the West
            "lo_in_west": round(100 * v[v.side == "lo"].STUSPS.isin(WEST).mean()),
            "hi_in_west": round(100 * v[v.side == "hi"].STUSPS.isin(WEST).mean()),
            "top_lo": v[v.side == "lo"].word.value_counts().head(3).to_dict(),
            "top_hi": v[v.side == "hi"].word.value_counts().head(3).to_dict(),
        }
        by = v.groupby("STUSPS").side.apply(lambda x: (x == "hi").mean() * 100)
        n = v.groupby("STUSPS").size()
        s[view]["states"] = {k: {"pct": round(by[k]), "n": int(n[k])} for k in by.index}
    return s


def png(d, states, s, view, title, lede, fname):
    v = d[d.view == view]
    v = v[~v.STUSPS.isin(LOWER48_DROP)]
    pts = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.prim_long_dec, v.prim_lat_dec), crs=4326).to_crs(5070)
    lo, hi = pts[pts.side == "lo"], pts[pts.side == "hi"]
    blue, brown = S.DIVERGING[0], S.DIVERGING[3]
    fig, ax = plt.subplots(figsize=(12, 8.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.04)
    states.plot(ax=ax, color=S.C["ctl"], edgecolor=S.BG, linewidth=0.6)
    # draw the rarer side on top so it isn't buried
    first, second = (lo, hi) if len(lo) >= len(hi) else (hi, lo)
    for part in (first, second):
        part.plot(ax=ax, color=blue if part is lo else brown, markersize=3.2, alpha=0.85, linewidth=0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    S.title(fig, title, y=0.965)
    fig.text(0.5, 0.912, lede, ha="center", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED,
             linespacing=1.5)
    lab_lo, lab_hi = VIEWS[view]["lo"][0], VIEWS[view]["hi"][0]
    ax.scatter([], [], color=blue, s=22, label=f"{lab_lo}  ({len(lo):,})")
    ax.scatter([], [], color=brown, s=22, label=f"{lab_hi}  ({len(hi):,})")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=S.SIZE["legend"] + 1, handletextpad=0.3,
              title="Each dot is a named place", title_fontsize=S.SIZE["legend"] + 1, alignment="left")
    S.credit(fig, f"Named lakes, streams, peaks and towns with a feeling word; lower 48 shown, % count all 50 states  ·  "
                  f"Data: USGS GNIS  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / fname, dpi=200)


def main():
    S.apply()
    d = load()
    s = stats(d)
    for k, v in s.items():
        print(k, {x: y for x, y in v.items() if x != "states"})
    states = gpd.read_file(RAW / "states.zip")
    l48 = states[~states.STUSPS.isin(LOWER48_DROP)].to_crs(5070)

    e = s["emotion"]
    T = {
        "emotion": {
            "title": f"The West has {e['lo_in_west']}% of America's good-feeling place names, but {e['hi_in_west']}% of the bad ones",
            "lede": ("Every US lake, creek, peak and town named for a feeling: blue for good ones like Pleasant "
                     "and Hope, brown for bad ones like Lonesome and Misery. Hover a dot for its name."),
        },
    }
    for view, fname in (("emotion", "feeling_places.png"),):
        lede = T[view]["lede"].replace(" Hover a dot for its name.", "").replace(": blue", ":\nblue")
        png(d, l48, s, view, T[view]["title"], lede, fname)

    # ---- interactive page: points in lon/lat, projected in the browser with d3.geoAlbersUsa ----
    cls = sorted(d.feature_class.unique())
    pts = [[round(r.prim_long_dec, 3), round(r.prim_lat_dec, 3), r.feature_name, cls.index(r.feature_class),
            r.STUSPS, r.county_name if isinstance(r.county_name, str) else "", r.word,
            0 if r.side == "lo" else 1]
           for r in d.itertuples()]
    st = states[states.STUSPS != "PR"].copy()
    st["geometry"] = st.geometry.simplify(0.02)
    data = {
        "points": pts, "classes": cls,
        "states": json_geo(st[["STUSPS", "NAME", "geometry"]]),
        "views": {k: {"lo": VIEWS[k]["lo"][0], "hi": VIEWS[k]["hi"][0], **T[k], "lo_n": s[k]["lo"], "hi_n": s[k]["hi"],
                      "states": s[k]["states"]} for k in VIEWS},
        "day": DAY, "source_url": SOURCE_URL,
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "feeling_places.html")
    print(f"wrote out/feeling_places.png, out/feeling_places.html ({size / 1e6:.1f} MB)")


def json_geo(gdf):
    import json
    return json.loads(gdf.to_json(drop_id=True))


if __name__ == "__main__":
    main()
