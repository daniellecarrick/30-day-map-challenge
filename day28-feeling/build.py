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


LABELS = [  # (name, state) of a few standout names to label on the LinkedIn image
    # (name, state, label offset in points)
    ("Disappointment Peak", "WY", (4, 3)), ("Mount Despair", "WA", (4, 3)), ("Good Grief", "ID", (4, -8)),
    ("Confusion Range", "UT", (4, 3)), ("Mount Misery", "CT", (-4, 4)), ("Delight", "MD", (4, 3)),
    ("Happy Hollow", "AR", (4, 3)),
]


def linkedin(d, states, s):
    """4:5 portrait (1080x1350) for LinkedIn, type sized for a phone."""
    from matplotlib import patheffects
    e = s["emotion"]
    v = d[(d.view == "emotion") & ~d.STUSPS.isin(LOWER48_DROP)]
    pts = gpd.GeoDataFrame(v, geometry=gpd.points_from_xy(v.prim_long_dec, v.prim_lat_dec), crs=4326).to_crs(5070)
    lo, hi = pts[pts.side == "lo"], pts[pts.side == "hi"]
    blue, brown = S.DIVERGING[0], S.DIVERGING[3]
    fig = plt.figure(figsize=(5.4, 6.75))
    ax = fig.add_axes([0.01, 0.25, 0.98, 0.53])
    states.plot(ax=ax, color=S.C["ctl"], edgecolor=S.BG, linewidth=0.4)
    lo.plot(ax=ax, color=blue, markersize=1.1, alpha=0.85, linewidth=0)
    hi.plot(ax=ax, color=brown, markersize=1.6, alpha=0.95, linewidth=0)
    halo = [patheffects.withStroke(linewidth=2.2, foreground=S.BG)]
    for name, st, off in LABELS:
        m = pts[(pts.feature_name == name) & (pts.STUSPS == st)]
        if m.empty:
            continue
        p = m.geometry.iloc[0]
        ax.plot(p.x, p.y, "o", ms=3.2, mfc=blue if m.side.iloc[0] == "lo" else brown, mec=S.INK, mew=0.6, zorder=5)
        ax.annotate(name, (p.x, p.y), xytext=off, textcoords="offset points", fontsize=6.2,
                    ha="right" if off[0] < 0 else "left",
                    fontweight="semibold", color=S.INK, path_effects=halo, zorder=6)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ratio = e["lo"] / e["hi"]
    fig.text(0.5, 0.972, f"For every American place named\nfor a bad feeling, there are\n{ratio:.0f} named for a good one",
             ha="center", va="top", fontsize=19, fontweight="bold", fontfamily=S.HEAD_FONT, color=S.INK,
             linespacing=1.15)
    fig.text(0.5, 0.815, "Every US lake, creek, peak and town named for a feeling",
             ha="center", va="center", fontsize=8.6, color=S.MUTED)
    # two big counts under the map, in each side's color, with each side's top words
    top = lambda d_: ", ".join(f"{w.title()} {n:,}" for w, n in d_.items())
    for x, n, color, label, words in ((0.27, e["lo"], blue, "named for a good feeling", top(e["top_lo"])),
                                      (0.73, e["hi"], brown, "named for a bad feeling", top(e["top_hi"]))):
        fig.text(x, 0.205, f"{n:,}", ha="center", va="top", fontsize=30, fontweight="bold",
                 fontfamily=S.HEAD_FONT, color=color)
        fig.text(x, 0.135, f"places {label}\n{words}", ha="center", va="top", fontsize=7.6, color=S.INK,
                 linespacing=1.45)
    fig.text(0.5, 0.018, "Names from the USGS Geographic Names Information System; lower 48 shown, counts are all 50 states\n"
             f"#30DayMapChallenge {DAY}  ·  daniellecarrick.github.io/30-day-map-challenge/day28",
             ha="center", va="bottom", fontsize=6.2, color=S.MUTED, linespacing=1.4)
    fig.savefig(OUT / "feeling_places_linkedin.png", dpi=200)
    plt.close(fig)
    print("wrote out/feeling_places_linkedin.png")


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
            "title": f"For every American place named for a bad feeling, there are {e['lo'] / e['hi']:.0f} named for a good one",
            "lede": ("Every US lake, creek, peak and town named for a feeling: blue for good ones like Pleasant "
                     "and Hope, brown for bad ones like Lonesome and Misery. Hover a dot for its name."),
        },
    }
    for view, fname in (("emotion", "feeling_places.png"),):
        lede = T[view]["lede"].replace(" Hover a dot for its name.", "").replace(": blue", ":\nblue")
        png(d, l48, s, view, T[view]["title"], lede, fname)
    linkedin(d, l48, s)

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
