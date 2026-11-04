"""Day 23 · Senses: Taste — where each cuisine is over-represented in NYC.

For every 400 m hex, look at all restaurants within 500 m of its center and compare a cuisine's share
there with its share citywide (location quotient: 2× = twice as common as in NYC overall).
Writes:
  - out/nyc_cuisine_capitals.png    (static, small multiples: 8 cuisines, one panel each)
  - out/nyc_cuisine_capitals.html   (interactive: every cuisine with 100+ restaurants, picked from a dropdown,
                                     with a side panel of numbers + a context note from context.json)

Data: NYC DOHMH Restaurant Inspection Results (43nn-pn8j), 2020 NTAs (9nt8-h7nd). Get it with ./fetch.sh
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 23: Senses: Taste"
SOURCE_URL = "https://data.cityofnewyork.us/d/43nn-pn8j"
CRS_M = "EPSG:32618"   # UTM 18N, metres
ACTIVE_SINCE = "2023-01-01"   # inspected since then -> treat as open
HEX = 400              # hex size, flat-to-flat, metres
R = 500                # neighborhood radius around each hex center, metres
MIN_N = 30             # fewer restaurants than this within R -> not enough to judge (grey)
PEAK_MIN_N = 40        # a "capital" must have at least this many restaurants within R

MIN_CUISINE = 100      # cuisines with fewer restaurants than this are too thin to map
SKIP = {"Coffee/Tea", "Other"}   # coffee: not a cuisine for this map; "Other": no meaning to map
# nicer names for DOHMH labels (anything not listed is shown as is)
DISPLAY = {
    "Jewish/Kosher": "Kosher",
    "Bakery Products/Desserts": "Bakeries & desserts",
    "Juice, Smoothies, Fruit Salads": "Juice & smoothies",
    "Bagels/Pretzels": "Bagels & pretzels",
    "Sandwiches/Salads/Mixed Buffet": "Sandwiches, salads & buffets",
    "Asian/Asian Fusion": "Asian & Asian fusion",
    "Frozen Desserts": "Ice cream & frozen desserts",
    "Chicken": "Chicken",
}
# the neighborhood people expect to win, for the cuisines that have one: (name, lon, lat of its landmark corner).
# It is scored at its best hex within R of that corner, so it gets a fair shot.
FAMOUS = {
    "Chinese": ("Chinatown", -73.9975, 40.7158),       # Mott & Canal
    "Italian": ("Little Italy", -73.9973, 40.7193),    # Mulberry & Grand
    "Korean": ("Koreatown", -73.9874, 40.7478),        # W 32nd St & Broadway
}
# the PNG's 8 panels (small multiples); the page offers every cuisine
PNG_PANELS = ["Chinese", "Italian", "Korean", "Eastern European", "Caribbean", "Jewish/Kosher", "Latin American", "Mexican"]
# dropdown groups by how much of a cuisine sits in a cluster (share of its restaurants at 3x+ the city share)
CLUSTER_X = 3
GROUPS = [(0.4, "Tightly clustered"), (0.2, "Partly clustered"), (0.0, "Spread across the city")]
# NTA names are long; shorten for labels
SHORT = {
    "Murray Hill-Broadway Flushing": "Murray Hill, Queens",
    "Murray Hill-Kips Bay": "Murray Hill, Manhattan",
    "Midtown South-Flatiron-Union Square": "Koreatown / Flatiron",
    "SoHo-Little Italy-Hudson Square": "SoHo / Little Italy",
    "Sunset Park (East)-Borough Park (West)": "Sunset Park (East)",
    "Flushing-Willets Point": "Flushing",
    "Chinatown-Two Bridges": "Chinatown",
    "Carroll Gardens-Cobble Hill-Gowanus-Red Hook": "Carroll Gardens / Cobble Hill",
    "Gravesend (East)-Homecrest": "Homecrest",
    "East Flatbush-Rugby": "East Flatbush",
    "East Flatbush-Erasmus": "East Flatbush",
    "East Flatbush-Remsen Village": "East Flatbush",
    "East Flatbush-Farragut": "East Flatbush",
    "Mott Haven-Port Morris": "Mott Haven",
    "Belmont": "Belmont (Arthur Ave)",
    "Astoria (North)-Ditmars-Steinway": "Astoria (Ditmars–Steinway)",
    "Astoria (East)-Woodside (North)": "Astoria (East)",
    "Sheepshead Bay-Manhattan Beach-Gerritsen Beach": "Sheepshead Bay",
    "Financial District-Battery Park City": "Financial District",
    "Upper East Side-Carnegie Hill": "Upper East Side (Carnegie Hill)",
    "Upper East Side-Lenox Hill-Roosevelt Island": "Upper East Side (Lenox Hill)",
    "Upper West Side-Lincoln Square": "Lincoln Square",
    "Downtown Brooklyn-DUMBO-Boerum Hill": "Downtown Brooklyn",
    "Concourse-Concourse Village": "Concourse (Yankee Stadium)",
    "Yankee Stadium-Macombs Dam Park": "Yankee Stadium",
    "East Midtown-Turtle Bay": "East Midtown",
    "Midtown-Times Square": "Midtown",
    "Chelsea-Hudson Yards": "Chelsea",
    "Soundview-Bruckner-Bronx River": "Soundview",
    "New Dorp-Midland Beach": "New Dorp",
    "Coney Island-Sea Gate": "Coney Island",
    "St. George-New Brighton": "St. George",
    "Tribeca-Civic Center": "Tribeca",
    "Long Island City-Hunters Point": "Long Island City",
    "Prospect Lefferts Gardens-Wingate": "Prospect Lefferts Gardens",
}

# Location quotient classes: below 1.5× blends into the land; the sequential blue ramp above it.
BINS = [1.5, 2, 3, 5, 10]
LABELS = ["1.5×", "2×", "3×", "5×", "10×"]
RAMP = [S.SEQUENTIAL_BLUE[i] for i in (1, 2, 3, 4, 6)]   # 5 of the 7 blues, one per bin
LAND = S.C["ctl"]                  # land: under 1.5× the city share, or too few restaurants to say


def short(name: str) -> str:
    return SHORT.get(name, name)


def display(key: str) -> str:
    return DISPLAY.get(key, key)


def load():
    r = pd.read_csv(DATA / "restaurants.csv", dtype=str)
    total = len(r)
    r["last_insp"] = pd.to_datetime(r.last_insp)
    r[["latitude", "longitude"]] = r[["latitude", "longitude"]].astype(float)
    r = r.sort_values("last_insp").drop_duplicates("camis", keep="last")
    r = r[r.latitude.between(40.45, 40.95) & (r.last_insp >= ACTIVE_SINCE)]
    g = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.longitude, r.latitude), crs=4326).to_crs(CRS_M)
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    land = boros.geometry.simplify(20).union_all()
    near_land = land.buffer(50)
    shapely.prepare(near_land)
    g = g[shapely.contains_xy(near_land, g.geometry.x, g.geometry.y)].reset_index(drop=True)
    nta = gpd.read_file(DATA / "nta2020.geojson").to_crs(CRS_M)[["ntaname", "geometry"]]
    # each restaurant's neighborhood (nearest NTA, so restaurants on piers still get one)
    g["ntaname"] = gpd.sjoin_nearest(g[["geometry"]], nta, how="left").groupby(level=0).ntaname.first()
    print(f"{len(g):,} active restaurants (of {total:,} in the file)")
    return g, boros, land, nta, total


def hexgrid(land) -> gpd.GeoDataFrame:
    """Pointy-top hexes HEX metres wide (flat side to flat side) covering the land."""
    s = HEX / np.sqrt(3)                  # side length
    xmin, ymin, xmax, ymax = land.bounds
    xs, ys, rows = np.arange(xmin, xmax + HEX, HEX), np.arange(ymin, ymax + 1.5 * s, 1.5 * s), []
    for j, y in enumerate(ys):
        off = HEX / 2 if j % 2 else 0
        rows += [(x + off, y) for x in xs]
    c = np.array(rows)
    ang = np.deg2rad(30 + 60 * np.arange(6))
    polys = [shapely.Polygon(np.c_[x + s * np.cos(ang), y + s * np.sin(ang)]) for x, y in c]
    h = gpd.GeoDataFrame({"x": c[:, 0], "y": c[:, 1]}, geometry=polys, crs=CRS_M)
    shapely.prepare(land)
    return h[shapely.intersects(land, h.geometry.values)].reset_index(drop=True)


def pairs(lists):
    """Neighbor lists -> flat (row, col) arrays, so per-cuisine counts are one bincount."""
    lens = np.array([len(i) for i in lists])
    return lens, np.repeat(np.arange(len(lists)), lens), np.concatenate([np.asarray(i, dtype=int) for i in lists])


def main():
    S.apply()
    g, boros, land, nta, total = load()
    city = g.cuisine_description.value_counts(normalize=True)
    counts = g.cuisine_description.value_counts()
    context = json.loads((HERE / "context.json").read_text())
    keys = [c for c, n in counts.items() if n >= MIN_CUISINE and c not in SKIP]
    missing = [k for k in keys if k not in context]
    assert not missing, f"add context for: {missing}"

    hexes = hexgrid(land)
    xy = np.c_[g.geometry.x, g.geometry.y]
    tree = cKDTree(xy)
    hn, hrows, hcols = pairs(tree.query_ball_point(np.c_[hexes.x, hexes.y], R))
    hexes["n"] = hn
    rn, rrows, rcols = pairs(tree.query_ball_point(xy, R))
    cuis = g.cuisine_description.values
    # a hex is named for the neighborhood most of its nearby restaurants are in (center's NTA if none)
    centers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(hexes.x, hexes.y), crs=CRS_M)
    center_nta = gpd.sjoin_nearest(centers, nta, how="left").groupby(level=0).ntaname.first().values
    mode = pd.Series(g.ntaname.values[hcols]).groupby(hrows).agg(lambda s: s.value_counts().index[0])
    hexes["nta"] = [mode.get(i, center_nta[i]) for i in range(len(hexes))]
    hexes["label"] = hexes.nta.map(short)
    print(f"{len(hexes):,} hexes; {(hexes.n >= MIN_N).sum():,} have {MIN_N}+ restaurants within {R} m")

    views = []
    for key in keys:
        k = np.bincount(hrows[cuis[hcols] == key], minlength=len(hexes))
        share = k / np.maximum(hexes.n.values, 1)
        lq = share / city[key]
        cand = np.where(hexes.n.values >= PEAK_MIN_N, lq, -1)
        order = np.argsort(-cand)
        peak = int(order[0])
        famous_name = famous_i = None
        if key in FAMOUS:
            # the namesake at its strongest: best hex within R of its landmark corner (with enough restaurants)
            famous_name, lon, lat = FAMOUS[key]
            pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(CRS_M)[0]
            d = np.hypot(hexes.x - pt.x, hexes.y - pt.y).values
            famous_i = int(np.argmax(np.where((d <= R) & (hexes.n.values >= PEAK_MIN_N), lq, -1)))
        # runners-up: the next best spots in other neighborhoods (not the peak's or the namesake's)
        also, seen = [], {hexes.label[peak]} | ({hexes.label[famous_i]} if famous_i is not None else set())
        taken = [peak] + ([famous_i] if famous_i is not None else [])
        for i in order[1:]:
            if cand[i] < BINS[0] or len(also) == 3:
                break
            far = all(np.hypot(hexes.x[i] - hexes.x[j], hexes.y[i] - hexes.y[j]) > 1000 for j in taken)
            if hexes.label[i] not in seen and far:   # a different neighborhood, not the same cluster renamed
                seen.add(hexes.label[i])
                taken.append(int(i))
                also.append(int(i))
        # how clustered: share of this cuisine's restaurants whose OTHER neighbors within 500 m are 3x+ the city
        # share for it (leaving the restaurant itself out, so a lone steakhouse doesn't count as a cluster)
        same = np.bincount(rrows[cuis[rcols] == key], minlength=len(g)) - (cuis == key)
        own = same / np.maximum(rn - 1, 1) / city[key]
        clustered = float((own[cuis == key] >= CLUSTER_X).mean())
        v = {"key": key, "label": display(key), "city": float(city[key]), "count": int(counts[key]),
             "k": k, "share": share, "lq": lq, "peak": peak, "also": also, "clustered": clustered,
             "famous": famous_name, "famous_i": famous_i, "context": context[key]}
        views.append(v)
        print(f"{v['label']:>30}: {v['city']:.1%} (n={v['count']}) · {clustered:.0%} clustered · peak "
              f"{hexes.label[peak]} {share[peak]:.0%} of {hexes.n[peak]} = {lq[peak]:.1f}× · also "
              + ", ".join(f"{hexes.label[i]} {share[i]:.0%}" for i in also))

    # headline: every namesake neighborhood loses to somewhere else
    for v in views:
        if v["famous"]:
            assert v["share"][v["peak"]] > v["share"][v["famous_i"]], v["label"]
    title = "Chinatown, Little Italy and Koreatown aren't where their cuisines are most concentrated"
    by_key = {v["key"]: v for v in views}
    png(boros, land, hexes, [by_key[k] for k in PNG_PANELS], title)
    web(g, boros, hexes, views, title, total)


def bin_color(lq: float, n: int) -> str:
    if n < MIN_N:
        return LAND
    i = np.searchsorted(BINS, lq, side="right") - 1
    return LAND if i < 0 else RAMP[i]


# where each panel's peak label sits (axes fraction, ha); default is the open water bottom-right
PEAK_POS = {"Italian": (0.99, 0.76, "right")}
PEAK_NAME = {"Italian": "Arthur Ave"}   # shorter label where the PNG has little room


def png(boros, land, hexes, panels, title):
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    clipped = hexes.copy()
    clipped["geometry"] = clipped.geometry.intersection(land)
    xmin, ymin, xmax, ymax = boros.total_bounds
    pad = 600
    # 2 rows x 4 panels under the title block
    left, right, bottom, top, gap_x, gap_y = 0.015, 0.985, 0.04, 0.80, 0.006, 0.02
    w = (right - left - 3 * gap_x) / 4
    h = (top - bottom - gap_y) / 2
    for idx, p in enumerate(panels):
        r, c = divmod(idx, 4)
        ax = fig.add_axes([left + c * (w + gap_x), top - (r + 1) * h - r * gap_y, w, h])
        colors = [bin_color(l, n) for l, n in zip(p["lq"], hexes.n)]
        clipped.plot(ax=ax, color=colors, linewidth=0)
        boros.boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"] * 0.8)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_axis_off()
        ax.set_aspect("equal")
        S.panel_heading(ax, p["label"], f"{p['city']:.1%} of NYC restaurants")
        pk = p["peak"]
        px, py, pha = PEAK_POS.get(p["key"], (0.98, 0.10, "right"))
        S.callout(ax, (hexes.x[pk], hexes.y[pk]), (px, py),
                  f"{PEAK_NAME.get(p['key'], hexes.label[pk])}: {p['share'][pk]:.0%}",
                  (f"of {hexes.n[pk]} · {p['lq'][pk]:.1f}× NYC" if p["key"] in PEAK_POS
                   else f"of {hexes.n[pk]} nearby restaurants · {p['lq'][pk]:.1f}× NYC"),
                  ha=pha, outline=hexes.geometry[pk], halo=True)
        if p["famous"]:
            f = p["famous_i"]
            S.callout(ax, (hexes.x[f], hexes.y[f]), (0.02, 0.62),
                      f"{p['famous']}: {p['share'][f]:.0%}",
                      f"of {hexes.n[f]} · {p['lq'][f]:.1f}× NYC",
                      ha="left", outline=hexes.geometry[f], halo=True)

    fig.text(0.015, 0.975, title, ha="left", va="top", fontsize=S.SIZE["title"], fontweight="bold",
             fontfamily=S.HEAD_FONT, color=S.INK)
    fig.text(0.015, 0.918,
             f"How much more common each cuisine is near every spot in the city than in NYC overall: its share of all "
             f"restaurants within {R} m,\ndivided by its share citywide. Outlined: the most concentrated spot "
             f"(at least {PEAK_MIN_N} restaurants nearby) and, where there is one, the namesake neighborhood.",
             ha="left", va="top", fontsize=S.SIZE["subtitle"] + 1.5, color=S.MUTED, linespacing=1.45)
    handles = [
        Patch(color=c, label=(f"{a}–{b}" if b else f"{a}+")) for c, a, b in
        zip(RAMP, LABELS, LABELS[1:] + [None])]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.86), ncol=5, fontsize=S.SIZE["legend"] + 1,
               title=f"How many times the city share (uncolored: under {LABELS[0]}, or fewer than {MIN_N} restaurants nearby)", title_fontsize=S.SIZE["legend"] + 1, alignment="left",
               handlelength=1.4, handleheight=1.0, columnspacing=1.2, borderaxespad=0, labelcolor=S.INK)
    S.credit(fig, f"Restaurants with an inspection since {ACTIVE_SINCE[:4]}, by the cuisine they registered; {HEX} m hexes, "
                  f"each scored on restaurants within {R} m of its center.  Data: NYC DOHMH Restaurant Inspection "
                  f"Results (Nov 2026)  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_cuisine_capitals.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", OUT / "nyc_cuisine_capitals.png")


def group_for(clustered: float) -> str:
    return next(name for lo, name in GROUPS if clustered >= lo)


def web(g, boros, hexes, views, title, total):
    labels = sorted(set(hexes.label))
    lab_i = {n: i for i, n in enumerate(labels)}
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(25)
    data = {
        "title": title, "day": DAY, "source_url": SOURCE_URL, "total": total, "n_rest": len(g),
        "hex": HEX, "r": R, "min_n": MIN_N, "peak_min_n": PEAK_MIN_N, "since": ACTIVE_SINCE[:4],
        "bins": BINS, "labels": LABELS, "ramp": RAMP, "cluster_x": CLUSTER_X, "min_cuisine": MIN_CUISINE,
        "boroughs": json.loads(b.to_json(drop_id=True)),
        "places": labels,
        # hexes as flat arrays: center x, y (m), restaurants within R, place-name index
        "hx": hexes.x.round(0).astype(int).tolist(), "hy": hexes.y.round(0).astype(int).tolist(),
        "hn": hexes.n.astype(int).tolist(), "hplace": [lab_i[n] for n in hexes.label],
        "cuisines": [],
    }
    # dropdown order: by group (most clustered first), then alphabetical
    rank = {name: i for i, (_, name) in enumerate(GROUPS)}
    for v in sorted(views, key=lambda v: (rank[group_for(v["clustered"])], v["label"])):
        sel = g[g.cuisine_description == v["key"]]
        nz = np.flatnonzero(v["k"])
        data["cuisines"].append({
            "key": v["key"], "label": v["label"], "group": group_for(v["clustered"]),
            "city": round(v["city"], 4), "count": v["count"], "clustered": round(v["clustered"], 3),
            "context": v["context"], "peak": v["peak"], "also": v["also"],
            "famous": v["famous"], "famous_i": v["famous_i"],
            "ki": nz.tolist(), "kv": v["k"][nz].astype(int).tolist(),   # restaurants of this cuisine near each hex (sparse)
            # restaurants: x, y flat + names (title case) for the dot tooltip
            "pts": np.c_[sel.geometry.x, sel.geometry.y].round(0).astype(int).ravel().tolist(),
            "names": sel.dba.str.title().str.replace(r"\s+", " ", regex=True).str.strip().tolist(),
        })
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "nyc_cuisine_capitals.html")
    print("wrote", OUT / "nyc_cuisine_capitals.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
