"""Day 12 · Power: Robert Moses's expressways, built and unbuilt.

The unbuilt routes are drawn as ghost lines through today's city, with every building that now stands in
their path. Pipeline (each step writes into data/):
  fetch.py    OSM built expressways + guide streets, 2020 census blocks, building footprints near the routes
  routes.py   traces the four unbuilt routes onto today's streets  -> unbuilt_routes.geojson
  analyze.py  buildings / people / homes in a 100 m corridor       -> path_stats.json, path_buildings.geojson
  build.py    this file: out/moses.png and out/moses.html

Data: route histories from nycroads.com (Steve Anderson), after the 1955 Joint Study of Arterial Facilities,
TBTA's 1963 Future Arterial Program and the 1966 TBTA-Blauvelt Cross Brooklyn study (full citations in
SOURCES.md); NYC Building Footprints; 2020 Census blocks; OpenStreetMap.
Also writes out/moses_linkedin.png (1080x1350 for the LinkedIn feed; --linkedin-only skips the rest).
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.ops import linemerge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

META = json.loads((HERE / "meta.json").read_text())
DAY = "Day 12: Power"
CRS_M = "EPSG:32618"
STATS = json.loads((DATA / "path_stats.json").read_text())
TOTAL = STATS["total"]
TITLE = META["title"]
BUILT_C, UNBUILT_C = S.CATEGORICAL[0], S.CATEGORICAL[1]   # categorical order: built, unbuilt
METHOD = ("Unbuilt routes are approximate: traced from published plan descriptions onto today's streets. "
          "Path = a corridor as wide as each plan's right-of-way (30–100 m). Buildings = footprints overlapping it; "
          "people = 2020 census blocks, weighted by the share of each block inside it.")

# Moses-era expressways, with opening dates from each route's nycroads.com history.
# (Staten Island Expressway, 1964, falls outside the frame and is left out.)
BUILT = {
    "Cross Bronx Expressway": "1955–1963 (construction began 1948)",
    "Brooklyn-Queens Expressway": "1939–1964",
    "Gowanus Expressway": "1941 as a parkway, widened 1964",
    "Prospect Expressway": "Mostly complete by 1960",
    "Major Deegan Expressway": "1939–1956",
    "Bruckner Expressway": "1961–1972",
    "Van Wyck Expressway": "1950–1963",
    "Long Island Expressway": "From 1940",
    "Clearview Expressway": "1960–1963",
}


def fmt(n: int) -> str:
    return f"{n:,}"


def load():
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    osm = json.loads((DATA / "osm_built.json").read_text())["elements"]
    from shapely.geometry import LineString
    rows = [{"name": e["tags"]["name"], "geometry": LineString([(g["lon"], g["lat"]) for g in e["geometry"]])}
            for e in osm if e["tags"].get("name") in BUILT and e["tags"].get("highway") == "motorway"]
    ways = gpd.GeoDataFrame(rows, crs=4326).to_crs(CRS_M)
    ways = gpd.clip(ways, boros.buffer(300).union_all())  # the city only (the LIE runs on into Nassau)
    built = ways.dissolve("name").reset_index()
    built["geometry"] = built.geometry.apply(linemerge)
    # dual carriageways count twice in OSM; the centerline length is about half the summed way length
    built["length_mi"] = (ways.assign(l=ways.length).groupby("name").l.sum() / 2 / 1609.344).round(1).values
    built["opened"] = built.name.map(BUILT)
    unbuilt = gpd.read_file(DATA / "unbuilt_routes.geojson").to_crs(CRS_M)
    st = {s["name"]: s for s in STATS["routes"]}
    for k in ("buildings", "buildings_pre1971", "people", "homes"):
        unbuilt[k] = unbuilt.name.map(lambda n: st[n][k])
    corridor = unbuilt.copy()
    corridor["geometry"] = unbuilt.buffer(unbuilt.width_m / 2)
    bldgs = gpd.read_file(DATA / "path_buildings.geojson").to_crs(CRS_M)
    context = gpd.read_file(DATA / "context_lomex.geojson").to_crs(CRS_M)
    return boros, built, unbuilt, corridor, bldgs, context


def frame(boros):
    """The four main boroughs; Staten Island runs off the map (nothing in this story is there)."""
    x0, y0, x1, y1 = boros[boros.boroname != "Staten Island"].total_bounds
    pad = 800
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def draw_routes(ax, boros, built, unbuilt, corridor, bldgs, lw=1.0):
    boros.boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"], zorder=1)
    built.plot(ax=ax, color=BUILT_C, linewidth=1.6 * lw, zorder=3, capstyle="round")
    corridor.plot(ax=ax, color=UNBUILT_C, alpha=0.16, linewidth=0, zorder=2)
    bldgs.plot(ax=ax, color=UNBUILT_C, linewidth=0, zorder=4)
    unbuilt.plot(ax=ax, color=UNBUILT_C, linewidth=1.3 * lw, linestyle=(0, (3, 2)), zorder=5)


def end_point(geom, side: str):
    """The route's westernmost or easternmost vertex (callout leaders attach at the end nearest their label)."""
    pts = [c for g in getattr(geom, "geoms", [geom]) for c in g.coords]
    return (min if side == "west" else max)(pts, key=lambda c: c[0])


def make_png(boros, built, unbuilt, corridor, bldgs, context):
    S.apply()
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.01, 0.07, 0.58, 0.80])
    x0, y0, x1, y1 = frame(boros)
    draw_routes(ax, boros, built, unbuilt, corridor, bldgs)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_axis_off(); ax.set_aspect("equal", adjustable="datalim")  # widen the view: room for labels at the sides

    # one callout per unbuilt route, label level with the route's end, in the empty space beside the city
    most = unbuilt.buildings.max()
    places = {  # (label x in axes fraction, which end of the route, ha)
        "Mid-Manhattan Expressway": (0.0, "west", "left", 0.035),
        "Lower Manhattan Expressway": (0.0, "west", "left", -0.035),
        "Cross-Brooklyn Expressway": (0.0, "west", "left", -0.02),
        "Bushwick Expressway": (1.0, "east", "right", -0.06),
    }
    to_axes = ax.transLimits.transform
    for _, r in unbuilt.iterrows():
        lx, side, ha, dy = places[r["name"]]
        tx, ty = end_point(r.geometry, side)
        fy = to_axes((tx, ty))[1] + dy
        detail = f"{fmt(r.buildings)} buildings and {fmt(r.people)} people\nin its path today"
        if r.buildings == most:
            detail += ", the most of the four"
        S.callout(ax, (tx, ty), (lx, fy), f"{r['name']} ({r['years']})", detail, ha=ha, halo=True)
    S.panel_heading(ax, "Moses's expressways, built and unbuilt",
                    "Unbuilt routes as proposed, drawn through today's city")
    ax.legend(handles=[
        Line2D([], [], color=BUILT_C, lw=1.8, label="Built"),
        Line2D([], [], color=UNBUILT_C, lw=1.4, ls=(0, (3, 2)), label="Unbuilt, with its right-of-way"),
        Rectangle((0, 0), 1, 1, color=UNBUILT_C, lw=0, label="Building standing in the path"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=S.SIZE["legend"], handlelength=2.2)

    # zoom panel: lower Manhattan, the densest path (most buildings per mile)
    name = "Lower Manhattan Expressway"
    r = unbuilt[unbuilt.name == name].iloc[0]
    zb = corridor[corridor.name == name].total_bounds
    zx = fig.add_axes([0.61, 0.24, 0.38, 0.52])
    w_in, h_in = 0.38 * S.MAP["png_size_in"][0], 0.52 * S.MAP["png_size_in"][1]
    cx, half_w = (zb[0] + zb[2]) / 2, (zb[2] - zb[0]) / 2 * 1.08
    half_h = half_w * h_in / w_in
    ylo = zb[1] - 120
    context.plot(ax=zx, color=S.MUTED, alpha=0.22, linewidth=0, zorder=1)
    draw_routes(zx, boros, built, unbuilt, corridor, bldgs, lw=2.0)
    zx.set_xlim(cx - half_w, cx + half_w); zx.set_ylim(ylo, ylo + 2 * half_h)
    zx.set_axis_off(); zx.set_aspect("equal")
    per_mi = r.buildings / r.length_mi
    S.panel_heading(zx, f"{name}, Broome Street",
                    f"{fmt(r.buildings)} buildings, {fmt(r.homes)} homes and {fmt(r.people)} people in its path, "
                    f"about {per_mi:,.0f} buildings a mile")
    ax.add_patch(Rectangle((cx - half_w, ylo), 2 * half_w, 2 * half_h, fill=False, ec=S.MUTED, lw=0.6, zorder=6))
    fig.text(0.61, 0.20, f"{fmt(TOTAL['buildings_pre1971'])} of the {fmt(TOTAL['buildings'])} buildings in the four paths "
                         "were already standing when the plans died in 1971.\nGrey: other buildings nearby today.",
             fontsize=S.SIZE["subtitle"], color=S.MUTED, va="top")

    S.title(fig, TITLE)
    S.credit(fig, f"{METHOD}\nRoutes: nycroads.com, after the 1955 Joint Study of Arterial Facilities (LOMEX, Mid-Manhattan), "
                  "TBTA's 1963 Future Arterial Program (Bushwick) and 1966 TBTA–Blauvelt study (Cross-Brooklyn)\n"
                  f"NYC Building Footprints · 2020 Census · OpenStreetMap  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "moses.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


def linkedin(boros, built, unbuilt, corridor, bldgs):
    """4:5 portrait (1080x1350) for the LinkedIn feed, type sized for a phone: the headline, the totals,
    then Manhattan and Brooklyn cropped to the four unbuilt routes, each labeled with its count."""
    S.apply()
    fig = plt.figure(figsize=(5.4, 6.75))
    fig.text(0.5, 0.975, TITLE.replace(" in the path of ", "\nin the path of ", 1),
             ha="center", va="top", fontsize=14, linespacing=1.25,
             fontweight="bold", color=S.INK, fontfamily=S.HEAD_FONT)
    for cx, n, what in ((0.2, TOTAL["buildings"], "buildings"), (0.5, TOTAL["homes"], "homes"),
                        (0.8, TOTAL["people"], "people")):
        fig.text(cx, 0.865, fmt(n), ha="center", fontsize=18, fontweight="bold", fontfamily=S.HEAD_FONT, color=UNBUILT_C)
        fig.text(cx, 0.855, what, ha="center", va="top", fontsize=9, color=S.MUTED)
    fig.text(0.5, 0.815, "standing today in the path of four expressways that were never built",
             ha="center", va="top", fontsize=9, color=S.MUTED)

    ax = fig.add_axes([0.0, 0.13, 1.0, 0.65])
    draw_routes(ax, boros, built, unbuilt, corridor, bldgs, lw=1.2)
    x0, y0, x1, y1 = unbuilt.total_bounds
    ax.set_xlim(x0 - 8500, x1 + 500); ax.set_ylim(y0 - 2000, y1 + 2500)
    ax.set_axis_off(); ax.set_aspect("equal", adjustable="datalim")
    PLAN_YEAR = {"Mid-Manhattan Expressway": 1955, "Lower Manhattan Expressway": 1955,
                 "Bushwick Expressway": 1963, "Cross-Brooklyn Expressway": 1966}
    places = {  # label (axes fraction), point along the route (fraction of its length), ha
        "Mid-Manhattan Expressway": ((0.02, 0.86), 0.05, "left"),
        "Lower Manhattan Expressway": ((0.02, 0.64), 0.05, "left"),
        "Bushwick Expressway": ((0.98, 0.58), 0.6, "right"),
        "Cross-Brooklyn Expressway": ((0.30, 0.07), 0.3, "left"),
    }
    for _, r in unbuilt.iterrows():
        xy, t, ha = places[r["name"]]
        g = r.geometry if r.geometry.geom_type == "LineString" else max(r.geometry.geoms, key=lambda l: l.length)
        p = g.interpolate(t, normalized=True)
        S.callout(ax, (p.x, p.y), xy, f"{r['name']}",
                  f"{fmt(r.buildings)} buildings · {PLAN_YEAR[r['name']]} plan", ha=ha, halo=True)
    fig.legend(handles=[
        Line2D([], [], color=BUILT_C, lw=1.8, label="Built Moses expressway"),
        Line2D([], [], color=UNBUILT_C, lw=1.4, ls=(0, (3, 2)), label="Unbuilt"),
        Rectangle((0, 0), 1, 1, color=UNBUILT_C, lw=0, label="Building in its path"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False, fontsize=9,
        handlelength=1.8, handletextpad=0.4, columnspacing=1.2)
    S.credit(fig, "Routes traced from the published plans; path as wide as each plan's right-of-way.\n"
                  "nycroads.com · NYC Building Footprints · 2020 Census  ·  "
                  f"#30DayMapChallenge {DAY}\ndaniellecarrick.github.io/30-day-map-challenge/day12", y=0.014)
    fig.savefig(OUT / "moses_linkedin.png", dpi=200)
    plt.close(fig)


def page_data(boros, built, unbuilt, corridor, bldgs):
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(25)
    bu = built[["name", "opened", "length_mi", "geometry"]].copy()
    bu["geometry"] = bu.geometry.simplify(15)
    keep = ["name", "short", "route", "borough", "years", "fate", "plan", "source", "length_mi", "width_m", "width_note",
            "buildings", "buildings_pre1971", "people", "homes"]
    un = unbuilt[keep + ["geometry"]].copy()
    un["geometry"] = un.geometry.simplify(3)
    co = corridor[["name", "geometry"]].copy()
    co["geometry"] = co.geometry.simplify(3)
    bl = bldgs.copy()
    bl["geometry"] = bl.geometry.simplify(1.5)
    bl["yr"] = bl.construction_year.fillna(0).astype(int)
    bl = bl[["route", "yr", "geometry"]]
    return {
        "boroughs": json.loads(b.to_json(drop_id=True)),
        "built": json.loads(bu.to_json(drop_id=True)),
        "unbuilt": json.loads(un.to_json(drop_id=True)),
        "corridor": json.loads(co.to_json(drop_id=True)),
        "buildings": json.loads(bl.to_json(drop_id=True)),
        "total": TOTAL, "title": TITLE, "day": DAY, "method": METHOD,
        "built_mi": round(float(built.length_mi.sum()), 0),
    }


def main():
    boros, built, unbuilt, corridor, bldgs, context = load()
    linkedin(boros, built, unbuilt, corridor, bldgs)
    if "--linkedin-only" in sys.argv:
        return
    make_png(boros, built, unbuilt, corridor, bldgs, context)
    size = S.build_page(HERE / "web" / "index.html", page_data(boros, built, unbuilt, corridor, bldgs), OUT / "moses.html")
    # dev copy so web/index.html works when opened from disk
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(page_data(boros, built, unbuilt, corridor, bldgs)) + ";")
    print("wrote out/moses.png and out/moses.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
