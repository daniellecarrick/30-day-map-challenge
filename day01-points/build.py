"""Day 1 · Points — Which NYC subway stations can you use without stairs?

Every subway station as a dot in its line's color: filled = fully ADA accessible,
lightly filled = accessible in one direction only, ring = not accessible.
Stations shared by several trunk lines are split into wedges, one per line color.
A pictogram beside the map stacks every line's stations to show % accessible by line.
Writes:
  - out/nyc_subway_accessible.png    (static, for posting)
  - out/nyc_subway_accessible.html   (interactive, self-contained)
  - out/stations.geojson             (lon/lat, for other tools)

Data: MTA Subway Stations (data.ny.gov 39hk-dx4f, listed on NYC Open Data),
      Borough Boundaries (NYC Open Data gthc-hcne).
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Wedge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 1: Points"
SOURCE_URL = "https://data.ny.gov/d/39hk-dx4f"
CRS_M = "EPSG:32618"  # UTM 18N, metres
INK, MUTED, BG = S.INK, S.MUTED, S.BG

# Route -> trunk line (the color family riders know: 4/5/6 = green, etc.)
TRUNK = {**dict.fromkeys("123", "123"), **dict.fromkeys("456", "456"), "7": "7",
         **dict.fromkeys("ACE", "ACE"), **dict.fromkeys("BDFM", "BDFM"), "G": "G",
         "J": "JZ", "Z": "JZ", "L": "L", **dict.fromkeys("NQRW", "NQRW"), "S": "S", "SIR": "SIR"}
ROUTE_ORDER = list("1234567ACEBDFMG") + ["J", "Z", "L", "N", "Q", "R", "W", "S", "SIR"]
TRUNK_ORDER = list(S.SUBWAY)
STATUS = {0: "Not accessible", 1: "Fully accessible", 2: "Accessible in one direction"}
BORO = {"M": "Manhattan", "Bk": "Brooklyn", "Bx": "Bronx", "Q": "Queens", "SI": "Staten Island"}


def load():
    st = pd.DataFrame(json.loads((DATA / "stations.json").read_text()))
    st["ada"] = st.ada.astype(int)
    st["routes"] = st.daytime_routes.str.split().apply(lambda rs: sorted(rs, key=ROUTE_ORDER.index))
    st["trunks"] = st.routes.apply(lambda rs: list(dict.fromkeys(TRUNK[r] for r in rs)))
    st["ada_notes"] = st.ada_notes.replace("NaN", "").fillna("")
    st["borough"] = st.borough.map(BORO)
    st = gpd.GeoDataFrame(
        st[["station_id", "stop_name", "line", "borough", "structure", "routes", "trunks", "ada", "ada_notes"]],
        geometry=gpd.points_from_xy(st.gtfs_longitude.astype(float), st.gtfs_latitude.astype(float)),
        crs="EPSG:4326").to_crs(CRS_M)
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    return st, boros


def by_route(st) -> pd.DataFrame:
    """One row per route: how many of its stations are fully / one-direction / not accessible.
    A station served by several routes counts once in each route's row."""
    r = st.explode("routes").rename(columns={"routes": "route"})
    g = r.groupby("route").ada.agg(n="size", full=lambda s: int((s == 1).sum()),
                                   part=lambda s: int((s == 2).sum()))
    g["none"] = g.n - g.full - g.part
    g["pct"] = 100 * g.full / g.n
    g["trunk"] = g.index.map(TRUNK)
    g["order"] = g.index.map(ROUTE_ORDER.index)
    return g.sort_values(["pct", "n"], ascending=[False, False])


def color(trunk):
    return S.SUBWAY[trunk]["color"]


def draw_marker(ax, x, y, trunks, ada, r, ring, zorder=3):
    """Station marker: wedges per trunk. Filled if accessible, ring if not, both (light fill +
    ring) if accessible in one direction only."""
    n = len(trunks)
    for i, t in enumerate(trunks):
        a0, a1 = 90 - 360 * (i + 1) / n, 90 - 360 * i / n
        if n == 1:
            a0, a1 = 0, 360
        c = color(t)
        if ada in (1, 2):
            ax.add_patch(Wedge((x, y), r, a0, a1, facecolor=c, edgecolor="none",
                               alpha=1 if ada == 1 else 0.35, zorder=zorder))
        if ada in (0, 2):
            ax.add_patch(Wedge((x, y), r, a0, a1, width=ring, facecolor=c, edgecolor="none", zorder=zorder + 1))
    if ada == 1:  # thin paper edge so touching filled dots stay separate
        ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=BG, linewidth=0.5, zorder=zorder + 2))


def bullet(ax, x, y, route, r, fs):
    t = S.SUBWAY[TRUNK[route]]
    ax.add_patch(Circle((x, y), r, facecolor=t["color"], edgecolor="none", zorder=3))
    ax.text(x, y - r * 0.05, "SIR" if route == "SIR" else route, ha="center", va="center", color=t["text"],
            fontsize=fs * (0.62 if route == "SIR" else 1), fontweight="bold", zorder=4)


def pictogram(ax, st, routes):
    """One row per route, sorted by % fully accessible. Each dot is one station on that route:
    filled first, then one-direction, then rings."""
    ROW, R, RING = 2.0, 0.40, 0.13
    X0 = 5.4  # first dot
    for i, (route, row) in enumerate(routes.iterrows()):
        y = -i * ROW
        bullet(ax, 0.55, y, route, 0.55, 9)
        ax.text(4.5, y, f"{row.pct:.0f}%", ha="right", va="center", fontsize=S.SIZE["legend"],
                fontweight="semibold", color=INK)
        seq = [1] * row.full + [2] * row.part + [0] * row.none
        for j, a in enumerate(seq):
            draw_marker(ax, X0 + j, y, [row.trunk], a, R, RING)
    # call out the most and least accessible lines
    best, worst = routes.iloc[0], routes.iloc[-1]
    for k, (route, row) in [(0, (routes.index[0], best)), (len(routes) - 1, (routes.index[-1], worst))]:
        ax.text(X0 + row.n + 0.6, -k * ROW, f"{route}: {row.full} of {row.n} stations",
                ha="left", va="center", fontsize=S.SIZE["annotation"], color=MUTED)
    ax.set_xlim(-0.2, X0 + routes.n.max() + 0.6)
    ax.set_ylim(-(len(routes) - 1) * ROW - 0.9, 0.9)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    S.apply()
    st, boros = load()
    routes = by_route(st)
    n, full, part = len(st), int((st.ada == 1).sum()), int((st.ada == 2).sum())
    pct = 100 * full / n
    worst = routes.iloc[-2:].index.tolist()[::-1]  # the two lowest
    print(f"{n} stations · fully accessible {full} ({pct:.1f}%) · one direction {part}")
    print(routes[["n", "full", "part", "pct"]].round(1).to_string())

    title = f"Only a third of NYC subway stations can be used without stairs"
    lede = (f"{full} of {n} stations are fully accessible and {part} work in one direction only. "
            f"The {routes.index[0]} is the most accessible line ({routes.pct.iloc[0]:.0f}%); "
            f"on the {worst[0]} and {worst[1]}, fewer than 1 in 4 stations are.")

    # ---- static PNG ----
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.0, 0.07, 0.56, 0.80])
    xmin, ymin, xmax, ymax = boros.total_bounds
    ax.set_xlim(xmin - 600, xmax + 600)
    ax.set_ylim(ymin - 600, ymax + 600)
    boros.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor="none")
    boros.boundary.plot(ax=ax, color=INK, linewidth=S.MAP["boundary_width"] * 0.6, alpha=0.5)
    # draw inaccessible first so filled dots sit on top
    for _, s in st.sort_values("ada", key=lambda a: a.map({0: 0, 2: 1, 1: 2})).iterrows():
        draw_marker(ax, s.geometry.x, s.geometry.y, s.trunks, s.ada, 215, 65)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # legend in the empty New Jersey space, top left
    lx, ly = 0.03, 0.93
    ax.text(lx, ly, "Each dot is a station, in its line's color", transform=ax.transAxes,
            fontsize=S.SIZE["legend"] + 0.5, fontweight="semibold", va="top")
    to_data = (ax.transAxes + ax.transData.inverted()).transform
    for k, (ada, label) in enumerate([(1, "Fully accessible"), (2, "Accessible in one direction only"),
                                      (0, "Not accessible (stairs only)")]):
        yy = ly - 0.045 - k * 0.032
        cx, cy = to_data((lx + 0.008, yy))
        draw_marker(ax, cx, cy, ["456"], ada, 260, 80)
        ax.text(lx + 0.024, yy, label, transform=ax.transAxes, va="center", fontsize=S.SIZE["legend"])
    yy = ly - 0.045 - 3 * 0.032
    cx, cy = to_data((lx + 0.008, yy))
    draw_marker(ax, cx, cy, ["123", "456"], 1, 260, 80)
    ax.text(lx + 0.024, yy, "Station on several lines", transform=ax.transAxes, va="center",
            fontsize=S.SIZE["legend"])

    pax = fig.add_axes([0.575, 0.08, 0.41, 0.74])
    pictogram(pax, st, routes)
    pax.apply_aspect()
    top = pax.get_position().y1  # place the heading right above the rows
    fig.text(0.575, top + 0.035, "Accessible stations by line", fontsize=S.SIZE["panel_title"],
             fontweight="semibold", fontfamily=S.HEAD_FONT, color=INK)
    fig.text(0.575, top + 0.012, "Each dot is one station on that line, sorted by % fully accessible",
             fontsize=S.SIZE["subtitle"], color=MUTED)

    S.title(fig, title)
    fig.text(0.5, 0.925, lede, ha="center", fontsize=S.SIZE["subtitle"] + 1, color=MUTED)
    S.credit(fig, "Accessible = ADA accessible per MTA. Lines are weekday daytime service; a station on several "
             "lines counts in each line's row.  Data: MTA Subway Stations via NYC Open Data (Nov 2026)"
             f"  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_subway_accessible.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", OUT / "nyc_subway_accessible.png")

    # ---- interactive page: data in projected metres, inlined into one HTML file ----
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(25)
    web = {
        "title": title, "lede": lede, "source_url": SOURCE_URL, "day": DAY,
        "colors": S.SUBWAY, "trunk_order": TRUNK_ORDER, "route_order": ROUTE_ORDER,
        "stations": [{"id": s.station_id, "name": s.stop_name, "line": s.line, "borough": s.borough,
                      "structure": s.structure, "routes": s.routes, "trunks": s.trunks,
                      "ada": int(s.ada), "notes": s.ada_notes,
                      "x": round(s.geometry.x, 1), "y": round(s.geometry.y, 1)} for _, s in st.iterrows()],
        "routes": [{"route": r, "trunk": row.trunk, "n": int(row.n), "full": int(row.full),
                    "part": int(row.part), "pct": round(row.pct, 1)} for r, row in routes.iterrows()],
        "boroughs": json.loads(b.to_json(drop_id=True)),
    }
    # data.js lets web/index.html open straight from disk while editing (gitignored)
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "nyc_subway_accessible.html")
    print("wrote", OUT / "nyc_subway_accessible.html", f"({size / 1e6:.2f} MB)")

    g = st.to_crs("EPSG:4326").copy()
    g["routes"] = g.routes.str.join(" ")
    g["trunks"] = g.trunks.str.join(" ")
    g["status"] = g.ada.map(STATUS)
    g.to_file(OUT / "stations.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)


if __name__ == "__main__":
    main()
