"""Day 3 · Polygons — Nearest public restroom in NYC.

Builds Voronoi cells around operational public restrooms (all vs. accessible),
clipped to borough land, and colors each cell by the LONGEST walk you could face
inside it (from its farthest corner to its restroom). Writes:
  - out/nyc_restrooms_voronoi.png    (static side-by-side map)
  - out/nyc_restrooms_voronoi.html   (interactive, self-contained)
  - out/cells_all.geojson, out/cells_accessible.geojson, out/points.geojson

Data: NYC Open Data "Public Restrooms" (i7jb-7jku), "Borough Boundaries" (gthc-hcne).
Run:  ../.venv/bin/python build.py
"""
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

CRS_M = "EPSG:32618"  # UTM 18N, metres
ACCESSIBLE = {"Fully Accessible", "Partially Accessible", "Limited Accessibility"}

# Walk time = straight-line distance x street detour factor, at an average walking pace.
WALK_M_PER_MIN = 80     # ≈ 4.8 km/h
DETOUR = 1.3            # streets are ~30% longer than "as the crow flies"

# The series' diverging scale (shared/tokens.json), centered on the citywide average:
# blue = shorter than average, neutral = average, amber -> brown = longer.
CENTER = 11             # minutes; ≈ the citywide average walk to the nearest restroom
STOPS = S.diverging_stops(CENTER)      # 0 -> 22 min
SCALE_MAX = STOPS[-1][0]
TICKS = [0, CENTER, SCALE_MAX]
walk_color = S.make_scale(STOPS)
CLOSED_GREY, INK, MUTED, BG = S.CLOSED, S.INK, S.MUTED, S.BG


def load():
    rec = pd.DataFrame(json.loads((DATA / "restrooms.json").read_text()))
    rec["latitude"] = rec["latitude"].astype(float)
    rec["longitude"] = rec["longitude"].astype(float)
    pts = gpd.GeoDataFrame(
        rec[["facility_name", "operator", "status", "accessibility", "restroom_type",
             "hours_of_operation", "location_type"]],
        geometry=gpd.points_from_xy(rec.longitude, rec.latitude), crs="EPSG:4326",
    ).to_crs(CRS_M)
    pts["operational"] = pts["status"].eq("Operational")
    pts["accessible"] = pts["accessibility"].isin(ACCESSIBLE)

    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    # tag each restroom with its borough (nearest, for points sitting on the shoreline)
    pts = gpd.sjoin_nearest(pts, boros, how="left").drop(columns="index_right")
    pts = pts[~pts.index.duplicated()]
    return pts, boros


def voronoi_cells(points: gpd.GeoDataFrame, boros: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Voronoi per borough, clipped to that borough's land, so cells never jump rivers
    between boroughs (a Manhattan toilet can't 'own' part of Queens)."""
    out = []
    for _, b in boros.iterrows():
        p = points[points.boroname == b.boroname]
        if p.empty:
            continue
        p = p[~p.geometry.apply(lambda g: g.wkb).duplicated()]  # same spot -> first one owns it
        if len(p) == 1:
            regions = [b.geometry]
        else:
            vor = shapely.voronoi_polygons(shapely.MultiPoint(list(p.geometry)),
                                           extend_to=b.geometry.envelope.buffer(5000))
            regions = list(vor.geoms)
        cells = gpd.GeoDataFrame(geometry=regions, crs=CRS_M)
        cells["geometry"] = cells.intersection(b.geometry)
        cells = cells[~cells.is_empty]
        cells = gpd.sjoin(cells, p[["facility_name", "operator", "accessibility", "geometry"]],
                          predicate="contains", how="inner")
        site = p.geometry.loc[cells.index_right].values
        cells = cells.drop(columns="index_right")
        # Longest walk inside the cell: distance is convex, so its maximum over a polygon
        # is always at a vertex -> check every vertex of the cell.
        far = []
        for geom, s in zip(cells.geometry, site):
            # islands: measure only the piece of the cell that holds its restroom
            if geom.geom_type == "MultiPolygon":
                geom = min(geom.geoms, key=lambda part: part.distance(s))
            xy = shapely.get_coordinates(geom.boundary)
            far.append(np.hypot(xy[:, 0] - s.x, xy[:, 1] - s.y).max())
        cells["max_walk_min"] = np.array(far) * DETOUR / WALK_M_PER_MIN
        cells["boroname"] = b.boroname
        out.append(cells)
    cells = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=CRS_M)
    cells["color"] = cells.max_walk_min.apply(walk_color)
    return cells


def average_walk(points: gpd.GeoDataFrame, boros: gpd.GeoDataFrame, n: int = 40000, seed: int = 1) -> float:
    """Mean walk (min) to the nearest restroom from evenly spread random spots on land,
    using the same per-borough rule as the polygons."""
    rng = np.random.default_rng(seed)
    total = boros.area.sum()
    mins = []
    for _, b in boros.iterrows():
        k = int(n * b.geometry.area / total)
        x0, y0, x1, y1 = b.geometry.bounds
        xy = np.empty((0, 2))
        while len(xy) < k:
            cand = rng.uniform([x0, y0], [x1, y1], (k * 3, 2))
            xy = np.vstack([xy, cand[shapely.contains_xy(b.geometry, cand[:, 0], cand[:, 1])]])
        xy = xy[:k]
        p = points[points.boroname == b.boroname]
        P = np.c_[p.geometry.x, p.geometry.y]
        for chunk in np.array_split(xy, max(1, k // 2000)):
            d = np.sqrt(((chunk[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
            mins.append(d * DETOUR / WALK_M_PER_MIN)
    return float(np.concatenate(mins).mean())


def draw_panel(ax, cells, open_pts, closed_pts, boros, title, subtitle):
    cells.plot(ax=ax, color=cells.color, edgecolor="white", linewidth=0.35)
    boros.boundary.plot(ax=ax, color=INK, linewidth=0.4)
    open_pts.plot(ax=ax, color=INK, markersize=1.6, linewidth=0)
    closed_pts.plot(ax=ax, color=CLOSED_GREY, markersize=14, edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_axis_off()
    ax.set_aspect("equal")
    S.panel_heading(ax, title, subtitle)

    # Call out the best- and worst-covered polygons
    def anchor(geom):  # a point inside the polygon's largest piece
        part = max(geom.geoms, key=lambda g: g.area) if geom.geom_type == "MultiPolygon" else geom
        p = part.representative_point()
        return p.x, p.y
    best = cells.loc[cells.max_walk_min.idxmin()]
    worst = cells.loc[cells.max_walk_min.idxmax()]
    # put the label level with its polygon so the leader is a short horizontal line
    best_xy = anchor(best.geometry)
    best_y = (ax.transData + ax.transAxes.inverted()).transform(best_xy)[1]
    S.callout(ax, best_xy, (0.03, best_y),
              f"Shortest: every spot within {best.max_walk_min:.0f} min",
              f"{short_name(best.facility_name)}, {best.boroname}", outline=best.geometry)
    S.callout(ax, anchor(worst.geometry), (0.98, 0.10),
              f"Longest: up to {worst.max_walk_min:.0f} min",
              f"{short_name(worst.facility_name)}, {worst.boroname}", ha="right", outline=worst.geometry)


def short_name(name: str) -> str:
    """'East River Waterfront Esplanade Zone 1' -> 'East River Waterfront Esplanade'."""
    return re.sub(r"\s+Zone \d+$", "", name)


def main():
    S.apply()
    pts, boros = load()
    closed = pts[~pts.operational]
    open_all = pts[pts.operational]
    open_acc = open_all[open_all.accessible]

    cells_all = voronoi_cells(open_all, boros)
    cells_acc = voronoi_cells(open_acc, boros)

    avg = {"all": round(average_walk(open_all, boros), 1),
           "accessible": round(average_walk(open_acc, boros), 1)}
    print("average walk from a random spot:", avg)
    share = {}  # % of polygons whose longest walk is under the CENTER (blue side)
    for name, c in [("all", cells_all), ("accessible", cells_acc)]:
        share[name] = round(100 * np.mean(c.max_walk_min < CENTER))
        print(f"{name}: {len(c)} cells · median longest walk {c.max_walk_min.median():.1f} min · "
              f"≤5 min {np.mean(c.max_walk_min <= 5):.0%} of cells · <{CENTER} min {share[name]}% of cells · "
              f"worst {c.max_walk_min.max():.0f} min")

    # ---- static PNG ----
    fig, axes = plt.subplots(1, 2, figsize=S.MAP["png_size_in"])
    xmin, ymin, xmax, ymax = boros.total_bounds
    for ax in axes:
        ax.set_xlim(xmin - 1500, xmax + 1500)
        ax.set_ylim(ymin - 1500, ymax + 1500)
    draw_panel(axes[0], cells_all, open_all, closed, boros, "Every public restroom",
               f"{len(open_all):,} operational · {share['all']}% of polygons under {CENTER} min")
    draw_panel(axes[1], cells_acc, open_acc, closed, boros, "Accessible restrooms only",
               f"{len(open_acc):,} accessible · {share['accessible']}% of polygons under {CENTER} min")

    # continuous color bar
    cax = fig.add_axes([0.36, 0.075, 0.28, 0.018])
    grid = np.linspace(0, SCALE_MAX, 301)
    sm = ScalarMappable(norm=Normalize(0, SCALE_MAX), cmap=ListedColormap([walk_color(m) for m in grid]))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=TICKS)
    cb.ax.set_xticklabels(["0", f"{CENTER:g} (city avg)", f"{SCALE_MAX:g}+ min"], fontsize=S.SIZE["legend"], color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=3)
    cb.ax.set_title("Longest walk to the restroom from anywhere in its polygon", fontsize=S.SIZE["subtitle"], color=INK, pad=5)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, markersize=2.5, label="Restroom"),
                        Line2D([], [], marker="o", ls="", color=CLOSED_GREY, markeredgecolor="white",
                               markersize=6, label="Closed / not operational")],
               loc="lower left", bbox_to_anchor=(0.66, 0.062), fontsize=S.SIZE["legend"], labelcolor=INK)
    S.title(fig, "Where's the nearest bathroom in NYC?")
    S.credit(fig, "Polygons: the area closer to each restroom than any other in its borough.  "
             f"Walk time = straight-line distance × {DETOUR} at {WALK_M_PER_MIN} m/min.  "
             "Data: NYC Open Data, Public Restrooms (Nov 2026)  ·  #30DayMapChallenge Day 3: Polygons")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12, wspace=0.02)
    fig.savefig(OUT / "nyc_restrooms_voronoi.png", dpi=S.MAP["png_dpi"])
    print("wrote", OUT / "nyc_restrooms_voronoi.png")

    # ---- interactive page: data in projected metres, inlined into one HTML file ----
    cols = ["facility_name", "operator", "boroname", "max_walk_min", "color"]
    web = {"share_pct": share, "avg": avg, "center": CENTER, "stops": STOPS, "ticks": TICKS,
           "counts": {"all": len(open_all), "accessible": len(open_acc)}}
    for key, gdf, cc in [
        ("all", cells_all, cols), ("accessible", cells_acc, cols), ("boroughs", boros, ["boroname"]),
        ("points", pts.assign(accessibility=pts.accessibility.fillna("Unknown")),
         ["facility_name", "operator", "status", "accessibility", "operational", "accessible"]),
    ]:
        g = gdf[cc + ["geometry"]].copy()
        g["geometry"] = g.geometry.simplify(3)
        if "max_walk_min" in g:
            g["max_walk_min"] = g.max_walk_min.round(1)
        web[key] = json.loads(g.to_json(drop_id=True))
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "nyc_restrooms_voronoi.html")
    print("wrote", OUT / "nyc_restrooms_voronoi.html", f"({size / 1e6:.1f} MB)")

    # ---- GeoJSON (lon/lat) for use in other tools ----
    for f in OUT.glob("walk_bands_*.geojson"):
        f.unlink()
    def write(gdf, fname, cc):
        g = gdf[cc + ["geometry"]].to_crs("EPSG:4326").copy()
        g["geometry"] = g.geometry.simplify(0.00002)
        g.to_file(OUT / fname, driver="GeoJSON", COORDINATE_PRECISION=5)
    write(cells_all, "cells_all.geojson", cols)
    write(cells_acc, "cells_accessible.geojson", cols)
    write(boros, "boroughs.geojson", ["boroname"])
    write(pts.assign(hours_of_operation=pts.hours_of_operation.fillna("")), "points.geojson",
          ["facility_name", "operator", "status", "accessibility", "restroom_type",
           "hours_of_operation", "operational", "accessible"])


if __name__ == "__main__":
    main()
