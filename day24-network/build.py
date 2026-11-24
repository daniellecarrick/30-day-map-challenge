"""Day 24 · Network — Citi Bike as a route map.

Every Citi Bike trip in September 2026 is counted per (start station, end station). Drawn like
an airline route map: pick a station and arcs fan out to everywhere its riders went, thicker
and darker for busier routes. Round trips (same dock) are left out of the routes.

Writes:
  - out/citibike_routes.png   (static: every route out of the Grand Central dock)
  - out/citibike_routes.html  (interactive: hover any station for its routes, toggle from/to)

Data: Citi Bike System Data, https://citibikenyc.com/system-data (Sept 2026 trip CSVs, the same
      download as Day 2; data_prep.py reduces them to data/od_202609_all.csv.gz).
      Borough Boundaries (NYC Open Data), copied from day02-lines/data.
Run:  ../.venv/bin/python data_prep.py   (once, ~2 min; needs day02-lines/data/raw)
      ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 24: Network"
SOURCE = "Citi Bike System Data, Sept 2026"
SOURCE_URL = "https://citibikenyc.com/system-data"
CRS_M = "EPSG:32618"
DAYS = 30  # September
MIN_TRIPS = 3  # routes with fewer rides in the month are not drawn (they'd be hairlines anyway)
FEATURED = "6432.09"  # Park Ave & E 42 St, outside Grand Central
FEATURED_LABEL = "Grand Central"
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
TITLE = "Riders leaving the Citi Bike dock at Grand Central went to 840 different stations in September"


BEARING = 29   # degrees: turn the map so Manhattan's avenues point up (same as the page)
TILT = 58      # degrees the ground is tipped away from the viewer (page: pitch 52)
ARC_K = 0.22   # arc peak as a share of the route's length


class Tilt:
    """A simple parallel 3D view: rotate the ground by BEARING about `origin`, tip it back by
    TILT (squash north-south), and lift arc points straight up the page by their height."""
    def __init__(self, origin):
        self.ox, self.oy = origin
        a = np.radians(BEARING)
        self.c, self.s, self.k = np.cos(a), np.sin(a), np.cos(np.radians(TILT))
        self.up = np.sin(np.radians(TILT))

    def __call__(self, x, y, z=0.0):
        x, y = np.asarray(x) - self.ox, np.asarray(y) - self.oy
        xr, yr = x * self.c - y * self.s, x * self.s + y * self.c
        return xr, yr * self.k + np.asarray(z) * self.up

    def geom(self, g):
        from shapely import affinity
        g = affinity.rotate(g, BEARING, origin=(self.ox, self.oy))
        g = affinity.translate(g, -self.ox, -self.oy)
        return affinity.scale(g, 1, self.k, origin=(0, 0))

    def arc(self, x0, y0, x1, y1, n=40):
        t = np.linspace(0, 1, n)
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        z = ARC_K * np.hypot(x1 - x0, y1 - y0) * 4 * t * (1 - t)
        return np.column_stack(self(x, y, z))


def load():
    od = pd.read_csv(DATA / "od_202609_all.csv.gz", dtype={"a": str, "b": str})
    st = pd.read_csv(DATA / "stations_202609.csv", dtype={"id": str})
    g = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lng, st.lat), crs="EPSG:4326").to_crs(CRS_M)
    st["x"], st["y"] = g.geometry.x.round(1), g.geometry.y.round(1)
    rt = od[od.a == od.b].set_index("a").trips
    od = od[od.a != od.b]
    st = st.set_index("id")
    st["out"] = od.groupby("a").trips.sum().reindex(st.index).fillna(0).astype(int)
    st["inn"] = od.groupby("b").trips.sum().reindex(st.index).fillna(0).astype(int)
    st["nout"] = od.groupby("a").size().reindex(st.index).fillna(0).astype(int)
    st["nin"] = od.groupby("b").size().reindex(st.index).fillna(0).astype(int)
    st["round"] = rt.reindex(st.index).fillna(0).astype(int)
    st = st[(st.out + st.inn) >= 30].copy()  # drop test docks / stations open a few days
    od = od[od.a.isin(st.index) & od.b.isin(st.index)]
    boros = gpd.read_file(HERE.parent / "day02-lines" / "data" / "boroughs.geojson").to_crs(CRS_M)
    # each station's borough (nearest, for docks on piers just outside the outline)
    pts = gpd.GeoDataFrame(st[[]], geometry=gpd.points_from_xy(st.x, st.y), crs=CRS_M)
    st["boro"] = gpd.sjoin_nearest(pts, boros[["boroname", "geometry"]], how="left").groupby(level=0).boroname.first()
    return od, st, boros


def frame(st, pad=600):
    x0, y0, x1, y1 = st.x.min() - pad, st.y.min() - pad, st.x.max() + pad, st.y.max() + pad
    return box(x0, y0, x1, y1)


def png(od, st, boros):
    S.apply()
    f = od[(od.a == FEATURED)].sort_values("trips")
    total, ndest = f.trips.sum(), len(f)
    fx, fy = st.loc[FEATURED, ["x", "y"]]
    f = f[f.trips >= MIN_TRIPS].copy()
    f["x"], f["y"] = f.b.map(st.x).values, f.b.map(st.y).values
    per_day = f.trips / DAYS
    vmax = per_day.max()
    T = Tilt((fx, fy))

    segs = [T.arc(fx, fy, x, y) for x, y in zip(f.x, f.y)]
    allp = np.vstack(segs)
    pad = 700
    x0, x1 = allp[:, 0].min() - pad, allp[:, 0].max() + pad
    y0, y1 = allp[:, 1].min() - pad, allp[:, 1].max() + pad
    W, H = S.MAP["png_size_in"]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0.04, 1, 0.86])
    aspect = W / (H * 0.86)   # widen the frame to the axes' shape: the sides hold the labels
    cx, half = (x0 + x1) / 2, (y1 - y0) * aspect / 2
    x0, x1 = cx - half, cx + half
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()

    land = boros.copy()
    land["geometry"] = land.geometry.apply(T.geom)
    land = land.clip(box(x0 - 3000, y0 - 3000, x1 + 3000, y1 + 3000))
    land.plot(ax=ax, color=S.C["ctl"], edgecolor=S.INK, linewidth=0.4, zorder=1)
    sx, sy = T(st.x.values, st.y.values)
    ax.scatter(sx, sy, s=1.0, color=S.C["closed"], lw=0, zorder=2)

    t = (per_day / vmax).clip(0, 1) ** 0.5
    # arc color = destination borough (categorical, fixed order), as on the page
    cols = [S.CATEGORICAL[BOROUGHS.index(st.boro[b])] for b in f.b]
    widths = 0.35 + 3.2 * t ** 2
    for k, pad, a in [(4.5, 2.5, 0.05), (2.2, 1.0, 0.11)]:   # glow: wider, fainter copies underneath (as on the page)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths * k + pad, alpha=a, zorder=3, capstyle="round"))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths, alpha=0.85, zorder=3, capstyle="round"))
    dx, dy = T(f.x.values, f.y.values)
    ax.scatter(dx, dy, s=3 + 26 * per_day / vmax, color=cols, lw=0.3, edgecolor="white", zorder=4)
    gx, gy = T(fx, fy)
    ax.scatter([gx], [gy], s=60, color=S.INK, edgecolor="white", lw=1.2, zorder=6)

    # callouts: the dock, its top destination and the farthest one
    top = f.iloc[-1]
    tx, ty = T(top.x, top.y)
    S.callout(ax, (tx, ty), (0.10, 0.62), f"Most popular: {st.name[top.b]}",
              f"{top.trips / DAYS:.0f} rides a day, {top.trips / total:.0%} of rides from Grand Central\n"
              f"median ride {top.med_min:.0f} min", halo=True)
    far = f.assign(d=np.hypot(f.x - fx, f.y - fy)).sort_values("d").iloc[-1]
    rx, ry = T(far.x, far.y)
    S.callout(ax, (rx, ry), (0.66, 0.10), f"Farthest regular route: {st.name[far.b]}",
              f"{far.d / 1609:.1f} miles as the crow flies, {far.trips} rides in September", halo=True)
    S.callout(ax, (gx, gy), (0.66, 0.84), f"Park Ave & E 42 St ({FEATURED_LABEL})",
              f"{total / DAYS:.0f} rides a day leave this dock", halo=True)
    # width key
    kx, ky = 0.10, 0.40
    ax.text(kx, ky + 0.06, "Rides a day on the route", transform=ax.transAxes, fontsize=S.SIZE["legend"],
            fontweight="semibold", color=S.INK)
    for i, v in enumerate([1, 4, 10]):
        yy, tv = ky - i * 0.03, (v / vmax) ** 0.5
        ax.plot([kx, kx + 0.05], [yy, yy], transform=ax.transAxes, color=S.MUTED,
                lw=0.35 + 3.2 * tv ** 2, solid_capstyle="round")
        ax.text(kx + 0.06, yy, f"{v}", transform=ax.transAxes, va="center", fontsize=S.SIZE["legend"], color=S.INK)
    ax.text(kx, ky - 0.12, "Arc color: where the ride ended", transform=ax.transAxes, fontsize=S.SIZE["legend"],
            fontweight="semibold", color=S.INK)
    shares = f.assign(boro=f.b.map(st.boro)).groupby("boro").trips.sum() / total
    for i, b in enumerate([b for b in BOROUGHS if b in shares.index]):
        yy = ky - 0.16 - i * 0.03
        ax.plot([kx, kx + 0.05], [yy, yy], transform=ax.transAxes, color=S.CATEGORICAL[BOROUGHS.index(b)], lw=2.6,
                solid_capstyle="round")
        ax.text(kx + 0.06, yy, b, transform=ax.transAxes, va="center", fontsize=S.SIZE["legend"], color=S.INK)
    ax.text(kx, ky - 0.30, f"Thinnest lines: {MIN_TRIPS} rides in the month.\nArc height grows with distance.",
            transform=ax.transAxes, fontsize=S.SIZE["annotation"], color=S.MUTED, va="top")

    S.title(fig, TITLE)
    fig.text(0.5, 0.925, f"Every route out of Park Ave & E 42 St in September 2026. The most popular one got just "
             f"{top.trips / total:.0%} of its riders.", ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, f"Trips per start and end station, Sept 2026, round trips excluded  ·  Data: {SOURCE} "
             f"(citibikenyc.com/system-data)  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "citibike_routes.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    return total, ndest, top


def page(od, st, boros):
    st = st.sort_values("out", ascending=False)
    idx = {sid: i for i, sid in enumerate(st.index)}
    keep = od[od.trips >= MIN_TRIPS].copy()
    keep["ai"], keep["bi"] = keep.a.map(idx), keep.b.map(idx)
    keep = keep.sort_values(["ai", "trips"], ascending=[True, False])
    routes = [[] for _ in range(len(st))]
    for ai, bi, t, m in zip(keep.ai, keep.bi, keep.trips, keep.med_min):
        routes[ai].extend([int(bi), int(t), int(round(m))])
    # each station's rides by the borough they ended in. All rides, not just the routes drawn.
    boro_of = st.boro
    names = [b for b in BOROUGHS if b in set(boro_of)]
    to_boro = (od.assign(boro=od.b.map(boro_of)).pivot_table(index="a", columns="boro", values="trips", aggfunc="sum")
                 .reindex(index=st.index, columns=names).fillna(0).astype(int))
    fr = frame(st, 1500)
    land = boros.clip(fr)[["boroname", "geometry"]].explode(index_parts=False)
    land = land[land.area > 2e4]
    land["geometry"] = land.geometry.simplify(15)
    data = {
        "stations": {"name": st.name.tolist(), "lng": st.lng.round(5).tolist(), "lat": st.lat.round(5).tolist(),
                     "out": st.out.tolist(), "nout": st.nout.tolist(), "to_boro": to_boro.values.tolist(),
                     "boro": [names.index(b) for b in boro_of.reindex(st.index)]},
        "boro_names": names,
        "title": TITLE,
        "routes": routes, "featured": idx[FEATURED], "days": DAYS, "min_trips": MIN_TRIPS,
        "boroughs": json.loads(land.to_crs("EPSG:4326").to_json(drop_id=True)),
        # opening view: below ~125th St to Prospect Park, the Hudson to western Queens
        "core": [-74.03, 40.655, -73.90, 40.815],
        "day": DAY, "source": SOURCE, "source_url": SOURCE_URL,
        "method": "Rides per start and end dock, averaged over the 30 days of September 2026; round trips left out",
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "citibike_routes.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    return size


def main():
    od, st, boros = load()
    total, ndest, top = png(od, st, boros)
    print(f"Grand Central: {total} rides out ({total / DAYS:.0f}/day) to {ndest} stations; top {st.name[top.b]} "
          f"{top.trips} ({top.trips / total:.1%})")
    size = page(od, st, boros)
    print(f"wrote out/citibike_routes.png and out/citibike_routes.html ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
