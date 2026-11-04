"""Day 4 · Clusters (experiment) — crashes as a 3D landscape: more crashes nearby = taller hill.

Bins every located crash (2023-2025, pedestrian or cyclist hurt or killed) onto a 50 m grid,
sets each grid point's height to the crashes within 100 m of it, and writes a three.js page you can orbit. Writes:
  - out/nyc_crash_terrain.html

Reuses load() and clusters() from build.py. Run:  ../.venv/bin/python terrain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

import build as B

S = B.S
CELL = 50       # metres per grid point
RADIUS = 100    # tooltip: "N crashes within RADIUS m" of each cell centre, counted exactly
N_PEAKS = 8


def main():
    pts, boros, total = B.load()
    corners = B.clusters(pts, 25, 8)

    x0, y0, x1, y1 = boros.total_bounds
    x0, y0 = np.floor(x0 / CELL) * CELL - CELL, np.floor(y0 / CELL) * CELL - CELL
    nx, ny = int(np.ceil((x1 - x0) / CELL)) + 2, int(np.ceil((y1 - y0) / CELL)) + 2
    px, py = pts.geometry.x.values, pts.geometry.y.values

    # land mask at cell centres, grown by one cell so shoreline crashes keep their slope
    cx = x0 + (np.arange(nx) + 0.5) * CELL
    cy = y0 + (np.arange(ny) + 0.5) * CELL
    gx, gy = np.meshgrid(cx, cy)
    shore = boros.geometry.simplify(20).union_all()
    shapely.prepare(shore)   # without this, 220k point-in-polygon tests take minutes
    land = shapely.contains_xy(shore, gx, gy)
    print("land mask done")
    land_wide = ndimage.binary_dilation(land, iterations=1)

    # height of every grid point = exactly how many crashes happened within RADIUS m of it,
    # the same number the tooltip shows (shoreline points included so slopes run down to the water).
    # A 50 m grid keeps every point within ~35 m of the true peak, so peak counts are within a crash or two.
    count = np.full((ny, nx), -1, dtype=int)          # -1 = water (no surface drawn)
    count[land_wide] = cKDTree(np.c_[px, py]).query_ball_point(np.c_[gx[land_wide], gy[land_wide]], RADIUS,
                                                               return_length=True)

    # named intersections for the tooltip's "Near X & Y" (and the peak labels): every street pair in the
    # crash records, placed at the median location of its crashes
    named = pts[pts.place.str.contains(" & ")].assign(x=pts.geometry.x, y=pts.geometry.y)
    xs = named.groupby("pair").agg(x=("x", "median"), y=("y", "median"),
                                   name=("place", lambda s: s.value_counts().index[0]),
                                   boro=("boroname", lambda s: s.mode().iat[0])).reset_index(drop=True)
    xs_tree = cKDTree(np.c_[xs.x, xs.y])
    print(f"{len(xs):,} named intersections for tooltips")

    # peaks: highest grid point first, skipping any within 1.5 km of a peak already taken
    iy, ix = np.nonzero(count > 0)
    order = np.argsort(-count[iy, ix], kind="stable")
    chosen = []
    for i in order:
        if all(np.hypot(ix[i] - ix[c], iy[i] - iy[c]) * CELL >= 1500 for c in chosen):
            chosen.append(i)
        if len(chosen) == N_PEAKS:
            break
    peaks = []
    for i in chosen:
        k = int(xs_tree.query([cx[ix[i]], cy[iy[i]]])[1])
        peaks.append({"i": int(ix[i]), "j": int(iy[i]), "name": xs.name.iat[k], "boroname": xs.boro.iat[k],
                      "within": int(count[iy[i], ix[i]])})
    for p in peaks:
        print(f"peak {p['name']} ({p['boroname']}): {p['within']} crashes within {RADIUS} m")
    # The top three are within a crash or two of each other, and which one "wins" shifts with exactly where
    # the grid points fall, so the headline names all three. Stop if the set ever changes.
    top3 = {p["name"] for p in peaks[:3]}
    assert any("2nd Ave" in n for n in top3) and any("Broadway & Flushing" in n for n in top3) \
        and any("125th St" in n for n in top3), top3
    title = "NYC’s three tallest crash peaks: 2nd Ave at 39th St, Broadway at Flushing Ave, and 125th St in Harlem"
    print("TITLE:", title)

    web = {
        "title": title,
        "n": len(pts), "total": total, "years": B.YEARS, "day": B.DAY, "source_url": B.SOURCE_URL,
        "nx": nx, "ny": ny, "cell": CELL, "x0": float(x0), "y0": float(y0),
        "max_count": int(count.max()), "radius_m": RADIUS,
        # crashes within RADIUS m of each grid point (-1 = water), row-major from the south-west corner
        "c": count.ravel().tolist(),
        "peaks": peaks,
        "hotspots": [{"x": round(float(r.x)), "y": round(float(r.y)), "name": r["name"], "boroname": r.boroname,
                      "crashes": int(r.crashes)} for _, r in corners.iterrows()],
        "amber": S.SEQUENTIAL_AMBER,
    }
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(30)
    web["outlines"] = [[[round(x), round(y)] for x, y in ring.coords]
                       for geom in b.geometry for poly in getattr(geom, "geoms", [geom])
                       for ring in [poly.exterior] if poly.area > 2e5]
    names = sorted(xs.name.unique())
    boro_list = sorted(xs.boro.unique())
    name_i, boro_i = {n: k for k, n in enumerate(names)}, {b: k for k, b in enumerate(boro_list)}
    web["xs_names"], web["boros"] = names, boro_list
    web["xs"] = [v for r in xs.itertuples() for v in (round(r.x), round(r.y), name_i[r.name], boro_i[r.boro])]
    size = S.build_page(B.HERE / "web" / "terrain.html", web, B.OUT / "nyc_crash_terrain.html", data_src="terrain.js")
    print("wrote", B.OUT / "nyc_crash_terrain.html", f"({size / 1e6:.1f} MB), grid {nx}x{ny}")


if __name__ == "__main__":
    main()
