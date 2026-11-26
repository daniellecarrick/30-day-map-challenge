"""Day 26 · Water: one day of New York City's water, poured into Midtown's towers in turn.

Each landmark's volume comes from its 3D shape in OpenStreetMap (building:part pieces with heights),
measured by slicing it every 5 ft and taking the union of the pieces at each height, so overlapping
parts aren't counted twice. Where a model leaves the ground floors out (the Empire State Building's
base), the building outline fills the gap up to its first big setback.

Writes out/one_day_of_water.png and out/one_day_of_water.html
Run:  ../.venv/bin/python data_prep.py  (once)  then  ../.venv/bin/python build_midtown.py
"""
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "out"
OUT.mkdir(exist_ok=True)
DAY = "Day 26: Water"
FT3_PER_GAL = 0.133681
DZ = 5  # ft, slice height for volumes

# Filled in this order, starting with the most famous; OSM names
LANDMARKS = ["Empire State Building", "Chrysler Building", "One Vanderbilt", "30 Rockefeller Plaza",
             "Bank of America Tower", "MetLife Building", "432 Park Avenue"]
SHORT = {"30 Rockefeller Plaza": "30 Rock", "Bank of America Tower": "Bank of America Tower"}
SOURCE = "Data: buildings © OpenStreetMap contributors; water use from NYC Open Data"
SOURCE_HTML = ('Data: buildings © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors; '
               'water use from <a href="https://data.cityofnewyork.us/d/ia2d-e54m">NYC Open Data</a>')


def num(v):
    if v is None:
        return None
    m = re.match(r"\s*([\d.]+)\s*(m|ft|')?", str(v))
    if not m:
        return None
    x = float(m.group(1))
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def load_buildings():
    osm = json.load(open(DATA / "midtown_osm.json"))
    rows = []
    for e in osm["elements"]:
        t = e.get("tags", {})
        if e["type"] == "way" and "geometry" in e:
            c = [(p["lon"], p["lat"]) for p in e["geometry"]]
            if len(c) < 4 or c[0] != c[-1]:
                continue
            g = Polygon(c)
        elif e["type"] == "relation":
            ring = lambda role: [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                                 for m in e.get("members", []) if m.get("role") == role and "geometry" in m]
            outer, inner = ring("outer"), ring("inner")
            if not outer:
                continue
            g = unary_union(list(polygonize(unary_union(outer))))
            if inner:
                g = g.difference(unary_union(list(polygonize(unary_union(inner)))))
        else:
            continue
        if g.is_empty or not g.is_valid:
            g = g.buffer(0)
        h = num(t.get("height"))
        if h is None and num(t.get("building:levels")):
            h = num(t["building:levels"]) * 3.5
        mh = num(t.get("min_height")) or (num(t.get("building:min_level")) or 0) * 3.5
        rows.append({"id": f"{e['type'][0]}{e['id']}", "part": "building:part" in t and "building" not in t,
                     "name": t.get("name"), "h": h if h else 12.0, "mh": mh, "geometry": g})
    df = gpd.GeoDataFrame(rows, crs=4326).to_crs(2263)   # NY State Plane, feet
    df["h"] /= 0.3048
    df["mh"] /= 0.3048
    df = df[df.geometry.area > 2].explode(index_parts=False)
    df = df[df.geom_type == "Polygon"]
    return df[~df.part].reset_index(drop=True), df[df.part].reset_index(drop=True)


def landmark_parts(outline, parts):
    """3D pieces of one landmark, with the ground-floor gap filled from its outline."""
    inside = parts[parts.representative_point().within(outline.buffer(3))].copy()
    big = inside[inside.area > 0.2 * outline.area]
    base_top = big.mh.min() if len(big) else 0
    if base_top > 0:
        inside = pd.concat([inside, gpd.GeoDataFrame([{"geometry": outline, "mh": 0.0, "h": base_top}], crs=parts.crs)])
    return inside


def volume_table(p):
    """Cumulative volume (ft³) below each 5-ft level, from 0 to the top."""
    top = p.h.max()
    levels = np.arange(0, top + DZ, DZ)
    vols = [0.0]
    for z in levels[:-1]:
        s = p[(p.mh <= z + DZ / 2) & (p.h > z + DZ / 2)]
        vols.append(vols[-1] + (unary_union(list(s.geometry)).area * DZ if len(s) else 0))
    return levels, np.array(vols)


def ring(g, ox, oy):
    return [v for x, y in list(g.exterior.coords)[:-1] for v in (round(x - ox), round(y - oy))]


# ---- PNG: the same view as the page (from the south, looking uptown) at midnight on a 2024 day ----
AZ, EL = np.radians(185), np.radians(32)      # camera bearing from the target, and height angle
RIGHT = np.array([-np.cos(AZ), np.sin(AZ)])   # screen-right on the ground
FWD = np.array([-np.sin(AZ), -np.cos(AZ)])    # away from the camera on the ground


def proj(xy, z=0.0):
    xy = np.atleast_2d(xy)
    return np.c_[xy @ RIGHT, xy @ FWD * np.sin(EL) + np.asarray(z) * np.cos(EL)]


def fmt_hour(h):
    m = int(round(h * 60)) % 1440
    hh, mm = divmod(m, 60)
    return f"{hh % 12 or 12}:{mm:02d} {'am' if hh < 12 else 'pm'}"


PNG_DX = {"Bank of America Tower": -1500}   # ft; moves a label off a neighbour's leader line


def png(lms, ctx, day_ft3, title, lede):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from shapely.geometry.polygon import orient
    S.apply()
    blue, dark, glass = S.SEQUENTIAL_BLUE[3], S.SEQUENTIAL_BLUE[5], S.C["ctl"]
    fig = plt.figure(figsize=(12, 9.4))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.78])
    ax.set_aspect("equal"); ax.axis("off")

    ax_ = np.array([l["outline"].centroid.coords[0] for l in lms])
    c = (ax_.min(0) + ax_.max(0)) / 2
    rad = 4600
    ring = c + rad * np.c_[np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200))]
    ax.fill(*proj(ring).T, color=glass, lw=0, zorder=0)
    near = ctx[(ctx.mh == 0) & (ctx.geometry.centroid.distance(gpd.points_from_xy([c[0]], [c[1]])[0]) < rad)]
    ax.add_collection(PolyCollection([proj(np.array(g.exterior.coords)) for g in near.geometry],
                                     facecolor=S.C["other"], edgecolor="none", zorder=1))

    # water poured by midnight, tower by tower
    left, faces = day_ft3, []
    for j, l in enumerate(lms):
        v = min(l["vol"][-1], max(0.0, left)); left -= l["vol"][-1]
        L = np.interp(v, l["vol"], l["levels"])
        for g, mh, h in zip(l["parts"].geometry, l["parts"].mh, l["parts"].h):
            if g.geom_type != "Polygon":
                continue
            pts = np.array(orient(g, 1).exterior.coords)
            for z0, z1, col in ((mh, min(h, L), blue), (max(mh, L), h, glass)):
                if z1 - z0 < 0.5:
                    continue
                for a, b in zip(pts[:-1], pts[1:]):      # walls facing the camera
                    n = np.array([b[1] - a[1], a[0] - b[0]])
                    if n @ FWD >= 0 or not n.any():
                        continue
                    shade = 0.82 + 0.18 * abs(n @ RIGHT) / np.hypot(*n)
                    quad = np.r_[proj(a, z0), proj(b, z0), proj(b, z1), proj(a, z1)]
                    depth = ((a + b) / 2) @ FWD * np.cos(EL) - (z0 + z1) / 2 * np.sin(EL)
                    faces.append((depth, quad, col, shade))
                faces.append(((pts[:-1].mean(0)) @ FWD * np.cos(EL) - z1 * np.sin(EL) - 1, proj(pts, z1), col, 1.0))
        l["L"], l["full_at"] = L, None
    faces.sort(key=lambda f: -f[0])
    from matplotlib.colors import to_rgb
    ax.add_collection(PolyCollection([f[1] for f in faces],
                                     facecolors=[tuple(np.array(to_rgb(f[2])) * f[3]) for f in faces],
                                     edgecolors=[dark if f[2] == blue else S.MUTED for f in faces],
                                     linewidths=0.25, zorder=2))
    ax.autoscale_view()

    # labels in rows above the towers, with leader lines
    per_hour, cum = day_ft3 / 24, 0
    items = []
    for l in lms:
        cum += l["vol"][-1]
        top = proj(np.array(l["outline"].centroid.coords[0]), l["levels"][-1])[0]
        if cum <= day_ft3:
            status, col = f"full by {fmt_hour(cum / per_hour)}", dark
        elif cum - l["vol"][-1] < day_ft3:
            status, col = f"{(day_ft3 - cum + l['vol'][-1]) / l['vol'][-1]:.0%} full at midnight", S.INK
        else:
            status, col = "not reached", S.MUTED
        items.append((top, l["name"], status, col))
    y0 = max(t[0][1] for t in items) + 500
    order = sorted(range(len(items)), key=lambda i: items[i][0][0])
    rows, row_end = {}, []
    for i in order:
        x = items[i][0][0]
        r = next((k for k, e in enumerate(row_end) if e < x - 1500), len(row_end))
        if r == len(row_end):
            row_end.append(x)
        row_end[r] = x; rows[i] = r
    for i, (top, name, status, col) in enumerate(items):
        ly = y0 + (max(rows.values()) - rows[i]) * 380
        lx = top[0] + PNG_DX.get(name, 0)
        ax.plot([top[0], lx], [top[1] + 40, ly - 30], color=S.MUTED, lw=0.6, zorder=3)
        ax.text(lx, ly + 150, name, ha="center", va="bottom", fontsize=S.SIZE["annotation"] + 0.5,
                fontweight="semibold", color=S.INK, zorder=4)
        ax.text(lx, ly + 140, status, ha="center", va="top", fontsize=S.SIZE["annotation"], color=col, zorder=4)
    ax.relim(); ax.autoscale_view()

    S.title(fig, title)
    fig.text(0.5, 0.915, lede, ha="center", va="top", fontsize=11, color=S.MUTED, wrap=True)
    S.credit(fig, "Volume of each tower’s 3D shape, in 5-ft slices; 2024’s average day  ·  "
                  f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "one_day_of_water.png", dpi=150)
    plt.close(fig)
    print("wrote out/one_day_of_water.png")


def main():
    b, parts = load_buildings()
    use = pd.DataFrame(json.load(open(DATA / "consumption.json"))).astype(float).set_index("year") \
        .nyc_consumption_million_gallons_per_day
    days = {"2024": use[2024], "1991": use[1991]}
    day_ft3 = {k: v * 1e6 * FT3_PER_GAL for k, v in days.items()}

    lms, used = [], set()
    for name in LANDMARKS:
        r = b[b.name == name]
        if r.empty:
            r = b[b.name.fillna("").str.contains(name, case=False)]
        r = r.iloc[0]
        p = landmark_parts(r.geometry, parts)
        used |= set(p.id.dropna())
        used.add(r.id)
        lv, vol = volume_table(p)
        lms.append({"name": SHORT.get(name, name), "outline": r.geometry, "parts": p, "levels": lv, "vol": vol})
        print(f"  {name:24s} {vol[-1] / 1e6:5.1f}M ft³  top {lv[-1]:.0f} ft  {len(p)} pieces")

    esb = lms[0]["vol"][-1]
    hours = 24 * esb / day_ft3["2024"]
    print(f"one day 2024 = {day_ft3['2024'] / 1e6:.1f}M ft³ = {day_ft3['2024'] / esb:.2f} Empire States "
          f"(one every {hours:.1f} h); 1991 = {day_ft3['1991'] / 1e6:.1f}M ft³")
    assert 5.5 <= hours < 6.5, "headline says about every six hours"
    # Central Park (843 acres, per the Central Park Conservancy) flooded with one day of water
    park_ft = day_ft3["2024"] / (843 * 43560)
    print(f"one day floods Central Park {park_ft:.1f} ft deep")
    assert 3.5 < park_ft < 4, "subhead says more than 3½ ft"
    cum = np.cumsum([l["vol"][-1] for l in lms])
    for k, v in day_ft3.items():
        assert v < cum[-1], f"{k}: a day's water outruns the landmark list"

    # context: every other building (parts where a building has them, else its outline)
    has_parts = set(b.id[b.geometry.apply(lambda g: parts.sindex.query(g, "contains").size > 0)])
    ctx = pd.concat([b[~b.id.isin(has_parts | used)], parts[~parts.id.isin(used)]])
    ctx = ctx[ctx.h - ctx.mh > 1]
    ctx["geometry"] = ctx.geometry.simplify(2)
    ox, oy = np.round(unary_union([l["outline"] for l in lms]).centroid.coords[0])

    data = {
        "title": "New York uses an Empire State Building’s worth of water about every six hours",
        "park_ft": round(park_ft, 1),
        "day": DAY, "source": SOURCE_HTML,
        "days": {k: {"mgd": round(days[k]), "ft3": round(day_ft3[k])} for k in days},
        "ctx": [[ring(g, ox, oy), round(mh), round(h)] for g, mh, h in zip(ctx.geometry, ctx.mh, ctx.h)
                if g.geom_type == "Polygon"],
        "landmarks": [{
            "name": l["name"], "vol": round(l["vol"][-1]),
            "table": [round(v / 1e4) for v in l["vol"]],    # cumulative, in 10,000 ft³, every 5 ft
            "parts": [[ring(g, ox, oy), round(mh), round(h)] for g, mh, h in
                      zip(l["parts"].geometry.simplify(1), l["parts"].mh, l["parts"].h) if g.geom_type == "Polygon"],
            "top": round(l["levels"][-1]),
            "at": [round(v) for v in np.array(l["outline"].centroid.coords[0]) - (ox, oy)],
        } for l in lms],
        "dz": DZ,
    }
    more = round((days["1991"] / days["2024"] - 1) * 100)
    assert sum(cum <= day_ft3["2024"]) == 4, "PNG subhead says a day fills four towers"
    png(lms, ctx, day_ft3["2024"], data["title"],
        f"One day of city water in 2024, a billion gallons (enough to flood Central Park more than 3½ ft deep),\n"
        f"poured into Midtown towers in turn. By midnight it fills four of them. In 1991 New York used {more}% more.")
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "one_day_of_water.html")
    print(f"wrote out/one_day_of_water.html ({size / 1e6:.1f} MB), {len(data['ctx']):,} context buildings")


if __name__ == "__main__":
    main()
