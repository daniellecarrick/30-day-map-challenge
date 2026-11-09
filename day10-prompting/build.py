"""Day 10 · Prompting only — Claude guesses where places are, from memory, and we check.

Each place in data/spot-checks.csv was given to Claude by name and country only (no web,
no tools). Its guess is joined to the real location by a line; the guess marker grows with
how far off it was.
Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained)

Data: data/spot-checks.csv (Claude's memory-only guesses + looked-up true coordinates),
      Natural Earth 1:110m countries (data/countries.geojson)
Run:  ../.venv/bin/python build.py
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Geod, Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 10: Prompting only"
PAGE_TITLE = "How well does Claude know where places are?"   # web headline: the experience; the PNG title states the finding
SOURCE = ("Data: Claude's memory-only guesses vs. USGS GNIS, BC Geographical Names, Wikipedia and "
          "OpenStreetMap · Map: Natural Earth")
CRS = "+proj=eqearth +lon_0=0 +R=6378137 +units=m"   # spherical, so the page can project visitor guesses with the same formula
WORKER_URL = os.environ.get("WORKER_URL", "")   # set after deploying worker/, e.g. https://day10-claude-geo-guess.<you>.workers.dev
NEAR_KM = 25   # the "close" threshold used in the headline and the confidence column

geod = Geod(ellps="WGS84")
to_xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform


def radius(km: float) -> float:
    """Marker radius (screen px) for a guess that was `km` off. Area-ish growth, never invisible."""
    return 3 + 0.5 * math.sqrt(km)


def load():
    rows, unresolved = [], []
    for r in csv.DictReader(open(DATA / "spot-checks.csv", encoding="utf-8")):
        if not r["true_lat"]:
            unresolved.append(r["place"])
            continue
        glat, glon, tlat, tlon = (float(r[k]) for k in ("guess_lat", "guess_lon", "true_lat", "true_lon"))
        _, _, m = geod.inv(glon, glat, tlon, tlat)
        gx, gy = to_xy(glon, glat)
        tx, ty = to_xy(tlon, tlat)
        rows.append(dict(
            place=r["place"], admin=r["admin"], country=r["country"], glat=glat, glon=glon, tlat=tlat, tlon=tlon,
            g=[gx, gy], t=[tx, ty], km=round(m / 1000, 1), conf=r["confidence"],
            pct=int(r["chance_within_25km_pct"]), why=r["reasoning"], src=r["true_source"]))
    rows.sort(key=lambda d: -d["km"])   # draw the biggest first so small ones sit on top
    return rows, unresolved


def land() -> gpd.GeoDataFrame:
    g = gpd.read_file(DATA / "countries.geojson")[["NAME", "geometry"]]
    g = g[g.NAME != "Antarctica"].to_crs(CRS)
    g["geometry"] = g.geometry.simplify(8000)
    return g


def headline(rows) -> str:
    n = len(rows)
    near = sum(d["km"] <= NEAR_KM for d in rows)
    worst = max(rows, key=lambda d: d["km"])
    return (f"Claude put {near} of {n} places within {NEAR_KM} km from memory, "
            f"but placed {worst['place']} {worst['km']:.0f} km from where it is")


def main():
    S.apply()
    rows, unresolved = load()
    g = land()
    title = headline(rows)
    print(title)
    for d in sorted(rows, key=lambda d: d["km"]):
        print(f"  {d['place']:<22} {d['km']:>7.1f} km  ({d['conf']}, {d['pct']}%)")

    # ---------------- static PNG ----------------
    fig, ax = plt.subplots(figsize=S.MAP["png_size_in"])
    g.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor=S.EDGE, linewidth=S.MAP["cell_edge_width"])
    for d in rows:
        ax.plot([d["g"][0], d["t"][0]], [d["g"][1], d["t"][1]], color=S.INK, lw=0.8, zorder=3)
        ax.scatter(*d["g"], s=math.pi * radius(d["km"]) ** 2 * 1.6, color=S.CATEGORICAL[0], alpha=0.55,
                   edgecolor=S.INK, linewidth=0.7, zorder=4)
        ax.scatter(*d["t"], s=14, color=S.INK, zorder=5)
    ax.set_axis_off()
    ax.set_position([0.02, 0.07, 0.96, 0.84])
    ax.set_aspect("equal", adjustable="datalim")
    xmin, ymin, xmax, ymax = g.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def frac(x, y):  # data -> axes fraction
        return ((x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin))

    best = min(rows, key=lambda d: d["km"])
    worst = max(rows, key=lambda d: d["km"])
    wx, wy = frac(*worst["t"])
    bx, by = frac(*best["t"])
    S.callout(ax, worst["t"], (min(wx - 0.02, 0.8), min(wy + 0.12, 0.93)), f"{worst['place']}, {worst['country']}: {worst['km']:.0f} km off",
              "A village of about 260 people on the Lena river", ha="right", halo=True)
    S.callout(ax, best["t"], (0.02, by + 0.10), f"{best['place']}, {best['admin']}: {best['km']:.1f} km off",
              "Tiny town, but Claude knew it", ha="left", halo=True)

    # size key (bottom-left, over the empty South Pacific)
    ax.text(0.02, 0.30, "Bigger circle = guess was farther off", transform=ax.transAxes, fontsize=S.SIZE["legend"],
            fontweight="semibold", va="bottom")
    for i, km in enumerate([1, 25, 100, 300]):
        y = 0.25 - i * 0.045
        ax.scatter(0.035, y, s=math.pi * radius(km) ** 2 * 1.6, transform=ax.transAxes, color=S.CATEGORICAL[0],
                   alpha=0.55, edgecolor=S.INK, linewidth=0.7, zorder=4)
        ax.text(0.06, y, f"{km} km", transform=ax.transAxes, fontsize=S.SIZE["legend"], va="center")
    ax.scatter(0.035, 0.04, s=14, transform=ax.transAxes, color=S.INK)
    ax.text(0.06, 0.04, "Actual location (line joins it to the guess)", transform=ax.transAxes,
            fontsize=S.SIZE["legend"], va="center")

    S.title(fig, title)
    note = f"{len(rows)} places · {', '.join(unresolved)} not found in any gazetteer" if unresolved else f"{len(rows)} places"
    S.credit(fig, f"Distance = geodesic km from Claude's guess (name + country only, no web) to the real location · "
                  f"{note}\n{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])

    # ---------------- interactive page ----------------
    land_fc = json.loads(g.to_json(drop_id=True))
    data = dict(land=land_fc, places=rows, unresolved=unresolved, near=NEAR_KM, title=title, page_title=PAGE_TITLE, day=DAY, source=SOURCE, worker=WORKER_URL)
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
