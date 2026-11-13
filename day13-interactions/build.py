"""Day 13 · Interactions — can you draw these countries?

Visitors draw the outline of a country on a blank world map (coastlines only), see the real one,
and after five countries unlock a heat map of everyone's drawings. The drawings live in a
Cloudflare Worker + D1 database (worker/). This script builds the page and, from the drawings
collected so far, the PNG.
Writes:
  - out/map.png    (static, for posting: small multiples of the crowd's drawings)
  - out/map.html   (interactive, self-contained)

Data: Natural Earth 1:110m countries (data/countries.geojson); visitors' drawings (worker/)
Run:  WORKER_URL=https://day13-draw-the-countries.<you>.workers.dev ../.venv/bin/python build.py
      (without WORKER_URL the page is built but can't save or show drawings; the PNG uses
       data/drawings/*.json if they exist)
"""
import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 13: Interactions"
PAGE_TITLE = "Can you draw these countries?"
SOURCE = "Data: visitors' drawings · Map: Natural Earth"
CRS = "+proj=eqearth +lon_0=0 +R=6378137 +units=m"   # equal-area, so drawn vs real area compares fairly
WORKER_URL = os.environ.get("WORKER_URL", "").rstrip("/")
MIN_DRAWINGS = 10   # fewer than this and a country's heat map says "not enough drawings yet"

# code -> (display name, group). Keep in step with COUNTRIES in worker/src/index.js.
TARGETS = {
    "RUS": ("Russia", "size"), "CAN": ("Canada", "size"), "BRA": ("Brazil", "size"),
    "COD": ("DR Congo", "size"), "MEX": ("Mexico", "size"),
    "UKR": ("Ukraine", "news"), "IRN": ("Iran", "news"), "VEN": ("Venezuela", "news"), "AFG": ("Afghanistan", "news"),
    "NER": ("Niger", "confused"),
}
PNG_PANELS = ["RUS", "CAN", "COD", "BRA", "UKR", "NER"]   # big-north, big-equator, in the news, confused


def load_countries() -> gpd.GeoDataFrame:
    g = gpd.read_file(DATA / "countries.geojson")[["ADM0_A3", "NAME", "geometry"]]
    g = g[g.NAME != "Antarctica"].rename(columns={"ADM0_A3": "code", "NAME": "name"})
    g.loc[g.code.isin(TARGETS), "name"] = g.code.map(lambda c: TARGETS.get(c, ("",))[0])
    return g.reset_index(drop=True)


def load_drawings() -> dict[str, list[list]]:
    """code -> list of rings ([[lon, lat], ...]). Fetches from the Worker and caches to data/drawings/."""
    cache = DATA / "drawings"
    cache.mkdir(exist_ok=True)
    out = {}
    for code in TARGETS:
        f = cache / f"{code}.json"
        if WORKER_URL:
            try:
                req = urllib.request.Request(f"{WORKER_URL}/drawings?country={code}", headers={"User-Agent": "day13-build"})
                f.write_bytes(urllib.request.urlopen(req, timeout=20).read())
            except Exception as e:  # keep going with whatever is cached
                print(f"  couldn't fetch {code}: {e}")
        out[code] = [d["ring"] for d in json.loads(f.read_text())["drawings"]] if f.exists() else []
    return out


def to_poly(ring, crs=CRS):
    """A visitor's ring (lon/lat) as a valid polygon in the equal-area CRS."""
    ring = [(min(180, max(-180, x)), min(89.9, max(-89.9, y))) for x, y in ring]
    try:
        s = gpd.GeoSeries([Polygon(ring)], crs="EPSG:4326").to_crs(crs)
        return make_valid(s.iloc[0]).buffer(0)
    except Exception:
        return Polygon()


def score(drawn, truth) -> dict:
    inter = drawn.intersection(truth).area
    return dict(iou=inter / drawn.union(truth).area, size=drawn.area / truth.area)


def summarise(countries, drawings):
    eq = countries.to_crs(CRS)
    stats = {}
    for code, rings in drawings.items():
        truth = eq.loc[eq.code == code].geometry.union_all()
        polys = [to_poly(r) for r in rings]
        polys = [p for p in polys if not p.is_empty]
        sc = [score(p, truth) for p in polys]
        stats[code] = dict(
            n=len(polys), polys=polys, truth=truth,
            iou=statistics.median(s["iou"] for s in sc) if sc else None,
            size=statistics.median(s["size"] for s in sc) if sc else None,
        )
    return stats


def headline(stats) -> str:
    ok = {c: s for c, s in stats.items() if s["n"] >= MIN_DRAWINGS}
    if len(ok) < 2:
        return PAGE_TITLE
    big = max(ok, key=lambda c: ok[c]["size"])
    small = min(ok, key=lambda c: ok[c]["size"])
    b, s = ok[big]["size"], ok[small]["size"]
    if b > 1.25 and s < 0.8:
        return (f"People drew {TARGETS[big][0]} {b:.1f}× too big "
                f"and {TARGETS[small][0]} at {s:.0%} of its real size")
    best = max(ok, key=lambda c: ok[c]["iou"])
    worst = min(ok, key=lambda c: ok[c]["iou"])
    return (f"People's drawings of {TARGETS[best][0]} overlapped the real one by {ok[best]['iou']:.0%}. "
            f"For {TARGETS[worst][0]}, just {ok[worst]['iou']:.0%}")


def coverage(polys, bounds, n=220):
    """Share of drawings covering each cell of an n×n grid over `bounds` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bounds
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys)
    count = np.zeros_like(X)
    for p in polys:
        shapely.prepare(p)
        count += shapely.contains_xy(p, X, Y)
    return xs, ys, count / max(len(polys), 1)


def view_bounds(truth, pad=0.75):
    """Square view around the country's main landmass (so Russia's tip past 180° doesn't widen it),
    with extra room at the top for the panel heading."""
    parts = sorted(getattr(truth, "geoms", [truth]), key=lambda p: -p.area)
    x0, y0, x1, y1 = parts[0].bounds
    m = max(x1 - x0, y1 - y0) * pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - m, cy - m, cx + m, cy + m * 1.35


LEVELS = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.001]   # share-of-drawings bands; colors = sequential_blue


def png(countries, stats, title):
    S.apply()
    land = countries.to_crs(CRS)
    land["geometry"] = land.geometry.simplify(8000)
    if not any(st["n"] >= MIN_DRAWINGS for st in stats.values()):
        return teaser(countries)
    fig, axes = plt.subplots(2, 3, figsize=S.MAP["png_size_in"])
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.15, wspace=0.04, hspace=0.08)
    for ax, code in zip(axes.flat, PNG_PANELS):
        st = stats[code]
        b = view_bounds(st["truth"])
        land.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor="none")
        if st["n"] >= MIN_DRAWINGS:
            xs, ys, share = coverage(st["polys"], b)
            ax.contourf(xs, ys, share, levels=LEVELS, colors=S.SEQUENTIAL_BLUE[:len(LEVELS) - 1], zorder=2)
            sub = f"{st['n']} drawings · median overlap {st['iou']:.0%} · drawn {st['size']:.1f}× real size"
        else:
            sub = f"{st['n']} drawings so far, not enough to map"
        gpd.GeoSeries([st["truth"]]).boundary.plot(ax=ax, color=S.INK, linewidth=1.1, zorder=3)
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        ax.set_aspect("equal"); ax.set_axis_off()
        # paper band behind the heading, so the title and stats never sit on land
        ax.axhspan(b[3] - (b[3] - b[1]) * 0.16, b[3], color=S.BG, zorder=4)
        S.panel_heading(ax, TARGETS[code][0], sub)
        ax.texts[-1].set_y(0.91)   # these panels are taller than usual; give the stats line room under the name
        for t in ax.texts:
            t.set_zorder(5)   # above the paper band
    # legend: the share bands, bottom center
    lax = fig.add_axes([0.3, 0.085, 0.4, 0.018])
    for i, c in enumerate(S.SEQUENTIAL_BLUE[:len(LEVELS) - 1]):
        lax.axvspan(i, i + 1, color=c)
    lax.set_xlim(0, len(LEVELS) - 1); lax.set_yticks([])
    lax.set_xticks(range(len(LEVELS))); lax.set_xticklabels([f"{v:.0%}" if v <= 1 else "100%" for v in LEVELS], fontsize=S.SIZE["legend"])
    for s in lax.spines.values():
        s.set_visible(False)
    lax.set_title("Share of people whose drawing covered this spot  ·  ink line = the real border",
                  fontsize=S.SIZE["legend"], fontweight="semibold", pad=4)
    S.title(fig, title)
    S.credit(fig, "Visitors drew each country freehand on a blank Equal Earth map (coastlines only); overlap = shared area ÷ combined area\n"
                  f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])


def teaser(land):
    """Before there are drawings: a blank coastline map, so the link preview gives nothing away."""
    fig, ax = plt.subplots(figsize=S.MAP["png_size_in"])
    gpd.GeoSeries([land.geometry.union_all()], crs=land.crs).to_crs(CRS).simplify(8000).plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"],
                                                    edgecolor=S.MUTED, linewidth=0.5)
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_position([0.03, 0.08, 0.94, 0.8])
    S.title(fig, PAGE_TITLE)
    fig.text(0.5, 0.905, "Outline five countries on a map with no borders, then see how everyone else drew them",
             ha="center", fontsize=S.SIZE["panel_title"], color=S.MUTED)
    S.credit(fig, f"Map: Natural Earth  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])


def main():
    countries = load_countries()
    drawings = load_drawings()
    stats = summarise(countries, drawings)
    title = headline(stats)
    print(title)
    for code, st in sorted(stats.items(), key=lambda kv: -(kv[1]["n"])):
        if st["n"]:
            print(f"  {TARGETS[code][0]:<12} n={st['n']:<4} overlap {st['iou']:.0%}  size {st['size']:.2f}×")
    png(countries, stats, title)

    # ---------------- interactive page (lon/lat; the page projects with d3.geoEqualEarth) ----------------
    # Coordinates go out as integer thousandths of a degree: S.build_page trims decimals on any number
    # with 3+ integer digits (meant for metres), which would blur longitudes past ±100°.
    milli = lambda geom: shapely.transform(geom, lambda xy: np.round(xy * 1000))
    c = countries.copy()
    c["geometry"] = c.geometry.simplify(0.02).apply(milli)
    land = gpd.GeoSeries([countries.geometry.union_all()], crs=countries.crs).simplify(0.02).apply(milli)
    fc = lambda gdf: json.loads(gdf.to_json(drop_id=True).replace(".0,", ",").replace(".0]", "]"))
    data = dict(
        land=fc(gpd.GeoDataFrame(geometry=land)),
        countries=fc(c[["code", "name", "geometry"]]),
        targets=[dict(code=k, name=v[0], group=v[1]) for k, v in TARGETS.items()],
        min_drawings=MIN_DRAWINGS, title=PAGE_TITLE, day=DAY, source=SOURCE, worker=WORKER_URL,
    )
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
