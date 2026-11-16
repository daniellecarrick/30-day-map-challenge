"""Day 16 · Collaborative map: the world as everyone drew it.

Every outline visitors drew in Day 13's "Can you draw these countries?" game, layered on one world map.
For each country: a heat map of how many drawings covered each spot, next to the real border.

The page fetches the drawings live from the Day 13 Worker every time it opens, so it keeps growing as
people play Day 13. This script embeds a snapshot (shown first, and if the Worker can't be reached) and
draws the PNG from the same snapshot.

Writes:
  - out/map.png    (static, for posting)
  - out/map-only.png (just the map: no title, callouts, legend or credit)
  - out/map.html   (interactive, self-contained, live)

Data: visitors' drawings (Day 13 Worker + D1); Natural Earth 1:110m countries
Run:  ../.venv/bin/python build.py      (WORKER_URL defaults to the live Day 13 Worker)
"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import box as bbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
import style as S  # noqa: E402

# Day 13 owns the drawings, the country list and the scoring. Reuse them so both days agree.
os.environ.setdefault("WORKER_URL", "https://day13-draw-the-countries.daniellecarrick.workers.dev")
sys.path.insert(0, str(ROOT / "day13-interactions"))
import build as D13  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 16: Collaborative map"
SOURCE = "Data: drawings by visitors to Day 13 · Map: Natural Earth"
DAY13_URL = "https://daniellecarrick.github.io/30-day-map-challenge/day13/"
CRS = D13.CRS
TARGETS = D13.TARGETS
MIN = D13.MIN_DRAWINGS
MAJORITY = 0.5
LEVELS = D13.LEVELS          # share-of-drawings bands; colors = sequential_blue
GRID = 500                   # cells along the grid's long side


def box_for(polys):
    """Grid box: every drawing, so even the faintest band isn't cut off."""
    x0, y0, x1, y1 = shapely.union_all(polys).bounds
    m = 0.03 * max(x1 - x0, y1 - y0)
    return x0 - m, y0 - m, x1 + m, y1 + m


def km_between(a, b):
    """Great-circle km between two equal-area points."""
    ll = gpd.GeoSeries([a, b], crs=CRS).to_crs("EPSG:4326")
    (lo1, la1), (lo2, la2) = [(p.x, p.y) for p in ll]
    r = math.radians
    h = math.sin(r(la2 - la1) / 2) ** 2 + math.cos(r(la1)) * math.cos(r(la2)) * math.sin(r(lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def analyse(countries, drawings):
    eq = countries.to_crs(CRS)
    truth = {c: eq.loc[eq.code == c].geometry.union_all() for c in TARGETS}
    out = {}
    for code, rings in drawings.items():
        polys = [p for p in (D13.to_poly(r) for r in rings) if not p.is_empty]
        st = dict(n=len(polys), polys=polys, truth=truth[code], area=truth[code].area)
        if len(polys) >= MIN:
            sc = [D13.score(p, truth[code]) for p in polys]
            st["iou"] = statistics.median(s["iou"] for s in sc)
            st["size"] = statistics.median(s["size"] for s in sc)
            st["km"] = statistics.median(km_between(p.centroid, truth[code].centroid) for p in polys)
            b = box_for(polys)
            xs, ys, share = D13.coverage(polys, b, n=GRID)
            st.update(xs=xs, ys=ys, share=share, agree=float(share.max()))
            st["agree_k"] = round(st["agree"] * len(polys))
        out[code] = st
    return out


def findings(st):
    """The page's insight lines (the page recomputes them from live data with the same rules)."""
    ok = {c: s for c, s in st.items() if s["n"] >= MIN}
    f = {}
    if len(ok) >= 6:
        by_area = sorted(ok, key=lambda c: ok[c]["area"])
        small = [r["size"] for c in by_area[:5] for r in [D13.score(p, ok[c]["truth"]) for p in ok[c]["polys"]]]
        big = [r["size"] for c in by_area[-5:] for r in [D13.score(p, ok[c]["truth"]) for p in ok[c]["polys"]]]
        f["small"] = dict(codes=by_area[:5], small=statistics.median(small), big=statistics.median(big))
    if "NER" in ok and "NGA" in st:
        ngr, nga = ok["NER"]["truth"], st["NGA"]["truth"]
        f["niger"] = dict(n=ok["NER"]["n"], k=sum(p.intersection(nga).area > p.intersection(ngr).area for p in ok["NER"]["polys"]))
    return f


def headline(st):
    ok = {c: s for c, s in st.items() if s["n"] >= MIN}
    if not ok:
        return "The world, as everyone drew it"
    best = max(ok, key=lambda c: ok[c]["iou"])
    worst = min(ok, key=lambda c: ok[c]["agree"])
    w = ok[worst]
    return (f"The typical drawing of {TARGETS[best][0]} overlaps the real one by {ok[best]['iou']:.0%}. "
            f"For {TARGETS[worst][0]}, no spot is in more than {w['agree_k']} of {w['n']} drawings")


def draw_map(ax, countries, st):
    """Land, the heat map and the real borders: the map with nothing on top."""
    land = gpd.GeoSeries([countries.geometry.union_all()], crs=countries.crs).to_crs(CRS).simplify(8000)
    land.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor=S.MUTED, linewidth=0.3)
    ok = [c for c in TARGETS if st[c]["n"] >= MIN]
    # Paint band by band across every country, so where two countries' heat maps meet the higher share wins
    for i in range(len(LEVELS) - 1):
        for c in ok:
            s = st[c]
            ax.contourf(s["xs"], s["ys"], s["share"], levels=[LEVELS[i], LEVELS[-1]],
                        colors=[S.SEQUENTIAL_BLUE[i]], zorder=2)
    gpd.GeoSeries([st[c]["truth"] for c in TARGETS]).boundary.plot(ax=ax, color=S.INK, linewidth=0.7, zorder=3)
    x0, y0, x1, y1 = land.total_bounds
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_axis_off()
    return ok


def png_bare(countries, st):
    """Map only: no title, callouts, legend or credit (for reuse in other layouts)."""
    S.apply()
    fig = plt.figure(figsize=(S.MAP["png_size_in"][0], S.MAP["png_size_in"][0] * 0.5))
    ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
    draw_map(ax, countries, st)
    fig.savefig(OUT / "map-only.png", dpi=S.MAP["png_dpi"], bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def png(countries, st, title):
    S.apply()
    fig, ax = plt.subplots(figsize=S.MAP["png_size_in"])
    ax.set_position([0.01, 0.07, 0.98, 0.83])
    ok = draw_map(ax, countries, st)

    # call out the extremes
    if ok:
        best = max(ok, key=lambda c: st[c]["iou"])
        b = st[best]
        S.callout(ax, (b["truth"].centroid.x, b["truth"].centroid.y), (0.17, 0.66), f"{TARGETS[best][0]}: drawn best",
                  f"typical drawing overlaps\nthe real border by {b['iou']:.0%}", ha="right")
    if ok:
        worst = min(ok, key=lambda c: st[c]["agree"])
        w = st[worst]
        S.callout(ax, (w["truth"].centroid.x, w["truth"].centroid.y), (0.86, 0.3), f"{TARGETS[worst][0]}: no agreement",
                  f"no spot in more than {w['agree_k']} of {w['n']} drawings", halo=True)
    if "NER" in ok:
        k, n = sum(p.intersection(st["NGA"]["truth"]).area > p.intersection(st["NER"]["truth"]).area
                   for p in st["NER"]["polys"]), st["NER"]["n"]
        t = st["NER"]["truth"]
        S.callout(ax, (t.centroid.x, t.centroid.y), (0.445, 0.6), "Niger or Nigeria?",
                  f"{k} of {n} Niger drawings cover\nmore of Nigeria than Niger", ha="right", halo=True)
    f = findings(st)
    big = max(ok, key=lambda c: st[c]["size"], default=None)
    if big:
        g = st[big]
        S.callout(ax, (g["truth"].centroid.x, g["truth"].centroid.y), (0.42, 0.12), f"{TARGETS[big][0]}: drawn {g['size']:.1f}× too big",
                  f"Small countries come out big: the 5 smallest {f['small']['small']:.1f}×, the 5 biggest {f['small']['big']:.1f}×"
                  if "small" in f else "median drawing", ha="left")

    # legend, bottom left over the Pacific
    lax = fig.add_axes([0.03, 0.2, 0.17, 0.016])
    for i, c in enumerate(S.SEQUENTIAL_BLUE[:len(LEVELS) - 1]):
        lax.axvspan(i, i + 1, color=c)
    lax.set_xlim(0, len(LEVELS) - 1); lax.set_yticks([])
    lax.set_xticks(range(len(LEVELS))); lax.set_xticklabels([f"{min(v, 1):.0%}" for v in LEVELS], fontsize=S.SIZE["legend"])
    for sp in lax.spines.values():
        sp.set_visible(False)
    lax.set_title("Share of a country's drawings\ncovering this spot", fontsize=S.SIZE["legend"], fontweight="semibold", pad=4, loc="left")
    fig.text(0.03, 0.165, "——  The real border", fontsize=S.SIZE["legend"], color=S.INK, va="top")

    n = sum(st[c]["n"] for c in TARGETS)
    S.title(fig, title)
    fig.text(0.5, 0.925, f"{n} outlines of {len(TARGETS)} countries, drawn freehand by visitors on a map with coastlines but no borders",
             ha="center", fontsize=S.SIZE["panel_title"], color=S.MUTED)
    S.credit(fig, "Overlap = shared area ÷ combined area, on an equal-area map. Add yours: Day 13, “Can you draw these countries?”\n"
                  f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])


def main():
    countries = D13.load_countries()
    drawings = D13.load_drawings()
    st = analyse(countries, drawings)
    title = headline(st)
    print(title)
    for c, s in sorted(st.items(), key=lambda kv: -kv[1].get("agree", 0)):
        if s["n"] >= MIN:
            print(f"  {TARGETS[c][0]:<12} n={s['n']:<3} agree {s['agree']:.0%}  overlap {s['iou']:.0%}  size {s['size']:.2f}×  {s['km']:.0f} km")
    print(" ", findings(st))
    png(countries, st, title)
    png_bare(countries, st)

    # ---------------- interactive page (lon/lat in thousandths of a degree; see Day 13 for why) ----------------
    milli = lambda g: shapely.transform(g, lambda xy: np.round(xy * 1000))
    c = countries.copy()
    c["geometry"] = c.geometry.simplify(0.02).apply(milli)
    land = gpd.GeoSeries([countries.geometry.union_all()], crs=countries.crs).simplify(0.02).apply(milli)
    fc = lambda gdf: json.loads(gdf.to_json(drop_id=True).replace(".0,", ",").replace(".0]", "]"))
    snap = {code: [[[round(x * 100), round(y * 100)] for x, y in r] for r in rings] for code, rings in drawings.items()}
    data = dict(
        land=fc(gpd.GeoDataFrame(geometry=land)),
        countries=fc(c[["code", "name", "geometry"]]),
        targets=[dict(code=k, name=v[0]) for k, v in TARGETS.items()],
        snapshot=snap,                    # rings in hundredths of a degree, as of this build
        snapshot_date=str(np.datetime64("today")),
        min_drawings=MIN, majority=MAJORITY, levels=LEVELS[:-1],
        worker=os.environ["WORKER_URL"].rstrip("/"), day13=DAY13_URL,
        title=title, day=DAY,
    )
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
