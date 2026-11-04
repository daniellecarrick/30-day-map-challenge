"""Day 4 · Clusters — Where crashes that hurt people walking or biking pile up in NYC.

Every crash in 2023-2025 that injured or killed a pedestrian or cyclist, clustered with
DBSCAN at two scales (a corner, a block or two). Writes:
  - out/nyc_crash_hotspots.png    (static, corner scale, worst spot per borough called out)
  - out/nyc_crash_hotspots.html   (interactive, segmented toggle between the two scales)

Data: NYC Open Data "Motor Vehicle Collisions - Crashes" (h9gi-nx95), "Borough Boundaries".
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
from matplotlib.lines import Line2D
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 4: Clusters"
SOURCE_URL = "https://data.cityofnewyork.us/d/h9gi-nx95"
CRS_M = "EPSG:32618"  # UTM 18N, metres
YEARS = "2023–2025"
DAYS = 1096  # 2023-01-01 through 2025-12-31

# DBSCAN settings per view: a hotspot is a group where every crash has at least `min_n`
# crashes (itself included) within `eps` metres, chained together.
SCALES = [
    {"key": "corner", "label": "Corners", "eps": 25, "min_n": 8},
    {"key": "block", "label": "Blocks", "eps": 100, "min_n": 20},
]

HOT = S.SEQUENTIAL_AMBER[2]       # hotspot fill
HOT_DARK = S.SEQUENTIAL_AMBER[4]  # hotspot edge
# Corner circles: radius grows LINEARLY from the smallest to the biggest hotspot (not area = value),
# so the worst corners stand out more than a true area scale would allow.
WEB_R = (1.8, 8.0)    # radius range, viewBox px
PNG_D = (1.6, 10.0)   # marker diameter range, pt

ABBR = [(r"\bStreet\b", "St"), (r"\bAvenue\b", "Ave"), (r"\bAv\b", "Ave"), (r"\bBoulevard\b", "Blvd"),
        (r"\bRoad\b", "Rd"), (r"\bParkway\b", "Pkwy"), (r"\bPlace\b", "Pl"), (r"\bDrive\b", "Dr"),
        (r"\bExpressway\b", "Expy"), (r"^East\b", "E"), (r"^West\b", "W"), (r"^North\b", "N"),
        (r"^South\b", "S"), (r"\bSaint\b", "St")]


def nice(s) -> str:
    """'WEST 124 STREET' -> 'W 124th St'."""
    if not isinstance(s, str) or not s.strip():
        return ""
    s = re.sub(r"\s+", " ", s.strip()).title()
    for pat, rep in ABBR:
        s = re.sub(pat, rep, s)

    def ordinal(m):
        n = int(m.group(1))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    s = re.sub(r"\b(\d+)\b(?= (St|Ave|Rd|Dr|Pl|Ter|Blvd)\b)", ordinal, s)
    return re.sub(r"(\d)(St|Nd|Rd|Th)\b", lambda m: m.group(1) + m.group(2).lower(), s)


def plural(k: int, word: str) -> str:
    return f"{k:,} {word}{'' if k == 1 else 's'}"


def hurt(w) -> str:
    """'19 pedestrians, 6 cyclists hurt' (+ killed when any)."""
    s = f"{plural(w.ped_inj, 'pedestrian')}, {plural(w.cyc_inj, 'cyclist')} hurt"
    return s + (f", {w.killed} killed" if w.killed else "")


def load():
    d = pd.read_csv(DATA / "crashes_ped_cyc_2023_2025.csv")
    total = len(d)
    d = d[d.latitude.between(40.45, 40.95) & d.longitude.between(-74.3, -73.65)].copy()
    d["ped_inj"] = d.number_of_pedestrians_injured
    d["cyc_inj"] = d.number_of_cyclist_injured
    d["killed"] = d.number_of_pedestrians_killed + d.number_of_cyclist_killed
    on, cross = d.on_street_name.map(nice), d.cross_street_name.map(nice)
    d["place"] = np.where((on != "") & (cross != ""), on + " & " + cross,
                          np.where(on != "", on, d.off_street_name.str.replace(r"^\s*\d+[-\d]*\s+", "", regex=True).map(nice)))
    d["pair"] = [" & ".join(sorted(p.split(" & "))) for p in d.place]
    pts = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs="EPSG:4326").to_crs(CRS_M)

    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    pts = gpd.sjoin_nearest(pts, boros, how="left", max_distance=500).drop(columns="index_right")
    pts = pts[~pts.index.duplicated()]
    pts = pts[pts.boroname.notna()].reset_index(drop=True)   # drops bad geocodes far off land
    print(f"{len(pts):,} of {total:,} crashes have a usable location")
    return pts, boros, total


def clusters(pts: gpd.GeoDataFrame, eps: float, min_n: int) -> gpd.GeoDataFrame:
    X = np.c_[pts.geometry.x, pts.geometry.y]
    lab = DBSCAN(eps=eps, min_samples=min_n).fit_predict(X)
    rows = []
    for c in range(lab.max() + 1):
        m = pts[lab == c]
        xy = X[lab == c]
        named = m[m.place.str.contains(" & ")]
        if named.empty:
            named = m[m.place != ""]
        if len(named):
            top = named.pair.value_counts().index[0]
            name = named[named.pair == top].place.value_counts().index[0]  # most common spelling/order
        else:
            name = "Unnamed location"
        rows.append({
            "name": name, "boroname": m.boroname.mode().iat[0], "crashes": len(m),
            "ped_inj": int(m.ped_inj.sum()), "cyc_inj": int(m.cyc_inj.sum()), "killed": int(m.killed.sum()),
            "spots": len(np.unique(xy.round(0), axis=0)),
            "extent_m": float(np.ptp(xy, 0).max()),
            "x": float(xy[:, 0].mean()), "y": float(xy[:, 1].mean()),
            "geometry": shapely.MultiPoint(xy).convex_hull.buffer(eps / 2),
        })
    g = gpd.GeoDataFrame(rows, crs=CRS_M).sort_values("crashes", ascending=False).reset_index(drop=True)
    return g


def main():
    S.apply()
    pts, boros, total = load()
    n = len(pts)

    views = {}
    for sc in SCALES:
        g = clusters(pts, sc["eps"], sc["min_n"])
        share = g.crashes.sum() / n
        views[sc["key"]] = (sc, g, share)
        print(f"{sc['label']:>13}: {len(g):4d} hotspots hold {g.crashes.sum():,} crashes ({share:.1%}); "
              f"biggest {g.crashes.iat[0]} at {g.name.iat[0]} ({g.boroname.iat[0]})")

    sc, corners, share = views["corner"]
    worst = corners.iloc[0]
    assert worst["name"] == "W 42nd St & 8th Ave", worst["name"]  # the headline names this corner
    weeks = round(DAYS / 7 / worst.crashes)
    words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    title = (f"At 8th Ave and 42nd St, someone walking or biking was hurt in a crash "
             f"about every {words.get(weeks, weeks)} weeks")
    print("TITLE:", title, f"({DAYS / 7 / worst.crashes:.1f} weeks; {worst.crashes} crashes in {DAYS} days)")

    # ---- static PNG: corner scale ----
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.30, 0.06, 0.69, 0.86])
    xmin, ymin, xmax, ymax = boros.total_bounds
    ax.set_xlim(xmin - 1500, xmax + 1500)
    ax.set_ylim(ymin - 800, ymax + 800)
    boros.plot(ax=ax, color=S.C["ctl"], edgecolor="white", linewidth=0.6)
    ax.scatter(pts.geometry.x, pts.geometry.y, s=0.6, color=S.MUTED, alpha=0.45, linewidths=0, zorder=2)
    lo_n, hi_n = sc["min_n"], int(corners.crashes.max())
    def png_s(c):  # scatter size (pt^2) for a hotspot with c crashes
        return (PNG_D[0] + (np.asarray(c) - lo_n) / (hi_n - lo_n) * (PNG_D[1] - PNG_D[0])) ** 2
    ax.scatter(corners.x, corners.y, s=png_s(corners.crashes), color=HOT, edgecolor=HOT_DARK,
               linewidth=0.5, alpha=0.9, zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # worst spot per borough, labels placed in open water / off the coast
    spots = {"Manhattan": (0.05, 0.80), "Bronx": (0.99, 0.985), "Brooklyn": (0.06, 0.30),
             "Queens": (0.99, 0.75), "Staten Island": (0.02, 0.09)}
    for boro, txy in spots.items():
        b = corners[corners.boroname == boro]
        if b.empty:
            continue
        w = b.iloc[0]
        ha = "right" if txy[0] > 0.5 else "left"
        ax.scatter([w.x], [w.y], s=png_s(w.crashes), color=HOT, edgecolor=S.INK, linewidth=1.1, zorder=6)
        S.callout(ax, (w.x, w.y), txy, f"{boro}: {w['name']}", f"{w.crashes} crashes · {hurt(w)}", ha=ha, halo=True)

    # left column: title block + legend
    fig.text(0.03, 0.93, title.replace(" walking", "\nwalking").replace(" about", "\nabout"), ha="left", va="top",
             fontsize=S.SIZE["title"], fontweight="bold", fontfamily=S.HEAD_FONT, color=S.INK, linespacing=1.2)
    fig.text(0.03, 0.75,
             f"It's the worst of {len(corners)} hotspots in the city: places with\n"
             f"{sc['min_n']} or more crashes within {sc['eps']} m of each other (orange).\n"
             f"Each gray dot is one of the {n:,} crashes in {YEARS} that\n"
             f"injured or killed a pedestrian or cyclist.",
             ha="left", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED, linespacing=1.5)
    handles = [Line2D([], [], marker="o", ls="", color=S.MUTED, alpha=0.6, markersize=2, label="One crash")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.58), fontsize=S.SIZE["legend"] + 1,
               labelcolor=S.INK, handletextpad=0.6, labelspacing=0.9, borderaxespad=0)
    # smallest and biggest hotspot side by side in one row: "Hotspots ● 8  ⬤ 25 crashes"
    lax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_axis_off()
    row_y, fs = 0.525, S.SIZE["legend"] + 1
    lax.text(0.03, row_y, "Hotspots", ha="left", va="center", fontsize=fs, color=S.INK)
    for cx, c, label in [(0.078, lo_n, f"{lo_n}"), (0.098, hi_n, f"{hi_n} crashes")]:
        lax.scatter([cx], [row_y], s=png_s(c), color=HOT, edgecolor=HOT_DARK, linewidth=0.5)
        lax.text(cx + 0.0045 + np.sqrt(png_s(c)) / 2 / 72 / S.MAP["png_size_in"][0], row_y, label,
                 ha="left", va="center", fontsize=fs, color=S.INK)
    S.credit(fig, f"Hotspots: DBSCAN clustering, {sc['eps']} m radius, at least {sc['min_n']} crashes.  "
             f"{total - n:,} crashes without a location are left out.  "
             f"Data: NYC Open Data, Motor Vehicle Collisions (NYPD), {YEARS}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_crash_hotspots.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", OUT / "nyc_crash_hotspots.png")

    # ---- interactive page (metres; page uses d3.geoIdentity) ----
    web = {"title": title, "n": n, "total": total, "years": YEARS, "day": DAY, "source_url": SOURCE_URL,
           "hot": HOT, "hot_dark": HOT_DARK, "amber": S.SEQUENTIAL_AMBER, "web_r": WEB_R, "views": []}
    for key, (sc, g, share) in views.items():
        gg = g[["name", "boroname", "crashes", "ped_inj", "cyc_inj", "killed", "spots", "x", "y", "geometry"]].copy()
        gg["geometry"] = gg.geometry.simplify(4)
        gg["x"], gg["y"] = gg.x.round(0), gg.y.round(0)
        web["views"].append({**sc, "count": len(g), "share": round(100 * share, 1),
                             "in_clusters": int(g.crashes.sum()),
                             "features": json.loads(gg.to_json(drop_id=True))})
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(20)
    web["boroughs"] = json.loads(b.to_json(drop_id=True))
    web["points"] = np.c_[pts.geometry.x, pts.geometry.y].round(0).astype(int).ravel().tolist()
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "nyc_crash_hotspots.html")
    print("wrote", OUT / "nyc_crash_hotspots.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
