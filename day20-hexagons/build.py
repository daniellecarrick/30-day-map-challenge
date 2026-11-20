"""Day 20 · Hexagons — New York's street-tree mix, binned into quarter-mile hexagons.

Every living street tree the Parks Department tracks, binned into flat hexagons drawn in
NY State Plane (feet), so they look like true hexagons on the map. Two views:
  - "Most common tree": each hex colored by its top species (7 species + Other)
  - "How mixed": the top species' share of the hex, on a diverging scale centered on 10%,
    the forester's rule of thumb that no single species should pass 10% of a city's trees.
Writes:
  - out/nyc_tree_hexes.png   (static, most-common view, extremes called out)
  - out/nyc_tree_hexes.html  (interactive, segmented toggle between the two views)
  - out/nyc_tree_hexes_linkedin.png  (1080x1350 portrait for the LinkedIn feed; each borough's favorite called out)

Data: NYC Parks Forestry Tree Points (hn5i-inap), NTA 2020 (9nt8-h7nd), Borough Boundaries.
Fetch: ./fetch.sh    Run: ../.venv/bin/python build.py
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import Patch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 20: Hexagons"
SOURCE_URL = "https://data.cityofnewyork.us/d/hn5i-inap"
CRS_FT = "EPSG:2263"   # NY State Plane Long Island, feet
R = 800                # hex centre-to-corner, feet: ~0.26 mi flat-to-flat, ~0.3 mi corner to corner
MIN_TREES = 30         # hexes with fewer street trees are too small a sample to call
N_SPECIES = 7          # top species (by hexes won) get a color; the rest are "Other"
PHOTO_W = 300          # px, tooltip photo (shown at 200 CSS px)
RULE = 0.10            # 10-20-30 rule: no species above 10% of a city's trees

# Short display names, keyed by scientific name (cultivars folded into their species)
NAMES = {
    "Platanus x acerifolia": "London plane", "Gleditsia triacanthos var. inermis": "Honeylocust",
    "Prunus serrulata": "Japanese cherry", "Ginkgo biloba": "Ginkgo", "Liquidambar styraciflua": "Sweetgum",
    "Acer x freemanii": "Freeman maple", "Tilia x europaea": "Common linden",
}
COMMON = {}  # scientific name -> common name from the data, filled by load()


def taxon(genusspecies: str) -> str:
    """'Zelkova serrata 'Green Vase' - 'Green Vase' Zelkova' -> 'Zelkova serrata'."""
    sci = genusspecies.split(" - ")[0]
    sci = re.sub(r"\s*'[^']*'", "", sci).replace("x.", "x").replace("acerfolia", "acerifolia").strip()
    return re.sub(r"\s+", " ", sci)


def common(genusspecies: str) -> str:
    """'... - 'Green Vase' Zelkova' -> 'Zelkova' (common name without the cultivar)."""
    c = genusspecies.split(" - ")[-1]
    return re.sub(r"\s+", " ", re.sub(r"'[^']*'", "", c)).strip()


def nice(sp: str) -> str:
    c = NAMES.get(sp) or COMMON.get(sp, sp)
    return c[:1].upper() + c[1:]


def photos(taxa) -> dict:
    """Small inline JPEG + credit for each species, from data/photos.json (fetch_photos.py)."""
    f = DATA / "photos.json"
    meta = json.loads(f.read_text()) if f.exists() else {}
    out = {}
    for t in taxa:
        m = meta.get(t)
        if not m:
            continue
        im = Image.open(DATA / "photos" / m["file"]).convert("RGB")
        im.thumbnail((PHOTO_W, PHOTO_W))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=68, optimize=True, progressive=True)
        out[t] = {"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                  "w": im.width, "h": im.height, "credit": f"Photo: {m['artist']} · {m['license']}"}
    return out


def hex_index(x, y):
    """Pointy-top axial hex coordinates (q, r) for points in feet (cube rounding)."""
    q = (np.sqrt(3) / 3 * x - y / 3) / R
    r = (2 / 3 * y) / R
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = abs(rx - cx), abs(ry - cy), abs(rz - cz)
    a = (dx > dy) & (dx > dz)
    b = ~a & (dy > dz)
    rx[a] = -ry[a] - rz[a]
    ry[b] = -rx[b] - rz[b]
    c = ~a & ~b
    rz[c] = -rx[c] - ry[c]
    return rx.astype(int), rz.astype(int)


def hex_polygon(q, r):
    cx = R * np.sqrt(3) * (q + r / 2)
    cy = R * 1.5 * r
    ang = np.radians(np.arange(6) * 60 + 30)
    return shapely.Polygon(np.c_[cx + R * np.cos(ang), cy + R * np.sin(ang)]), cx, cy


def load():
    d = pd.read_csv(DATA / "trees.csv")
    gs = d.genusspecies.fillna("Unknown - Unknown")
    d["sp"] = gs.map(taxon)
    # common name for each species: the most frequent one among its rows, cultivar names removed
    cn = pd.DataFrame({"sp": d.sp, "c": gs.map(common)})
    COMMON.update(cn.groupby("sp").c.agg(lambda c: c.value_counts().index[0]).to_dict())
    xy = d.location.str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    pts = gpd.GeoDataFrame(d[["sp"]], geometry=gpd.points_from_xy(xy[0], xy[1]), crs=4326).to_crs(CRS_FT)
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_FT)[["boroname", "geometry"]]
    nta = gpd.read_file(DATA / "nta.geojson").to_crs(CRS_FT)[["ntaname", "boroname", "geometry"]]
    return pts, boros, nta


def bin_hexes(pts, boros, nta):
    q, r = hex_index(pts.geometry.x.values, pts.geometry.y.values)
    pts = pts.assign(q=q, r=r)
    counts = pts.groupby(["q", "r", "sp"]).size().rename("n").reset_index()
    city = pts.sp.value_counts()

    rows = []
    for (hq, hr), g in counts.groupby(["q", "r"]):
        g = g.sort_values("n", ascending=False)
        tot = int(g.n.sum())
        poly, cx, cy = hex_polygon(hq, hr)
        top = g.head(3)
        rows.append({"q": hq, "r": hr, "trees": tot, "species": len(g), "top": g.sp.iat[0],
                     "top_n": int(g.n.iat[0]), "share": g.n.iat[0] / tot,
                     "top3": [[nice(s), int(n)] for s, n in zip(top.sp, top.n)],
                     "px": cx, "py": cy, "geometry": poly})
    hx = gpd.GeoDataFrame(rows, crs=CRS_FT)
    hx = hx[hx.trees >= MIN_TREES].reset_index(drop=True)

    # clip to land so coastline hexes don't spill into the water
    land = boros.union_all()
    hx["geometry"] = hx.geometry.intersection(land)
    hx = hx[~hx.geometry.is_empty].reset_index(drop=True)

    # neighborhood name at each hex centre (nearest NTA, for centres on a pier or in the water)
    cen = gpd.GeoDataFrame(geometry=gpd.points_from_xy(hx.px, hx.py), crs=CRS_FT)
    j = gpd.sjoin_nearest(cen, nta, how="left")
    j = j[~j.index.duplicated()]
    hx["place"], hx["boro"] = j.ntaname.values, j.boroname.values

    (DATA / "needed_taxa.json").write_text(json.dumps(sorted(set(hx.top)), indent=0))  # for fetch_photos.py
    wins = hx.top.value_counts()
    cats = list(wins.index[:N_SPECIES])
    hx["cat"] = hx.top.where(hx.top.isin(cats), "other")
    return hx, cats, wins, city


def draw_linkedin(hx, boros, cats, colors, wins, boro_top, title, n, H):
    """4:5 portrait, the largest shape the LinkedIn feed shows, with type sized to read on a phone."""
    fig = plt.figure(figsize=(5.4, 6.75))
    ax = fig.add_axes([0.02, 0.085, 0.96, 0.72])
    ax.set_axis_off()
    x0, y0, x1, y1 = boros.total_bounds
    pos = ax.get_position()
    aspect = (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())
    h = max(y1 - y0, (x1 - x0) / aspect) * 1.03
    w = h * aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    ax.set_aspect("equal")
    boros.plot(ax=ax, color=S.C["ctl"], edgecolor="none")
    hx.plot(ax=ax, color=hx.color, edgecolor=S.EDGE, linewidth=0.15)
    boros.boundary.plot(ax=ax, color=S.INK, linewidth=0.3)

    # each borough's favorite tree, pointing at the middle of that species' hexes in the borough
    spots = {"Manhattan": ((0.03, 0.80), "left"), "Brooklyn": ((0.40, 0.035), "left"),
             "Staten Island": ((0.02, 0.43), "left")}
    for b, (txy, ha) in spots.items():
        sp, sh = boro_top[b]
        mine = hx[(hx.boro == b) & (hx.top == sp)]
        mx, my = mine.px.median(), mine.py.median()
        k = ((mine.px - mx) ** 2 + (mine.py - my) ** 2).idxmin()  # a real hex near the middle
        S.callout(ax, (hx.px[k], hx.py[k]), txy, f"{b}: {nice(sp)}", f"{sh:.0%} of the borough's street trees",
                  ha=ha, halo=True)

    # two-line headline, broken after "yet", shrunk until the longer line fits 94% of the width
    t1, t2 = title.split(", just ")
    head = fig.text(0.5, 0.972, t1 + ",\njust " + t2, ha="center", va="top", fontsize=17, fontweight="bold",
                    fontfamily=S.HEAD_FONT, color=S.INK, linespacing=1.18)
    renderer = fig.canvas.get_renderer()
    while head.get_window_extent(renderer).width > 0.94 * fig.bbox.width:
        head.set_fontsize(head.get_fontsize() - 0.25)
    fig.text(0.5, 0.875, f"{n:,} street trees in quarter-mile hexagons, each colored by its most common tree",
             ha="center", va="center", fontsize=8.5, color=S.MUTED)
    # legend: two rows under the subtitle, so it never sits on the map
    items = [(colors[c], nice(c)) for c in cats] + [(colors["other"], "Another species")]
    rows = [items[:4], items[4:]]
    for r, row in enumerate(rows):
        y = 0.842 - r * 0.028
        xs = np.linspace(0.07, 0.93, len(row) + 1)[:-1] + (0.86 / len(row) - 0.215) / 2
        for (col, lab), x in zip(row, xs):
            fig.patches.append(Rectangle((x, y - 0.007), 0.017, 0.0136, transform=fig.transFigure, fc=col, ec="none"))
            fig.text(x + 0.025, y, lab, ha="left", va="center", fontsize=8, color=S.INK)
    S.credit(fig, f"Cultivars counted as their species; hexagons with fewer than {MIN_TREES} street trees left blank.\n"
                  f"Data: NYC Parks, Forestry Tree Points (Nov 2026)  ·  #30DayMapChallenge {DAY}\n"
                  "daniellecarrick.github.io/30-day-map-challenge/day20", y=0.014)
    fig.savefig(OUT / "nyc_tree_hexes_linkedin.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / "nyc_tree_hexes_linkedin.png")


def main():
    S.apply()
    pts, boros, nta = load()
    n = len(pts)
    hx, cats, wins, city = bin_hexes(pts, boros, nta)
    H = len(hx)
    colors = {c: S.CATEGORICAL[i] for i, c in enumerate(cats)}
    colors["other"] = S.C["other"]
    hx["color"] = hx.cat.map(colors)

    city_share = city / n
    print(f"{n:,} trees, {H:,} hexes (>= {MIN_TREES} trees), median {hx.trees.median():.0f} trees/hex")
    for c in cats:
        print(f"  {nice(c):18s} wins {wins[c]:5d} hexes ({wins[c] / H:.1%}), {city_share[c]:.1%} of all trees")

    # headline facts that don't depend on the hexagon size: citywide and per-borough counts
    kinds = city.drop(labels=["Unknown"], errors="ignore").size
    top4 = city.head(4)
    top4_share = top4.sum() / n
    assert 0.31 <= top4_share <= 0.36, top4_share  # the headline says "a third"
    assert list(top4.index) == cats[:4], (list(top4.index), cats[:4])  # same four the map colors first
    j = gpd.sjoin(pts, boros, how="inner", predicate="within")
    boro_top = {}
    for b, g in j.groupby("boroname"):
        v = g.sp.value_counts()
        boro_top[b] = (v.index[0], v.iat[0] / len(g))
        print(f"  {b:14s} top {nice(v.index[0])} {v.iat[0] / len(g):.1%}")
    under = [b for b, (_, sh) in boro_top.items() if sh <= RULE]
    title = f"Of New York's {kinds} kinds of street tree, just four make up a third of the trees"
    print("TITLE:", title, f"({top4_share:.1%})")

    def bt(b, unit=""):  # "honeylocust in Manhattan (16%)"; proper-noun names keep their capital
        sp, sh = boro_top[b]
        name = nice(sp)
        name = name.lower() if name.split()[0] in {"Honeylocust", "Pin", "Red"} else name
        prep = "on" if b == "Staten Island" else "in"
        return f"{name} {prep} {b} ({sh:.0%}{unit})"
    boro_line = (f"The favorite changes by borough: {bt('Brooklyn', ' of its trees')}, "
                 f"{bt('Manhattan')}, {bt('Staten Island')}.")
    rule_line = (f"Citywide, London plane is {city_share[cats[0]]:.0%}. "
                 + (f"Only {' and '.join('the Bronx' if u == 'Bronx' else u for u in under)} "
                    f"keep{'s' if len(under) == 1 else ''} its top tree under 10%." if under else
                    "Every borough's top tree passes 10%."))
    print(boro_line, "|", rule_line)

    # extremes (only hexes with a solid sample): most one-species hex, most mixed hex
    solid = hx[hx.trees >= 100]
    mono = solid.loc[solid.share.idxmax()]
    mixed = solid.loc[solid.share.idxmin()]
    print(f"most uniform: {mono.place} {mono.share:.0%} {nice(mono.top)} of {mono.trees}")
    print(f"most mixed:   {mixed.place} top {mixed.share:.0%} ({nice(mixed.top)}), {mixed.species} species")

    # ---- static PNG ----
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.31, 0.05, 0.68, 0.88])
    xmin, ymin, xmax, ymax = boros.total_bounds
    ax.set_xlim(xmin - 3000, xmax + 3000)
    ax.set_ylim(ymin - 1500, ymax + 1500)
    boros.plot(ax=ax, color=S.C["ctl"], edgecolor="none")
    hx.plot(ax=ax, color=hx.color, edgecolor=S.EDGE, linewidth=S.MAP["cell_edge_width"])
    boros.boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"])
    ax.set_axis_off()
    ax.set_aspect("equal")

    def short(place):
        return place.split(" (")[0].split("-")[0].strip()

    S.callout(ax, (mono.px, mono.py), (0.74, 0.12),
              f"Most uniform: {short(mono.place)}",
              f"{mono.share:.0%} of its {mono.trees} street trees are {nice(mono.top)}",
              outline=mono.geometry, halo=True)
    S.callout(ax, (mixed.px, mixed.py), (0.385, 0.47),
              f"Most mixed: {short(mixed.place)}",
              f"{mixed.species} species; the most common,\n{nice(mixed.top)}, is only {mixed.share:.0%}",
              outline=mixed.geometry, halo=True, ha="right")

    t1, t2 = title.split(", just ")
    head = t1 + ",\njust " + t2
    fig.text(0.03, 0.93, head,
             ha="left", va="top", fontsize=S.SIZE["title"], fontweight="bold", fontfamily=S.HEAD_FONT, color=S.INK,
             linespacing=1.2)
    fig.text(0.03, 0.80,
             f"Each hexagon (about a quarter mile across) is colored by its\n"
             f"most common tree. The favorite changes by borough:\n"
             + boro_line.split(": ", 1)[1].replace("trees), ", "trees),\n")
             + "\n\nForesters suggest no species pass 10% of a city's trees.\n"
             + rule_line.replace(". Only", ".\nOnly"),
             ha="left", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED, linespacing=1.5)
    handles = [Patch(facecolor=colors[c], edgecolor="none",
                     label=f"{nice(c)}  ({wins[c] / H:.0%})") for c in cats]
    handles += [Patch(facecolor=colors["other"], edgecolor="none", label="Another species"),
                Patch(facecolor=S.C["ctl"], edgecolor="none", label=f"Fewer than {MIN_TREES} street trees")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.56), fontsize=S.SIZE["legend"] + 1,
               labelcolor=S.INK, handlelength=1.2, handleheight=1.2, labelspacing=0.7, borderaxespad=0,
               title="Most common tree  (share of hexagons)", title_fontsize=S.SIZE["legend"] + 1,
               alignment="left")
    S.credit(fig, f"{n:,} living street trees binned into {R * 2:,} ft hexagons (corner to corner); hexagons with "
             f"fewer than {MIN_TREES} trees left blank.  Data: NYC Parks, Forestry Tree Points (Nov 2026)  ·  "
             f"#30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_tree_hexes.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", OUT / "nyc_tree_hexes.png")
    draw_linkedin(hx, boros, cats, colors, wins, boro_top, title, n, H)

    # ---- interactive page (feet; page uses d3.geoIdentity) ----
    g = hx[["top", "cat", "trees", "species", "share", "top3", "place", "boro", "geometry"]].copy()
    g["share"] = g.share.round(3)
    g["geometry"] = g.geometry.simplify(15)
    b = boros.copy()
    b["geometry"] = b.geometry.simplify(60)
    # diverging around the 10% rule: blue = more mixed than the rule, amber/brown = one species dominates
    div = S.DIVERGING
    stops = [(0.04, div[0]), (0.07, div[1]), (RULE, div[2]), (0.25, div[3]), (0.50, div[4])]
    web = {
        "title": title, "day": DAY, "source_url": SOURCE_URL, "n": n, "hexes": H, "min_trees": MIN_TREES,
        "rule": RULE, "stops": stops, "top4_pct": round(100 * top4_share), "boro_line": boro_line,
        "rule_line": rule_line,
        "cats": [{"key": c, "label": nice(c), "wins": int(wins[c]), "city": round(100 * city_share[c], 1),
                  "var": f"--cat-{i + 1}"} for i, c in enumerate(cats)] + [
                 {"key": "other", "label": "Another species", "var": "--other"}],
        "london_city": round(100 * city_share[cats[0]], 1),
        "photos": photos(sorted(set(hx.top))),
        "names": {t: nice(t) for t in sorted(set(hx.top))},
        "features": json.loads(g.to_json(drop_id=True)),
        "boroughs": json.loads(b.to_json(drop_id=True)),
    }
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "nyc_tree_hexes.html")
    print("wrote", OUT / "nyc_tree_hexes.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
