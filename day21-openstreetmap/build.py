"""Day 21 · OpenStreetMap — ten unusual "folk tags" volunteers have mapped in New York City.

OpenStreetMap has no fixed list of things you may map: mappers invent and agree on tags as they go.
This maps ten of the odd ones in the five boroughs (ghost bikes, plate readers, tree stumps, ...).
Where a borough looks empty it usually means nobody has mapped it yet, so each view says where the
points cluster.
Writes:
  - out/nyc_folk_tags.png    (static, small multiples: one panel per tag)
  - out/nyc_folk_tags_linkedin.png  (1080x1350 portrait for the LinkedIn feed, same ten maps in a 3x4 grid)
  - out/nyc_folk_tags.html   (interactive: pick a tag from a dropdown; hover a dot for its tags,
                              click to open it on openstreetmap.org)

Data: OpenStreetMap contributors (ODbL), Geofabrik New York extract of 2026-11-20.
Get it with ./fetch.sh, then ../.venv/bin/python extract.py  ->  data/folk_tags.geojson
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from folk import FOLK, SHOW

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 21: OpenStreetMap"
EXTRACT_DATE = "Nov 20, 2026"
SOURCE = f"Data: © OpenStreetMap contributors (ODbL), Geofabrik extract of {EXTRACT_DATE}"
SOURCE_URL = "https://www.openstreetmap.org/copyright"
CRS_M = "EPSG:32618"   # UTM 18N, metres
BORO_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
THIS_YEAR = "2026"

TITLE = "OpenStreetMap volunteers have mapped 74 ghost bikes, 218 tree stumps and 68 hopscotch grids in NYC"
LINKEDIN_TITLE = "Mappers invent their own tags, including\nghost bikes, tree stumps and hopscotch"
TAKEAWAY = ("Mappers invent their own tags, and coverage follows them: "
            "an empty borough usually means nobody has mapped it yet.")
METHOD = ("Every OpenStreetMap feature in the five boroughs carrying one of ten tags; "
          "areas are drawn at their center")


def finding(key: str, n: int, b: dict, year: dict) -> str:
    """One literally-true sentence per tag, filled from the counts (so it can't drift from the data)."""
    pct = lambda k: round(100 * k / n)
    # the sentences below make claims about the data; stop if a fresh extract breaks one
    if key == "poop_bags":
        assert b["Bronx"] == 0, "poop bags now mapped in the Bronx: reword"
    if key == "ghost_bike":
        assert min(b.values()) > 0 and b["Brooklyn"] == b["Manhattan"], "ghost bike counts changed: reword"
    if key == "memorial_bench":
        assert b["Staten Island"] + b["Manhattan"] == n, "memorial benches elsewhere now: reword"
    return {
        "ghost_bike": f"{n} ghost bikes, in all five boroughs; Brooklyn and Manhattan have {b['Brooklyn']} each",
        "alpr": f"{n:,} license-plate readers; {year.get(THIS_YEAR, 0):,} were added or last edited in {THIS_YEAR}",
        "fire_alarm_box": f"{b['Queens']} of the {n} mapped fire alarm boxes are in Queens",
        "playground_map": f"{b['Queens']} of the {n} painted playground maps are in Queens",
        "hopscotch": f"Staten Island has {b['Staten Island']} of the {n} mapped hopscotch grids",
        "tree_stump": f"{b['Queens']} of the {n} mapped tree stumps are in Queens",
        "bookcase": f"Brooklyn has {b['Brooklyn']} of the {n} mapped little free libraries ({pct(b['Brooklyn'])}%)",
        "memorial_bench": f"{n} memorial benches, all on Staten Island ({b['Staten Island']}) or in Manhattan ({b['Manhattan']})",
        "poop_bags": f"{n} poop-bag dispensers, and none mapped in the Bronx",
        "closed_shop": f"{n:,} shops marked as closed, but only {b['Bronx']} of them in the Bronx",
    }[key]


def load():
    pts = gpd.read_file(DATA / "folk_tags.geojson").to_crs(CRS_M)
    pts["edited"] = pts.edited.astype(str).str[:10]   # read_file parses dates; keep them as YYYY-MM-DD text
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)
    boros["geometry"] = boros.geometry.simplify(40)
    return pts, boros


def summarize(pts):
    views = []
    for f in FOLK:
        sel = pts[pts.folk == f["key"]]
        boro = {b: int((sel.boro == b).sum()) for b in BORO_ORDER}
        year = sel.edited.str[:4].value_counts().to_dict()
        views.append({**{k: f[k] for k in ("key", "label", "one", "tag", "about")}, "n": len(sel), "boro": boro,
                      "finding": finding(f["key"], len(sel), boro, year), "sel": sel})
    return views


DOT = S.C["accent"]   # series blue


def draw_panel(ax, v, land, boros, size_small, size_large, fs):
    """One small map: land, borough lines, the tag's points, and two heading lines above it.
    fs = (heading, tag) font sizes; text offsets are in points so they hold at any figure size."""
    land.plot(ax=ax, color=S.C["ctl"], edgecolor="none")
    boros.boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"] * 0.6)
    g = v["sel"].geometry
    ax.scatter(g.x, g.y, s=size_small if v["n"] < 400 else size_large, color=DOT, edgecolor=S.BG,
               linewidth=0.2, zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_anchor("NW")   # map sits right under its heading, left-aligned with it
    lines = [(f"{v['label']}  {v['n']:,}", fs[0], dict(fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)),
             (v["tag"], fs[1], dict(fontfamily="monospace", color=S.MUTED))]
    y = 3   # points above the axes top, stacked bottom-up
    for text, size, kw in reversed(lines):
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, y), textcoords="offset points",
                    fontsize=size, va="bottom", ha="left", **kw)
        y += size * 1.35


def png(views, boros):
    fig, axes = plt.subplots(2, 5, figsize=S.MAP["png_size_in"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.04, wspace=0.04, hspace=0.36)
    land = boros.dissolve()
    for ax, v in zip(axes.flat, views):
        draw_panel(ax, v, land, boros, 14, 5, (S.SIZE["panel_title"], S.SIZE["subtitle"]))
    S.title(fig, TITLE)
    fig.text(0.5, 0.918, TAKEAWAY, ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, f"{METHOD}  ·  {SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_folk_tags.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


def linkedin(views, boros):
    """4:5 portrait (1080x1350), the largest shape the LinkedIn feed shows: the ten maps in a 3x4 grid,
    with the link in the two spare cells, type sized to read on a phone."""
    fig = plt.figure(figsize=(5.4, 6.75))
    land = boros.dissolve()
    cols, rows = 3, 4
    left, right, top, bottom = 0.04, 0.98, 0.855, 0.065
    cw, ch = (right - left) / cols, (top - bottom) / rows
    head = 0.045   # room for the two heading lines above each map, in figure height
    for i, v in enumerate(views):
        r, c = divmod(i, cols)
        ax = fig.add_axes([left + c * cw, top - (r + 1) * ch + 0.006, cw * 0.94, ch - head - 0.006])
        draw_panel(ax, v, land, boros, 5, 1.6, (7.5, 5.5))
    # spare cells (last row, columns 2-3): where to explore it
    fig.text(left + cw, bottom + ch * 0.45, "Hover and click every dot:\n"
             "daniellecarrick.github.io/30-day-map-challenge/day21",
             ha="left", va="center", fontsize=8, color=S.MUTED, linespacing=1.35)
    # two-line headline, shrunk until the longer line fits 94% of the width
    h = fig.text(0.5, 0.972, LINKEDIN_TITLE, ha="center", va="top", fontsize=15.5, fontweight="bold",
                 fontfamily=S.HEAD_FONT, color=S.INK, linespacing=1.18)
    renderer = fig.canvas.get_renderer()
    while h.get_window_extent(renderer).width > 0.94 * fig.bbox.width:
        h.set_fontsize(h.get_fontsize() - 0.25)
    fig.text(0.5, 0.89, "Ten odd OpenStreetMap tags in New York City, one map each",
             ha="center", va="center", fontsize=8.5, color=S.MUTED)
    fig.text(0.5, 0.012, f"Every OSM feature in the five boroughs with one of ten tags; areas drawn at their center.\n"
                         f"{SOURCE}\n#30DayMapChallenge {DAY}", ha="center", va="bottom", fontsize=5.5,
             color=S.MUTED, linespacing=1.4)
    fig.savefig(OUT / "nyc_folk_tags_linkedin.png", dpi=200)
    plt.close(fig)


def page(views, boros):
    def props(row):
        p = {k: row[k] for k in SHOW if k in row and isinstance(row[k], str) and row[k]}
        return {"t": p, "osm": row["osm"], "boro": row["boro"], "edited": row["edited"]}

    data = {
        "title": TITLE, "day": DAY, "source": SOURCE, "source_url": SOURCE_URL, "method": METHOD,
        "boros_order": BORO_ORDER,
        "boroughs": json.loads(boros[["boroname", "geometry"]].to_json(drop_id=True)),
        "views": [{
            **{k: v[k] for k in ("key", "label", "one", "tag", "about", "n", "boro", "finding")},
            "xy": [[round(p.x), round(p.y)] for p in v["sel"].geometry],
            "props": [props(r) for _, r in v["sel"].drop(columns="geometry").iterrows()],
        } for v in views],
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "nyc_folk_tags.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data) + ";")   # for dev from disk
    print("wrote out/nyc_folk_tags.html", f"({size / 1e6:.1f} MB)")


def main():
    S.apply()
    pts, boros = load()
    views = summarize(pts)
    for v in views:
        print(f"{v['label']:24} {v['n']:>5}  {v['finding']}")
    png(views, boros)
    print("wrote out/nyc_folk_tags.png")
    linkedin(views, boros)
    print("wrote out/nyc_folk_tags_linkedin.png")
    page(views, boros)


if __name__ == "__main__":
    main()
