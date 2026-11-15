"""Day 14 · Borgesian map: places that were on the map but not in the world.

Phantom islands, lost lands, mapmaking mistakes and hoaxes, each drawn where maps or
legends put it, plus the 27 places people have said Atlantis really was.

Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained; timeline of when each place appeared)

Data: data/places.csv and data/atlantis.csv (see data/SOURCES.md; rebuild them with
parse_phantoms.py then combine_places.py). Land: Natural Earth (via Day 13).
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 14: Borgesian map"
CRS = "EPSG:8857"  # Equal Earth
SOURCE = "Wikipedia (phantom islands, cartographic errors, Atlantis hypotheses), Wikidata, Natural Earth"
SOURCE_URL = "https://en.wikipedia.org/wiki/Phantom_island#List_of_phantom_islands"
HOW = ("Each place drawn where old maps, legends or its inventor put it; many positions are approximate. "
       "Dated by its first report or map.")

# Categorical, in tokens.json order.
CATS = [
    ("phantom", "Phantom islands"),
    ("lost", "Lost lands"),
    ("mistake", "Mapmaking mistakes"),
    ("hoax", "Hoaxes and paper towns"),
]
COLOR = {k: S.CATEGORICAL[i] for i, (k, _) in enumerate(CATS)}
LABEL = dict(CATS)

HEADLINE_PLACE = "Sandy Island"


def load():
    p = pd.read_csv(DATA / "places.csv")
    a = pd.read_csv(DATA / "atlantis.csv")
    land = gpd.read_file(DATA / "countries.geojson")
    land = gpd.GeoSeries([shapely.make_valid(land.geometry.union_all())], crs=land.crs)
    return p, a, land


def headline(p):
    s = p.set_index("name").loc[HEADLINE_PLACE]
    years = int(s.removed_year - s.year)
    assert s.removed_year == 2012, "headline assumes Sandy Island left the maps in 2012"
    return s, years, f"Sandy Island sat on maps for {years} years. Then a ship sailed through it."


# ---------------------------------------------------------------- PNG
def make_png(p, a, land, title):
    S.apply(palette="vintage")
    V = S.VINTAGE["color"]
    to = Transformer.from_crs(4326, CRS, always_xy=True).transform
    px, py = to(p.lon.values, p.lat.values)
    ax_, ay_ = to(a.lon.values, a.lat.values)
    atl = p.set_index("name").loc["Atlantis"]
    plx, ply = to(atl.lon, atl.lat)

    fig = plt.figure(figsize=S.MAP["png_size_in"], facecolor=S.BG)
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.83])
    ax.set_axis_off()

    # sphere outline + graticule, like an old world chart
    lons = np.linspace(-180, 180, 361)
    edge = np.r_[np.c_[np.full(181, -180), np.linspace(-90, 90, 181)],
                 np.c_[np.full(181, 180), np.linspace(90, -90, 181)]]
    ex, ey = to(edge[:, 0], edge[:, 1])
    ax.fill(ex, ey, color=S.BG, ec=S.INK, lw=0.8, zorder=0)
    for lat in range(-60, 61, 30):
        gx, gy = to(lons, np.full_like(lons, lat))
        ax.plot(gx, gy, color=S.MUTED, lw=0.25, alpha=0.5, zorder=1)
    for lon in range(-150, 151, 30):
        lats = np.linspace(-90, 90, 181)
        gx, gy = to(np.full_like(lats, lon), lats)
        ax.plot(gx, gy, color=S.MUTED, lw=0.25, alpha=0.5, zorder=1)

    land.to_crs(CRS).simplify(5000).plot(ax=ax, color=V["land"], edgecolor=S.INK, linewidth=0.35, zorder=2)

    # Atlantis: a thin line from Plato's spot to every place it has since been "found"
    for x, y in zip(ax_, ay_):
        ax.plot([plx, x], [ply, y], color=COLOR["lost"], lw=0.5, alpha=0.75, zorder=3)
    ax.scatter(ax_, ay_, s=10, marker="o", fc=S.BG, ec=COLOR["lost"], lw=0.8, zorder=4)

    for k, _ in CATS:
        m = (p.category == k).values
        ax.scatter(px[m], py[m], s=26, c=COLOR[k], ec=S.INK, lw=0.35, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(ex.min() * 1.005, ex.max() * 1.005)
    ax.set_ylim(ey.min() * 1.01, ey.max() * 1.01)

    # ---- callouts: the headline place, Atlantis, the longest-lived invented range, the paper town that came true
    def at(name):
        r = p.set_index("name").loc[name]
        return to(r.lon, r.lat), r
    (sx, sy), sandy = at(HEADLINE_PLACE)
    S.callout(ax, (sx, sy), (0.985, 0.25), "Sandy Island, Coral Sea",
              f"reported in {sandy.year:.0f}; in {sandy.removed_year:.0f} a research ship found only open ocean",
              ha="right", halo=True)
    S.callout(ax, (plx, ply), (0.495, 0.60), f"Atlantis: Plato's spot, and {len(a)} places it has been \"found\" since",
              f"from Sweden ({a.set_index('name').loc['Gamla Uppsala, Sweden'].year:.0f}) to Antarctica and the North Pole",
              ha="left", halo=True)
    (kx, ky), kong = at("Mountains of Kong")
    S.callout(ax, (kx, ky), (0.40, 0.33), "Mountains of Kong",
              f"a mountain range on maps of Africa {kong.year:.0f}–{kong.removed_year:.0f}", ha="left", halo=True)
    (gx_, gy_), agloe = at("Agloe")
    S.callout(ax, (gx_, gy_), (0.16, 0.71), "Agloe, New York",
              "a fake town that became real when a store took its name", ha="left", halo=True)

    # ---- legend, over empty Siberia
    counts = p.category.value_counts()
    hand = [Line2D([], [], ls="", marker="o", ms=6.5, mfc=COLOR[k], mec=S.INK, mew=0.4,
                   label=f"{lab}  ({counts.get(k, 0)})") for k, lab in CATS]
    hand.append(Line2D([0, 1], [0, 0], color=COLOR["lost"], lw=0.8, marker="o", ms=4, mfc=S.BG,
                       mec=COLOR["lost"], markevery=[1], label=f"Proposed Atlantis sites  ({len(a)})"))
    lg = ax.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.625, 0.95), frameon=False,
                   fontsize=S.SIZE["legend"] + 1, handlelength=1.6, borderaxespad=0, labelspacing=0.55,
                   title=f"{len(p)} places that weren't there", title_fontproperties={"family": S.HEAD_FONT,
                   "weight": "semibold", "size": S.SIZE["panel_title"]})
    lg._legend_box.align = "left"

    S.title(fig, title, y=0.975)
    fig.text(0.5, 0.915, "A MAP OF PLACES THAT WERE ON THE MAP BUT NOT IN THE WORLD", ha="center", va="center",
             fontsize=S.SIZE["subtitle"], color=S.MUTED, fontweight="semibold")
    S.credit(fig, f"{HOW}  Data: {SOURCE}  ·  #30DayMapChallenge {DAY}", y=0.016)
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"], facecolor=S.BG)
    plt.close(fig)


# ---------------------------------------------------------------- web
def clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def short(note, n=230):
    if not isinstance(note, str):
        return None
    note = note.strip()
    if len(note) <= n:
        return note
    cut = note[:n].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def web_data(p, a, land, title, sandy_years):
    g = land.simplify(0.12).set_precision(0.01)
    places = []
    for r in p.itertuples():
        approx = isinstance(r.coord_note, str) and r.coord_note.startswith("hand-approx")
        places.append(dict(
            name=r.name, cat=r.category, lon=round(r.lon, 2), lat=round(r.lat, 2),
            year=clean(r.year), prec=clean(r.year_precision), removed=clean(r.removed_year),
            removed_note=short(r.removed_note, 120) if r.category != "phantom" else None,
            note=short(r.note), approx=approx,
            wiki=r.article if isinstance(r.article, str) and r.article else None,
        ))
    sites = [dict(name=r.name, by=clean(r.proposed_by), year=clean(r.year), note=clean(r.note),
                  lon=r.lon, lat=r.lat) for r in a.itertuples()]
    atl = p.set_index("name").loc["Atlantis"]
    return dict(
        title=title, day=DAY, how=HOW, source=SOURCE, source_url=SOURCE_URL,
        lede=(f"Borges imagined a map so exact it was as large as the empire it showed. Real maps went the other way: "
              f"for centuries they showed islands, mountains, lakes and towns that were never there. "
              f"Here are {len(p)} of them, drawn where the maps put them."),
        land=json.loads(gpd.GeoDataFrame(geometry=g).to_json(drop_id=True)),
        cats=[dict(key=k, label=lab, n=int((p.category == k).sum())) for k, lab in CATS],
        places=places, sites=sites, plato=[atl.lon, atl.lat], sandy_years=sandy_years,
        undated=int(p.year.isna().sum()),
    )


def main():
    p, a, land = load()
    sandy, years, title = headline(p)
    print(title)
    print(p.category.value_counts().to_dict(), "| Atlantis sites", len(a))
    make_png(p, a, land, title)
    data = web_data(p, a, land, title, years)
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
