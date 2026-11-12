"""Day 8 · Utopia: America's utopian communities, sized by how long each one lasted.

Religious utopias typically lasted half a century; secular ones a few years.
Writes:
  - out/utopias.png    (static, for posting)
  - out/utopias.html   (interactive, self-contained: map + timeline)
  - out/utopias_linkedin.png (1080x1350 portrait for the LinkedIn feed; --linkedin-only skips the rest)

Data: Wikipedia lists of American utopian, Owenite, Fourierist and Shaker communities
      (fetch.py -> parse.py -> clean.py -> data/communities.csv); state outlines: US Census
      cartographic boundary file, 2023, 1:20m.
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 8: Utopia"
NOW = 2026
CRS_M = "EPSG:5070"  # Conterminous US Albers, metres
LOWER48_SKIP = {"AK", "HI", "PR"}
WIKI = "https://en.wikipedia.org/wiki/"
SOURCES = [
    ("utopian", "List_of_American_utopian_communities"),
    ("Owenite", "List_of_Owenite_communities_in_the_United_States"),
    ("Fourierist", "List_of_Fourierist_Associations_in_the_United_States"),
    ("Shaker", "Shaker_communities"),
]
CENSUS = "https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"
METHOD = ("Lifespan runs from the founding year to the year the community stopped as a communal "
          "experiment; ones still running are counted to 2026. Medians leave out 10 with unknown dates (grey)")

COLORS = {"religious": S.CATEGORICAL[0], "secular": S.CATEGORICAL[1]}
SIZE_KEYS = [10, 50, 100, 200]
R_MIN, R_MAX = 2.4, 17.0        # web radius (viewBox px) for 0 and MAX_YEARS years
MAX_YEARS = 232


def load():
    d = pd.read_csv(DATA / "communities.csv")
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326").to_crs(CRS_M)
    g["x"], g["y"] = g.geometry.x, g.geometry.y
    states = gpd.read_file(DATA / "raw" / "states" / "states.zip")
    states = states[~states.STUSPS.isin(LOWER48_SKIP)].to_crs(CRS_M)
    return g, states


def stats(g):
    k = g[g.years.notna()]
    med = k.groupby("basis").years.median()
    return {
        "n": len(g), "n_known": len(k),
        "religious": int((g.basis == "religious").sum()), "secular": int((g.basis == "secular").sum()),
        "med_religious": float(med["religious"]), "med_secular": float(med["secular"]),
        "share20_religious": round(100 * (k[k.basis == "religious"].years >= 20).mean()),
        "share20_secular": round(100 * (k[k.basis == "secular"].years >= 20).mean()),
        "first": int(g.start.min()), "last": int(g.start.max()),
    }


def headline(st):
    # Checked against the numbers: "over 50" and "3" must stay literally true.
    assert st["med_religious"] > 50 and st["med_secular"] == 3, st
    return "America's religious utopias typically lasted over 50 years. Secular ones lasted 3."


def png(g, states, st, title):
    S.apply()
    fig, ax = plt.subplots(figsize=S.MAP["png_size_in"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07)
    states.plot(ax=ax, facecolor="#f0efec", edgecolor="white", linewidth=0.8, zorder=1)
    states.dissolve().boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"], zorder=2)

    # marker size in points^2; area ∝ years, same mapping as the web radius
    def size(y):
        r = R_MIN + (R_MAX - R_MIN) * np.sqrt(np.clip(y, 0, None) / MAX_YEARS)
        return (r * 0.75) ** 2 * 1.9

    known = g[g.years.notna()].sort_values("years", ascending=False)
    unknown = g[g.years.isna()]
    ax.scatter(unknown.x, unknown.y, s=size(0), color=S.CLOSED, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(known.x, known.y, s=size(known.years), color=known.basis.map(COLORS), alpha=0.82,
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")
    x0, y0, x1, y1 = states.total_bounds
    ax.set_xlim(x0 - 150e3, x1 + 60e3)
    ax.set_ylim(y0 - 60e3, y1 + 60e3)

    # ---- callouts: the longest-lived of each kind, and the 1840s Fourierist boom ----
    def ring(row, pad=1.15):
        from shapely.geometry import Point
        r_pts = np.sqrt(size(row.years if pd.notna(row.years) else 0)) / 2 * pad
        m_per_pt = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (fig.get_size_inches()[0] * 0.98 * 72)
        return Point(row.x, row.y).buffer(r_pts * m_per_pt)

    sab = g[g.name == "Sabbathday Lake Shaker Village"].iloc[0]
    S.callout(ax, (sab.x, sab.y), (0.86, 0.93), "Sabbathday Lake, Maine: 232 years and counting",
              "Founded 1794, the last Shaker village still running", ha="right", outline=ring(sab))
    fair = g[g.name.str.startswith("Fairhope")].iloc[0]
    S.callout(ax, (fair.x, fair.y), (0.62, 0.06), "Fairhope, Alabama: the longest-lived secular one",
              "A single-tax colony since 1894, still running", outline=ring(fair))
    four = g[g.movement == "Fourierist"]
    fk = four[four.years.notna()]
    from shapely.ops import unary_union
    ax.scatter(four.x, four.y, s=size(four.years.fillna(0)), facecolor="none", edgecolor=S.INK,
               linewidth=0.7, zorder=5)
    ohio = four[four.place.str.contains("OH|Ohio")]
    tp = (ohio.x.median(), ohio.y.max())
    n40 = int(four.start.between(1840, 1849).sum())
    S.callout(ax, tp, (0.44, 0.86), f"The Fourierist boom: {n40} \"phalanxes\" founded in the 1840s (outlined)",
              f"Half of all {len(four)} Fourierist communities lasted {int(fk.years.median())} years or less",
              ha="right")

    # ---- legend, bottom-left over the Pacific ----
    handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=COLORS["religious"], mec="white", label="Religious"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=COLORS["secular"], mec="white", label="Secular"),
               Line2D([], [], ls="", marker="o", ms=5, mfc=S.CLOSED, mec="white", label="Dates unknown")]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.2), fontsize=S.SIZE["legend"] + 1,
                    title="Community", title_fontsize=S.SIZE["legend"] + 1, alignment="left", handletextpad=0.4)
    leg.get_title().set_fontweight("semibold")
    ax.add_artist(leg)
    sh = [Line2D([], [], ls="", marker="o", ms=np.sqrt(size(y)), mfc="none", mec=S.MUTED, label=f"{y} years")
          for y in SIZE_KEYS]
    leg2 = ax.legend(handles=sh, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=S.SIZE["legend"] + 1,
                     title="Dot size: how long it lasted", title_fontsize=S.SIZE["legend"] + 1, alignment="left",
                     ncols=4, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.6)
    leg2.get_title().set_fontweight("semibold")

    S.title(fig, title)
    fig.text(0.5, 0.925, f"{st['n']} American utopian communities founded {st['first']}–{st['last']}, "
             f"sized by how long each one lasted. Median lifespan: religious {st['med_religious']:g} years, "
             f"secular {st['med_secular']:g}.", ha="center", fontsize=S.SIZE["subtitle"] + 2, color=S.MUTED)
    S.credit(fig, f"{METHOD}.", y=0.03)
    S.credit(fig, "Data: Wikipedia lists of American utopian, Owenite, Fourierist and Shaker communities; "
                  f"US Census state outlines  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "utopias.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


def page(g, states, st, title):
    st_geo = states[["geometry"]].copy()
    st_geo["geometry"] = st_geo.geometry.simplify(4000)
    comm = []
    for r in g.itertuples():
        comm.append({
            "name": r.name, "basis": r.basis, "movement": r.movement, "place": r.place,
            "start": None if pd.isna(r.start) else int(r.start),
            "end": None if pd.isna(r.end) else int(r.end),
            "active": bool(r.active), "years": None if pd.isna(r.years) else int(r.years),
            "notes": "" if pd.isna(r.notes) else r.notes,
            "x": round(r.x), "y": round(r.y),
        })
    data = {
        "title": title, "day": DAY, "now": NOW, "maxYears": MAX_YEARS, "rMin": R_MIN, "rMax": R_MAX,
        "sizeKeys": SIZE_KEYS, "stats": st,
        "states": json.loads(st_geo.to_json(drop_id=True)),
        "outline": json.loads(gpd.GeoDataFrame(geometry=[states.union_all().simplify(4000)],
                                               crs=CRS_M).to_json(drop_id=True)),
        "communities": comm,
        "sources": [[label, WIKI + slug] for label, slug in SOURCES], "census": CENSUS, "method": METHOD,
    }
    return S.build_page(HERE / "web" / "index.html", data, OUT / "utopias.html")


def linkedin(g, states, st, title):
    """4:5 portrait (1080x1350), the largest shape the LinkedIn feed shows, with type sized for a phone.
    The two medians up top, then the map, with the longest-lived of each kind called out."""
    S.apply()
    fig = plt.figure(figsize=(5.4, 6.75))
    fig.text(0.5, 0.975, title.replace(" typically lasted ", " typically lasted\n", 1),
             ha="center", va="top", fontsize=14, linespacing=1.25,
             fontweight="bold", color=S.INK, fontfamily=S.HEAD_FONT)

    # ---- the two medians ----
    for cx, basis in ((0.29, "religious"), (0.71, "secular")):
        fig.text(cx, 0.83, f"{st[f'med_{basis}']:g} years", ha="center", fontsize=20, fontweight="bold",
                 fontfamily=S.HEAD_FONT, color=COLORS[basis])
        fig.text(cx, 0.82, f"median lifespan of {st[basis]}\n{basis} communities", ha="center", va="top",
                 fontsize=8.5, color=S.MUTED, linespacing=1.3)

    # ---- map: crop to the lower 48, drawn as wide as the image ----
    ax = fig.add_axes([0.0, 0.2, 1.0, 0.54])
    states.plot(ax=ax, facecolor="#f0efec", edgecolor="white", linewidth=0.4, zorder=1)
    states.dissolve().boundary.plot(ax=ax, color=S.INK, linewidth=0.3, zorder=2)

    def size(y):   # points^2; area ∝ years
        return (1.1 + 7.4 * np.sqrt(np.clip(y, 0, None) / MAX_YEARS)) ** 2

    known = g[g.years.notna()].sort_values("years", ascending=False)
    unknown = g[g.years.isna()]
    ax.scatter(unknown.x, unknown.y, s=size(0), color=S.CLOSED, edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter(known.x, known.y, s=size(known.years), color=known.basis.map(COLORS), alpha=0.85,
               edgecolor="white", linewidth=0.3, zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")
    x0, y0, x1, y1 = states.total_bounds
    ax.set_xlim(x0 - 30e3, x1 + 30e3)

    sab = g[g.name == "Sabbathday Lake Shaker Village"].iloc[0]
    fair = g[g.name.str.startswith("Fairhope")].iloc[0]
    for r in (sab, fair):
        ax.scatter([r.x], [r.y], s=size(r.years) * 1.3, facecolor="none", edgecolor=S.INK, lw=0.8, zorder=5)
    S.callout(ax, (sab.x, sab.y), (0.98, 0.98), "Sabbathday Lake, Maine",
              "Shaker village since 1794,\nstill running: 232 years", ha="right", halo=True)
    S.callout(ax, (fair.x, fair.y), (0.45, 0.05), "Fairhope, Alabama",
              "Longest-lived secular one,\na single-tax colony since 1894", halo=True)

    # ---- legend: color, then dot size ----
    handles = [Line2D([], [], marker="o", ls="", ms=6, mfc=COLORS["religious"], mec="none", label="Religious"),
               Line2D([], [], marker="o", ls="", ms=6, mfc=COLORS["secular"], mec="none", label="Secular"),
               Line2D([], [], marker="o", ls="", ms=3.5, mfc=S.CLOSED, mec="none", label="Dates unknown")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.135), ncol=3, frameon=False, fontsize=9,
               handletextpad=0.3, columnspacing=1.4)
    sh = [Line2D([], [], ls="", marker="o", ms=np.sqrt(size(y)), mfc="none", mec=S.MUTED, label=f"{y}")
          for y in SIZE_KEYS]
    leg = fig.legend(handles=sh, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4, frameon=False,
                     fontsize=9, title="Dot size: years it lasted", title_fontsize=9, handletextpad=0.4,
                     columnspacing=1.2)
    leg.get_title().set_color(S.MUTED)
    S.credit(fig, "Lifespan: founding year to the end of communal living; ones still running counted to 2026.\n"
                  f"Data: Wikipedia lists of American utopian communities  ·  #30DayMapChallenge {DAY}\n"
                  "daniellecarrick.github.io/30-day-map-challenge/day08", y=0.014)
    fig.savefig(OUT / "utopias_linkedin.png", dpi=200)
    plt.close(fig)


def main():
    g, states = load()
    st = stats(g)
    title = headline(st)
    print(json.dumps(st))
    linkedin(g, states, st, title)
    if "--linkedin-only" in sys.argv:
        return
    png(g, states, st, title)
    size = page(g, states, st, title)
    print("wrote out/utopias.png and out/utopias.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
