"""Day 5 · Senses: Sight — How much of the sky can you see from every NYC block?

From the middle of each block, the share of the sky dome not hidden by buildings (sky.py). Writes:
  - out/nyc_sky.png    (static: the city, a Manhattan zoom, three sky circles)
  - out/nyc_sky.html   (interactive: hover any block to look straight up from it)

Data: NYC Open Data street centerlines (inkn-q76z) and building footprints (5zhs-2jue).
Run:  ./fetch.sh, then ../.venv/bin/python build.py   (first run ~3 min; cached in data/raw/)
(The earlier "street henge" version of this day is build_henge.py + web/henge.html.)
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402
import henge as H  # noqa: E402
import sky as K  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 5: Senses: Sight"
SOURCE = "NYC Open Data (street centerlines, building footprints)"
SOURCE_URL = "https://data.cityofnewyork.us/d/5zhs-2jue"
BORO = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}
HOW = ("Horizon in 360 directions from the middle of each block, eye height, against every building's roof. "
       "Share of the sky dome's area left open. Trees aren't counted.")

# Sequential: less sky = darker. The darkest canyons get the darkest blue.
BLUES = S.SEQUENTIAL_BLUE
STOPS = [(i / (len(BLUES) - 1), BLUES[-1 - i]) for i in range(len(BLUES))]
STOPS_DARK = [(i / (len(BLUES) - 2), BLUES[-2 - i]) for i in range(len(BLUES) - 1)]   # darkest blue vanishes on a dark page
color_for = S.make_scale(STOPS)
HIDDEN, OPEN_SKY = BLUES[-1], BLUES[0]           # sky circles: disc = buildings, shape = sky
ABBR = [(r"\bAv\b", "Ave"), (r"\bBl\b", "Blvd")]


def nice(s) -> str:
    """'W 42 ST' -> 'W 42nd St'."""
    if not isinstance(s, str) or not s.strip():
        return "Unnamed street"
    s = re.sub(r"\s+", " ", s.strip()).title()
    for pat, rep in ABBR:
        s = re.sub(pat, rep, s)

    def ordinal(m):
        n = int(m.group(1))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    return re.sub(r"\b(\d+)\b(?= (St|Ave|Rd|Dr|Pl|Ter|Blvd|Ln|Way)\b)", ordinal, s)


def pct(v: float) -> str:
    return "<1%" if v < 0.01 else f"{v:.0%}"


def median_block(g):
    """The block whose sky share is closest to the group's median."""
    return g.iloc[(g.sky - g.sky.median()).abs().argmin()]


def sky_polygon(bins):
    """Stepped outline of a sky circle: each 10 degree bin an arc at radius r (north up, clockwise)."""
    pts = []
    for k, r in enumerate(bins):
        for a in np.radians(np.linspace(k * K.BIN, (k + 1) * K.BIN, 5)):
            pts.append((r * np.sin(a), r * np.cos(a)))
    return pts


def draw_circle(fig, rect, block, heading):
    ax = fig.add_axes(rect)
    ax.add_patch(Circle((0, 0), 1, color=HIDDEN, lw=0))
    ax.add_patch(Polygon(sky_polygon(block.bins), closed=True, color=OPEN_SKY, lw=0))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.set_axis_off()
    ax.text(0, -1.14, heading, ha="center", va="top", fontsize=S.SIZE["annotation"], fontweight="semibold")
    ax.text(0, -1.40, f"{pct(block.sky)} of the sky", ha="center", va="top", fontsize=S.SIZE["annotation"], color=S.MUTED)


def draw_png(s, st):
    S.apply()
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    d = s.sort_values("sky", ascending=False)      # dark canyons on top
    segs = [np.asarray(g.coords) for g in d.geometry]
    cols = [color_for(v) for v in d.sky]
    x0, y0, x1, y1 = s.total_bounds
    ax = fig.add_axes([0.01, 0.05, 0.60, 0.80])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.45, capstyle="round"))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()
    zx0, zy0 = H.to_m.transform(-74.035, 40.70); zx1, zy1 = H.to_m.transform(-73.945, 40.80)
    az = fig.add_axes([0.62, 0.05, 0.37, 0.80])
    az.add_collection(LineCollection(segs, colors=cols, linewidths=1.2, capstyle="round"))
    az.set_xlim(zx0, zx1); az.set_ylim(zy0, zy1); az.set_aspect("equal"); az.set_axis_off()

    S.title(fig, st["title"])
    fig.text(0.5, 0.925, st["subtitle"], ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, f"{HOW}\nData: {SOURCE}  ·  #30DayMapChallenge {DAY}", y=0.008)

    # legend + sky circles, in the empty space over New Jersey
    lx, ly = 0.03, 0.80
    fig.text(lx, ly + 0.024, "Share of the sky you can see", fontsize=S.SIZE["legend"], fontweight="semibold")
    lax = fig.add_axes([lx, ly, 0.2, 0.014])
    lax.imshow([[to_rgb(color_for(v)) for v in np.linspace(0, 1, 256)]], aspect="auto", extent=[0, 1, 0, 1])
    lax.set_yticks([]); lax.set_xticks([0, 0.5, 1]); lax.set_xticklabels(["0%", "50%", "100%"], fontsize=S.SIZE["legend"] - 1)
    lax.tick_params(length=3, color=S.MUTED, pad=2)
    for sp in lax.spines.values():
        sp.set_visible(False)
    fig.text(lx, 0.735, "Looking straight up: the center is overhead, the rim is the horizon",
             fontsize=S.SIZE["annotation"], color=S.MUTED)
    size = 0.066          # fig width; height follows from the figure's aspect
    w, h = S.MAP["png_size_in"]
    for k, (block, heading) in enumerate(st["circles"]):
        draw_circle(fig, [lx + k * 0.088, 0.615, size, size * w / h], block, heading)

    for c in st["callouts"]:
        S.callout(az if c["zoom"] else ax, c["xy"], c["text_xy"], c["heading"], c["detail"], ha=c.get("ha", "left"), halo=True)
    fig.savefig(OUT / "nyc_sky.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


def web_data(s, st):
    x0, y0 = np.floor(s.total_bounds[:2])
    h = s.reset_index(drop=True)
    names = sorted(set(zip(h.street, h.boro)))
    ix = {n: i for i, n in enumerate(names)}

    def delta(g):
        """[x, y, dx, dy, ...] in whole metres: far smaller than absolute coordinates."""
        xy = np.round(shapely.get_coordinates(shapely.simplify(g, 3)) - [x0, y0]).astype(int)
        return np.r_[xy[0], np.diff(xy, axis=0).ravel()].tolist()

    # one character per sky-circle value: r 0..1 in 64 steps, starting at '0'
    codes = np.round(np.stack(h.bins.values) * 63).astype(int) + 48
    bins = "".join(map(chr, codes.ravel()))
    zx0, zy0 = H.to_m.transform(-74.02, 40.698); zx1, zy1 = H.to_m.transform(-73.77, 40.823)   # opening view
    return {
        "title": st["title"], "lede": st["lede"], "how": HOW, "day": DAY, "source": SOURCE, "source_url": SOURCE_URL,
        "extent": [float(s.total_bounds[2] - x0), float(s.total_bounds[3] - y0)],
        "view": [zx0 - x0, zy0 - y0, zx1 - x0, zy1 - y0],
        "c": [delta(g) for g in h.geometry], "n": [ix[k] for k in zip(h.street, h.boro)],
        "sky": [int(round(v * 1000)) for v in h.sky], "bins": bins, "nbins": codes.shape[1],
        "names": [list(n) for n in names], "stops": STOPS, "stops_dark": STOPS_DARK,
        "hidden": HIDDEN, "open": OPEN_SKY,
        "refs": [{"i": int(h.index[h.physicalkey == b.physicalkey][0]), "label": lab} for b, lab in st["circles"]],
    }


def main():
    s = K.compute()
    s = s[~s.inside].reset_index(drop=True)
    s["street"] = s.name.map(nice)
    s["boro"] = s.boroughcode.map(BORO)
    s["physicalkey"] = s.index
    med = s.groupby("boro").sky.median()
    man = s[s.boro == "Manhattan"]
    wall, w42 = man[man.street == "Wall St"], man[man.street == "W 42nd St"]
    # deepest street, among named streets at least 300 m long (single short alleys are at the
    # limit of what a 5 m building grid can resolve, so no block is singled out)
    by = s.assign(L=s.geometry.length).groupby(["street", "boro"]).agg(sky=("sky", "median"), L=("L", "sum")).reset_index()
    deep = by[by.L >= 300].nsmallest(1, "sky").iloc[0]
    deep_g = s[(s.street == deep.street) & (s.boro == deep.boro)]
    si = s[s.boro == "Staten Island"]
    wall_m = wall.sky.median()

    st = {
        "title": f"On Wall Street you can see {wall_m:.0%} of the sky. On a typical Staten Island block, {med['Staten Island']:.0%}.",
        "subtitle": "How much of the sky isn't hidden by buildings, from the middle of every New York City block",
        "lede": ("From the middle of every block in New York City, how much of the sky isn't hidden by buildings? "
                 "Hover or tap a block to look straight up from it."),
        "circles": [(median_block(wall), "Wall St"), (median_block(w42), "W 42nd St"), (median_block(si), "Staten Island")],
        "callouts": [
            {"zoom": True, "xy": (median_block(deep_g).mx, median_block(deep_g).my), "text_xy": (0.02, 0.31),
             "heading": f"{deep.street}: {pct(deep.sky)}", "detail": "Deepest street\nover 300 m long"},
            {"zoom": True, "xy": (median_block(wall).mx, median_block(wall).my), "text_xy": (0.02, 0.17),
             "heading": f"Wall St: {pct(wall_m)}", "detail": None},
            {"zoom": True, "xy": (median_block(w42).mx, median_block(w42).my), "text_xy": (0.02, 0.72),
             "heading": f"W 42nd St: {pct(w42.sky.median())}", "detail": None},
            {"zoom": False, "xy": (median_block(si).mx, median_block(si).my), "text_xy": (0.03, 0.43),
             "heading": f"Staten Island: {pct(med['Staten Island'])}", "detail": "The most sky of any borough"},
        ],
    }
    assert med.idxmax() == "Staten Island", "callout says Staten Island has the most sky"
    print(med.round(3).to_dict(), f"Wall St {wall_m:.3f}; deepest {deep.street} {deep.sky:.3f}; W 42nd {w42.sky.median():.3f}")
    print(f"Manhattan blocks under half the sky: {(man.sky < .5).mean():.2f}; under a quarter: {(man.sky < .25).mean():.2f}")
    draw_png(s, st)
    size = S.build_page(HERE / "web" / "index.html", web_data(s, st), OUT / "nyc_sky.html")
    print("wrote out/nyc_sky.png and out/nyc_sky.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
