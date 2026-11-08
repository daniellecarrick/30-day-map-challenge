"""Day 9 · Urban-rural: New York City vs. the rest of New York State, as flowers.

One flower per county on a tile grid. Urban traits point up (blue), rural traits point down (amber);
petal AREA = county share / highest county share. Center dot = 2024 presidential margin.
The two big flowers (New York City = 5 boroughs combined, Other counties = the other 57) double as the key.

Writes:
  - out/urban-rural.png   (static, for posting)
  - out/urban-rural.html  (interactive, self-contained)
  - out/urban-rural_linkedin_{1,2}.png + urban-rural_linkedin.pdf  (1080x1350 slides for a LinkedIn document post)

Data: US Census ACS 2019-2023 5-year; 2020 US Religion Census; certified 2024 county results (NYS Board of Elections).
Run:  sh fetch.sh && ../.venv/bin/python prep.py && ../.venv/bin/python build.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 9: Urban-rural"
TITLE = "In New York City, 55% of households have no car. In the rest of the state, just 9% do."
LEDE = ("Each flower is a county. Urban traits point up in blue, rural traits down in amber, "
        "and one that barely differs, households earning $200k or more, in green.")
METHOD = "Petal area = share ÷ highest county's share. NYC = the 5 boroughs combined"
FAITH_NOTE = ("*Adherents of Jewish, Muslim, Hindu, Buddhist, Sikh and other non-Christian congregations, "
              "counted where the congregation meets")
SOURCE = "Data: US Census ACS 2019–2023; 2020 US Religion Census; 2024 results: NYS Board of Elections"
LINKS = {
    "acs": "https://www.census.gov/programs-surveys/acs/data/summary-file.html",
    "usrc": "https://www.usreligioncensus.org/node/1637",
    "votes": "https://results.elections.ny.gov/contest/5591",
}

URBAN, RURAL, EVEN = S.CATEGORICAL[0], S.CATEGORICAL[3], S.CATEGORICAL[2]   # blue, amber, green (web: --cat-1, --cat-4, --cat-3)
SIDE_COLOR = {"urban": URBAN, "rural": RURAL, "even": EVEN}
# (key, label, angle in degrees, side)
PETALS = [
    ("nocar",   "have no car",          150, "urban"),
    ("commute", "commute an hour or more", 110, "urban"),
    ("faith",   "non-Christian faith*",  70, "urban"),
    ("foreign", "born abroad",           30, "urban"),
    ("rich",    "earn \\$200k or more",  210, "even"),   # about equal in NYC and elsewhere: its own color
    ("own",     "own their home",        250, "rural"),
    ("veteran", "are veterans",          290, "rural"),
    ("mfg",     "work in manufacturing", 330, "rural"),
]
KEYS = [p[0] for p in PETALS]
WEB_LABEL = {"rich": "earn $200k or more", "mfg": "manufacturing jobs", "commute": "commute 1 hr or more"}   # fit beside the big flowers at 14px
VOTE_STOPS = S.diverging_stops(0, 0.5)   # Trump minus Harris share: blue = Harris, brown = Trump
VOTE = S.make_scale(VOTE_STOPS)
COLS, ROWS = 12, 9
CALLED_OUT = {"36087", "36061"}   # Rockland, Manhattan
BOROUGH = {"New York": "Manhattan", "Kings": "Brooklyn", "Richmond": "Staten Island"}   # tile labels people know


def load():
    d = pd.read_csv(DATA / "counties.csv", dtype={"fips": str}).set_index("fips")
    shp = gpd.read_file(f"zip://{DATA / 'counties.zip'}")
    shp = shp[shp.STATEFP == "36"].set_index("GEOID").to_crs("EPSG:32618")
    d["name"] = shp.NAME
    c = shp.geometry.representative_point()
    d["x"], d["y"] = c.x, c.y
    sums = d.groupby("nyc")[[f"{k}_{s}" for k in KEYS for s in "nd"] + ["gop_n", "dem_n", "vote_d"]].sum()
    for frame in (d, sums):
        for k in KEYS:
            frame[k] = frame[f"{k}_n"] / frame[f"{k}_d"]
        frame["margin"] = (frame.gop_n - frame.dem_n) / frame.vote_d
    return d, sums, d[KEYS].max()


def tile_grid(d):
    """Assign each county to one grid cell, keeping it as near its real position as possible."""
    x = (d.x - d.x.min()) / (d.x.max() - d.x.min()) * (COLS - 1)
    y = (d.y - d.y.min()) / (d.y.max() - d.y.min()) * (ROWS - 1)
    cells = np.array([(c, r) for r in range(ROWS) for c in range(COLS)])
    cost = ((np.c_[x, y][:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    i, j = linear_sum_assignment(cost)
    d = d.copy()
    d.loc[d.index[i], "gx"] = cells[j, 0]
    d.loc[d.index[i], "gy"] = cells[j, 1]
    return d


def petal(cx, cy, angle, length, width=0.5):
    """Teardrop petal from (cx, cy) outward; width scales with length so area ~ length^2.
    Same curve as petalPath() in web/index.html."""
    a = np.radians(angle)
    u, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    w = 0.55 * width * length
    p0 = np.array([cx, cy])
    pts = [p0, p0 + u * 0.25 * length + n * w, p0 + u * 0.85 * length + n * w, p0 + u * length,
           p0 + u * 0.85 * length - n * w, p0 + u * 0.25 * length - n * w, p0]
    return MPath(pts, [MPath.MOVETO] + [MPath.CURVE4] * 6)


def petal_len(row, key, maxes, R):
    return R * np.sqrt(row[key] / maxes[key])


def flower(ax, cx, cy, R, row, maxes, lw=0.5):
    for key, _, ang, side in PETALS:
        col = SIDE_COLOR[side]
        ax.add_patch(PathPatch(petal(cx, cy, ang, R), fc=S.C["ctl"], ec="none", zorder=2))
        L = petal_len(row, key, maxes, R)
        if L > 0.02 * R:
            ax.add_patch(PathPatch(petal(cx, cy, ang, L), fc=col, ec=S.C["bg"], lw=lw, zorder=3))
    ax.add_patch(plt.Circle((cx, cy), R * 0.13, fc=VOTE(row["margin"]), ec=S.INK, lw=lw * 0.8, zorder=4))


def spread(ys, gap, iters=60):
    """Push label centers apart (in place order) until neighbors are at least `gap` apart."""
    ys = list(ys)
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    for _ in range(iters):
        moved = False
        for a, b in zip(order, order[1:]):
            d = ys[b] - ys[a]
            if d < gap:
                ys[a] -= (gap - d) / 2; ys[b] += (gap - d) / 2; moved = True
        if not moved:
            break
    return ys


def hero_labels(row, maxes, R, gap):
    """Label anchor per petal, beside the tip of the COLORED petal (not the grey ghost), relative to the center.
    Labels on each side of the flower are then spread vertically so they never overlap."""
    out = []
    for key, lab, ang, side in PETALS:
        a = np.radians(ang)
        L = petal_len(row, key, maxes, R)
        r = max(L, 0.3 * R) + 0.07 * R
        right = np.cos(a) > 0
        x = r * np.cos(a) + (1 if right else -1) * (0.22 * L + 0.02 * R)   # clear the petal's bulge
        out.append(dict(key=key, label=lab, x=x, y=r * np.sin(a), right=right))
    for right in (True, False):
        grp = [o for o in out if o["right"] == right]
        for o, y in zip(grp, spread([o["y"] for o in grp], gap)):
            o["y"] = y
    return out


def hero(ax, cx, cy, R, row, maxes, heading, fs=S.SIZE["legend"], gap=None):
    """Big flower that is also the key: each petal labeled with its name and value, beside its colored tip.
    `gap`: minimum vertical distance between label centers, in data units (default suits the main PNG)."""
    flower(ax, cx, cy, R, row, maxes, lw=1)
    ax.text(cx - 1.75 * R, cy + 1.25 * R, heading, fontsize=S.SIZE["panel_title"], fontweight="semibold",
            fontfamily=S.HEAD_FONT, va="bottom")
    for o in hero_labels(row, maxes, R, gap=gap or 0.4 * R):
        ha = "left" if o["right"] else "right"
        x, y = cx + o["x"], cy + o["y"]
        ax.text(x, y, f"{row[o['key']]:.0%}", ha=ha, va="bottom", fontsize=fs + 2, fontweight="bold")
        ax.text(x, y, o["label"], ha=ha, va="top", fontsize=fs, color=S.MUTED)


def county_grid(ax, d, maxes, name_fs=6.2, y_min=-1.1, names=None):
    """Every county's flower on the tile grid; NYC tiles boxed, called-out tiles (Rockland, Manhattan) boxed heavier."""
    ax.set_xlim(-0.6, COLS - 0.4); ax.set_ylim(y_min, ROWS - 0.4); ax.set_aspect("equal"); ax.set_axis_off()
    names = names or BOROUGH
    for f, r in d.iterrows():
        if r.nyc or f in CALLED_OUT:
            ax.add_patch(FancyBboxPatch((r.gx - 0.47, r.gy - 0.47), 0.94, 0.94,
                                        boxstyle="round,pad=0,rounding_size=0.06", fc="none", ec=S.INK,
                                        lw=1.6 if f in CALLED_OUT else 0.8, zorder=6 if f in CALLED_OUT else 1))
        flower(ax, r.gx, r.gy + 0.04, 0.36, r, maxes)
        ax.text(r.gx, r.gy - 0.44, names.get(r["name"], r["name"]), ha="center", va="bottom",
                fontsize=name_fs, color=S.MUTED, zorder=5)


def frac(ax, x, y):
    """Data -> axes fraction, for S.callout's label position."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)


def vote_key(ax, x, y, fs=S.SIZE["legend"], step=0.15, r=0.055):
    """Five dots, Harris to Trump, with a caption above: the center-dot legend."""
    for i, m in enumerate([-0.5, -0.25, 0, 0.25, 0.5]):
        ax.add_patch(plt.Circle((x + (i - 2) * step, y), r, fc=VOTE(m), ec=S.INK, lw=0.5))
    ax.text(x - 2 * step - 0.12, y, "Harris", ha="right", va="center", fontsize=fs)
    ax.text(x + 2 * step + 0.12, y, "Trump", ha="left", va="center", fontsize=fs)
    ax.text(x, y + r + 0.05, "Center dot: 2024 presidential margin", ha="center", va="bottom", fontsize=fs, color=S.MUTED)


KEY_LABEL = {"nocar": "no car", "commute": "long commute", "faith": "non-Christian*", "foreign": "born abroad",
             "rich": "\\$200k+ income", "own": "own home", "veteran": "veterans", "mfg": "manufacturing"}
LINK = "daniellecarrick.github.io/30-day-map-challenge/day09"


def linkedin(d, sums, maxes):
    """Two 4:5 portrait slides (1080x1350) for a LinkedIn document post, type sized for a phone:
    1) the New York City and Other counties flowers, 2) the county map. Saved as two PNGs and one 2-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    S.apply()
    plt.rcParams["pdf.fonttype"] = 42          # embed real TrueType fonts in the PDF
    W, H = 5.4, 6.75
    import textwrap
    foot = "\n".join(["Petal area = share ÷ highest county's share. NYC = the 5 boroughs combined.",
                      *textwrap.wrap(FAITH_NOTE + ".", 100), SOURCE, f"#30DayMapChallenge {DAY}  ·  {LINK}"])
    figs = []

    # ---- slide 1: the two big flowers ----
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 0.975, "55% of New York City households have no car.\nIn the rest of the state, just 9% do.",
             ha="center", va="top", fontsize=14.5, linespacing=1.25, fontweight="bold", color=S.INK,
             fontfamily=S.HEAD_FONT)
    fig.text(0.5, 0.875, "Urban traits point up in blue, rural traits down in amber,\n"
             "and one that barely differs, \\$200k+ incomes, in green.", ha="center", va="top",
             fontsize=9.5, linespacing=1.35, color=S.MUTED)
    ax = fig.add_axes([0, 0.16, 1, 0.67])
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-4.6, 0.36); ax.set_aspect("equal"); ax.set_axis_off()
    hero(ax, 0.05, -1.0, 0.78, sums.loc[True], maxes, "New York City", fs=9.5, gap=0.36)
    hero(ax, 0.05, -3.35, 0.78, sums.loc[False], maxes, "Other counties", fs=9.5, gap=0.36)
    vax = fig.add_axes([0, 0.105, 1, 0.05]); vax.set_xlim(-2.6, 2.6); vax.set_ylim(-0.12, 0.3)
    vax.set_aspect("equal"); vax.set_axis_off()
    vote_key(vax, 0, 0, fs=9)
    fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=6.5, color=S.MUTED, linespacing=1.4)
    figs.append(fig)

    # ---- slide 2: the county map ----
    fig = plt.figure(figsize=(W, H))
    fig.text(0.5, 0.975, "County by county, the city blooms up\nand the rest of the state hangs down",
             ha="center", va="top", fontsize=14.5, linespacing=1.25, fontweight="bold", color=S.INK,
             fontfamily=S.HEAD_FONT)
    fig.text(0.5, 0.875, "Each flower is a county, placed roughly where it sits on the map.", ha="center",
             va="top", fontsize=9.5, color=S.MUTED)
    ax = fig.add_axes([0.0, 0.2, 1.0, 0.66])
    county_grid(ax, d, maxes, name_fs=5.2, y_min=-1.7, names={**BOROUGH, "Staten Island": "Staten Is.", "Richmond": "Staten Is."})
    # petal key in the empty top-left: every petal at full size, named
    kx, ky, kR = 2.3, 7.45, 0.5
    for key, lab, ang, side in PETALS:
        a = np.radians(ang)
        ax.add_patch(PathPatch(petal(kx, ky, ang, kR), fc=SIDE_COLOR[side], ec=S.C["bg"], lw=0.6, zorder=3))
        r = kR + 0.08
        ax.text(kx + r * np.cos(a), ky + r * np.sin(a), KEY_LABEL[key], fontsize=7,
                ha="left" if np.cos(a) > 0 else "right", va="center", color=S.MUTED)
    rk, mn = d.loc["36087"], d.loc["36061"]
    S.callout(ax, (rk.gx - 0.47, rk.gy + 0.1), frac(ax, -0.3, 1.0),
              f"Rockland: {rk.faith:.0%} in non-Christian congregations", "most in the state, mostly Orthodox Jewish")
    S.callout(ax, (mn.gx, mn.gy - 0.47), frac(ax, 2.6, -1.2),
              f"Manhattan: {mn.nocar:.0%} of households have no car", "most in the state")
    vax = fig.add_axes([0, 0.125, 1, 0.05]); vax.set_xlim(-2.6, 2.6); vax.set_ylim(-0.12, 0.3)
    vax.set_aspect("equal"); vax.set_axis_off()
    vote_key(vax, 0, 0, fs=9)
    fig.text(0.5, 0.012, foot, ha="center", va="bottom", fontsize=6.5, color=S.MUTED, linespacing=1.4)
    figs.append(fig)

    with PdfPages(OUT / "urban-rural_linkedin.pdf") as pdf:
        for i, f in enumerate(figs, 1):
            f.savefig(OUT / f"urban-rural_linkedin_{i}.png", dpi=200)
            pdf.savefig(f)
            plt.close(f)


def main():
    S.apply()
    d, sums, maxes = load()
    d = tile_grid(d)
    nyc, other = sums.loc[True], sums.loc[False]
    assert round(nyc.nocar * 100) == 55 and round(other.nocar * 100) == 9, "headline numbers changed"

    # ---- static PNG ----
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    S.title(fig, TITLE)
    fig.text(0.5, 0.925, LEDE, ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)

    hx = fig.add_axes([0.02, 0.07, 0.3, 0.83])
    hx.set_xlim(-1.9, 1.9); hx.set_ylim(-3.15, 2.65); hx.set_aspect("equal"); hx.set_axis_off()
    hero(hx, 0, 1.15, 0.9, nyc, maxes, "New York City")
    hero(hx, 0, -1.55, 0.9, other, maxes, "Other counties")
    vote_key(hx, 0, -3.0)

    ax = fig.add_axes([0.32, 0.06, 0.67, 0.84])
    county_grid(ax, d, maxes)
    rk = d.loc["36087"]   # Rockland: most non-Christian adherents
    S.callout(ax, (rk.gx - 0.47, rk.gy + 0.1), frac(ax, 1.4, rk.gy + 0.1),
              f"Rockland: {rk.faith:.0%} belong to non-Christian congregations",
              "the most in the state, mostly Orthodox Jewish")
    mn = d.loc["36061"]   # Manhattan: most car-free
    S.callout(ax, (mn.gx, mn.gy - 0.47), frac(ax, 3.6, -0.75),
              f"Manhattan: {mn.nocar:.0%} of households have no car", "the most in the state")

    S.credit(fig, f"{METHOD}. {FAITH_NOTE}.", y=0.03)
    S.credit(fig, f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "urban-rural.png", dpi=S.MAP["png_dpi"])

    # ---- interactive page ----
    def rec(row):
        out = {k: round(float(row[k]), 4) for k in KEYS}
        out["margin"] = round(float(row["margin"]), 4)
        return out

    counties = [dict(fips=f, name=r["name"], nyc=bool(r.nyc), gx=int(r.gx), gy=int(r.gy),
                     pop=int(r["pop"]), **rec(r)) for f, r in d.iterrows()]
    data = {
        "title": TITLE, "lede": LEDE, "method": METHOD, "faith_note": FAITH_NOTE, "day": DAY, "links": LINKS,
        "petals": [dict(key=k, label=WEB_LABEL.get(k, l), angle=a, side=s) for k, l, a, s in PETALS],
        "max": {k: round(float(maxes[k]), 4) for k in KEYS},
        "nyc": rec(nyc), "other": rec(other), "counties": counties,
        "cols": COLS, "rows": ROWS, "vote_stops": VOTE_STOPS,
    }
    linkedin(d, sums, maxes)
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "urban-rural.html")
    print("wrote out/urban-rural.png and out/urban-rural.html", f"({size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
