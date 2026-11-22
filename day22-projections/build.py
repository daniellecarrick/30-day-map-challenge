"""Day 22 · Projections — the same election drawn four ways: states, counties, states sized by
population (a contiguous cartogram), and one equal hexagon per state.

The point: a map's look depends on what it gives area to. Land votes Republican; people are split.

Geometry: lower 48 in Albers equal-area (EPSG:5070); Alaska and Hawaii moved into insets.
Cartogram: Dougenik, Chrisman & Niemeyer (1985) rubber-sheet algorithm on the lower 48, so states
keep their neighbours; Alaska and Hawaii are scaled to the same people-per-area afterwards.
Alaska reports results by state house district, not borough, so the county view shows it as one unit.

Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained)

Data: county results: github.com/tonmcg/US_County_Level_Election_Results_08-24 (from official sources)
      population: US Census Bureau Vintage 2024 state estimates; boundaries: Census cartographic 1:20m
Run:  ../.venv/bin/python build.py   (./fetch.sh first)
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, RegularPolygon
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 22: Projections"
SOURCE = "Data: county results compiled by Tony McGovern from official sources; US Census Bureau population and boundaries"
SOURCE_URL = "https://github.com/tonmcg/US_County_Level_Election_Results_08-24"
PNG_YEAR = 2020

YEARS = {
    2020: dict(votes="votes2020.csv", counties="cb_2020_us_county_20m.zip", pop="POPESTIMATE2020",
               dem="Biden", gop="Trump"),
    2024: dict(votes="votes2024.csv", counties="cb_2023_us_county_20m.zip", pop="POPESTIMATE2024",
               dem="Harris", gop="Trump"),
}

# Margin colors: every feature shows its winner (no grey middle), darker = bigger margin.
P = S.TOKENS["scales"]["party"]
MARGIN_STOPS = [(-0.6, P["dem"][2]), (-0.3, P["dem"][1]), (-1e-6, P["dem"][0]),
                (1e-6, P["gop"][0]), (0.3, P["gop"][1]), (0.6, P["gop"][2])]
margin_color = S.make_scale(MARGIN_STOPS)

HI_CRS = "+proj=aea +lat_1=19 +lat_2=22 +lat_0=20.5 +lon_0=-157 +datum=NAD83 +units=m +no_defs"
INSETS = {"02": ("EPSG:3338", 0.36, -1_850_000, -150_000), "15": (HI_CRS, 1.0, -720_000, 520_000)}

# One hexagon per state, pointy-top, odd rows shifted half a hex right: (row, col).
HEX = {
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 8), "NH": (1, 9),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "WI": (2, 5), "MI": (2, 6),
    "NY": (2, 8), "MA": (2, 9),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4), "IL": (3, 5), "IN": (3, 6),
    "OH": (3, 7), "PA": (3, 8), "CT": (3, 9), "RI": (3, 10),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4), "KY": (4, 5), "WV": (4, 6),
    "VA": (4, 7), "MD": (4, 8), "NJ": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5), "NC": (5, 6), "SC": (5, 7),
    "DC": (5, 8), "DE": (5, 9),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 7),
}


# ---------------------------------------------------------------- geometry
def to_layout(gdf: gpd.GeoDataFrame, key="STATEFP") -> gpd.GeoSeries:
    """Lower 48 in EPSG:5070, Alaska and Hawaii moved into insets. Returns geometry in metres."""
    parts = []
    for fips, g in gdf.groupby(key):
        if fips in INSETS:
            crs, k, dx, dy = INSETS[fips]
            parts.append(g.geometry.to_crs(crs).affine_transform([k, 0, 0, k, dx, dy]).set_crs(None, allow_override=True))
        else:
            parts.append(g.geometry.to_crs("EPSG:5070").set_crs(None, allow_override=True))
    return gpd.GeoSeries(pd.concat(parts)).reindex(gdf.index)


def true_area(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Land area in m² in an equal-area projection (no inset shrinking)."""
    return gdf.geometry.to_crs("EPSG:6933").area


def rings(geom):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    return [np.asarray(p.exterior.coords)[:-1] for p in polys]


def from_rings(rs):
    polys = [Polygon(r) for r in rs]
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def cartogram(geoms: list, values: np.ndarray, iters=60, tol=1.01):
    """Dougenik et al. (1985) contiguous cartogram. geoms: list of shapely polygons (no holes)."""
    R = [rings(g) for g in geoms]
    sizes = [[len(r) for r in rs] for rs in R]
    pts = np.concatenate([np.concatenate(rs) for rs in R])
    values = np.asarray(values, float)

    def rebuild(pts):
        out, i = [], 0
        for sz in sizes:
            rs = []
            for n in sz:
                rs.append(pts[i:i + n]); i += n
            out.append(from_rings(rs))
        return out

    for it in range(iters):
        polys = rebuild(pts)
        A = np.array([p.area for p in polys])
        C = np.array([[p.centroid.x, p.centroid.y] for p in polys])
        desired = A.sum() * values / values.sum()
        radius = np.sqrt(A / np.pi)
        mass = np.sqrt(desired / np.pi) - radius
        err = np.maximum(A, desired) / np.minimum(A, desired)
        if err.mean() < tol:
            break
        fr = 1 / (1 + err.mean())
        d = pts[:, None, :] - C[None, :, :]            # (N, regions, 2)
        dist = np.hypot(d[..., 0], d[..., 1]) + 1e-9
        ratio = dist / radius
        force = np.where(ratio > 1, mass * radius / dist, mass * ratio ** 2 * (4 - 3 * ratio))
        pts = pts + fr * ((force / dist)[..., None] * d).sum(axis=1)
    print(f"  cartogram: {it} iterations, mean size error {err.mean():.3f}")
    return rebuild(pts)


def carea(s):
    return gpd.GeoSeries(list(s.carto), index=s.index).area


def scale_to(geom, area):
    from shapely import affinity
    k = np.sqrt(area / geom.area)
    return affinity.scale(geom, k, k, origin=geom.centroid)


# ---------------------------------------------------------------- data
def load_states():
    st = gpd.read_file(DATA / "cb_2020_us_state_20m.zip")
    st = st[st.STATEFP.astype(int) <= 56].set_index("STATEFP").sort_index()
    st["land"] = true_area(st)
    st = gpd.GeoDataFrame(st.drop(columns="geometry"), geometry=to_layout(st.reset_index(), "STATEFP").values, index=st.index)
    pop = pd.read_csv(DATA / "pop2024.csv", dtype={"STATE": str})
    pop = pop[pop.SUMLEV == 40].set_index("STATE")
    for yr, y in YEARS.items():
        st[f"pop{yr}"] = pop.loc[st.index, y["pop"]].astype(int)
    return st


def load_votes(yr):
    v = pd.read_csv(DATA / YEARS[yr]["votes"], dtype={"county_fips": str})
    v["fips"] = v.county_fips.where(~v.county_fips.str.startswith("11"), "11001")  # DC wards -> DC
    v["fips"] = v.fips.where(~v.fips.str.startswith("02"), "02000")                # AK districts -> AK
    v = v.groupby("fips")[["votes_gop", "votes_dem", "total_votes"]].sum()
    return v


def load_counties(yr, st):
    c = gpd.read_file(DATA / YEARS[yr]["counties"])
    c = c[c.STATEFP.isin(st.index)].copy()
    ak = c[c.STATEFP == "02"].dissolve().assign(GEOID="02000", NAME="Alaska", STATEFP="02")
    c = pd.concat([c[c.STATEFP != "02"], ak[c.columns]], ignore_index=True)
    c = gpd.GeoDataFrame(c, geometry="geometry", crs=ak.crs)
    c["land"] = true_area(c)
    c = gpd.GeoDataFrame(c.drop(columns="geometry"), geometry=to_layout(c, "STATEFP").values, index=c.index)
    v = load_votes(yr)
    c = c.join(v, on="GEOID")
    c["abbr"] = c.STATEFP.map(st.STUSPS)
    return c


def summarize(gop, dem, total):
    return dict(gop=int(gop), dem=int(dem), total=int(total), margin=float((gop - dem) / total))


# ---------------------------------------------------------------- build
def main():
    S.apply()
    st = load_states()
    lower48 = [f for f in st.index if f not in INSETS]

    years = {}
    for yr in YEARS:
        c = load_counties(yr, st)
        sv = c.groupby("STATEFP")[["votes_gop", "votes_dem", "total_votes"]].sum()
        s = st.join(sv)
        s["margin"] = (s.votes_gop - s.votes_dem) / s.total_votes
        c["margin"] = (c.votes_gop - c.votes_dem) / c.total_votes
        # population cartogram: rubber-sheet the lower 48, then scale AK and HI to the same density
        carto = cartogram(list(s.loc[lower48].geometry), s.loc[lower48, f"pop{yr}"].values)
        cg = pd.Series(carto, index=lower48)
        dens = sum(g.area for g in carto) / s.loc[lower48, f"pop{yr}"].sum()
        for f in INSETS:
            cg[f] = scale_to(s.loc[f, "geometry"], s.loc[f, f"pop{yr}"] * dens)
        s["carto"] = cg.reindex(s.index)
        years[yr] = dict(states=s, counties=c)

        tot = s[["votes_gop", "votes_dem", "total_votes"]].sum()
        red_s = s.margin > 0
        red_c = c.margin > 0
        voted = c.margin.notna()
        cand = YEARS[yr]
        print(f"{yr}: {cand['gop']} {tot.votes_gop / tot.total_votes:.2%}  {cand['dem']} {tot.votes_dem / tot.total_votes:.2%}")
        print(f"  land red: states {s.land[red_s].sum() / s.land.sum():.1%}, counties {c.land[red_c].sum() / c.land[voted].sum():.1%}")
        print(f"  pop in red states {s[f'pop{yr}'][red_s].sum() / s[f'pop{yr}'].sum():.1%}, red states {red_s.sum()}/51")
        print(f"  drawn ink red: states {s.area[red_s].sum() / s.area.sum():.1%}, "
              f"counties {c.area[red_c].sum() / c.area[voted].sum():.1%}, "
              f"carto {carea(s)[red_s].sum() / carea(s).sum():.1%}")
        print(f"  red counties {red_c.sum()} / {voted.sum()}")

    # ---- one frame (viewBox) for the geographic map and both cartograms
    W, H, PAD = 960, 600, 8
    allg = gpd.GeoSeries(list(st.geometry) + [g for y in years.values() for g in y["states"].carto])
    x0, y0, x1, y1 = allg.total_bounds
    k = min((W - 2 * PAD) / (x1 - x0), (H - 2 * PAD) / (y1 - y0))
    ox, oy = (W - (x1 - x0) * k) / 2, (H - (y1 - y0) * k) / 2
    tx = lambda p: [round(ox + (p[0] - x0) * k, 1), round(oy + (y1 - p[1]) * k, 1)]  # noqa: E731
    vb = lambda g: [[tx(p) for p in r] for r in rings(g)]  # noqa: E731

    def svg_d(g, tol):
        g = g.simplify(tol / k, preserve_topology=True)
        polys = g.geoms if isinstance(g, MultiPolygon) else [g]
        out = []
        for p in polys:
            pts = [tx(q) for q in np.asarray(p.exterior.coords)[:-1]]
            if len(pts) >= 3:
                out.append("M" + "L".join(f"{a:g},{b:g}" for a, b in pts) + "Z")
        return "".join(out)

    # hex centers (pointy-top) in the same viewBox, centered
    r = 46
    hw = np.sqrt(3) * r
    hc = {a: ((c + 0.5 * (row % 2)) * hw, row * 1.5 * r) for a, (row, c) in HEX.items()}
    hx = np.array(list(hc.values()))
    hdx = (W - (hx[:, 0].max() - hx[:, 0].min())) / 2 - hx[:, 0].min()
    hdy = (H - (hx[:, 1].max() - hx[:, 1].min())) / 2 - hx[:, 1].min()
    hc = {a: (round(x + hdx, 1), round(y + hdy, 1)) for a, (x, y) in hc.items()}

    # ---- web data
    state_rows = []
    for f, row in st.iterrows():
        state_rows.append(dict(fips=f, abbr=row.STUSPS, name=row.NAME, land=round(row.land / 2.59e6),
                               geo=vb(row.geometry), hex=hc[row.STUSPS]))
    year_data, shapes = {}, {}
    for yr, y in years.items():
        s, c = y["states"], y["counties"]
        voted = c.margin.notna()
        tot = s[["votes_gop", "votes_dem", "total_votes"]].sum()
        red = s.margin > 0
        ink = {
            "states": s.area[red].sum() / s.area.sum(),
            "counties": c.area[voted & (c.margin > 0)].sum() / c.area[voted].sum(),
            "carto": carea(s)[red].sum() / carea(s).sum(),
            "hex": red.mean(),
        }
        year_data[yr] = dict(
            dem=YEARS[yr]["dem"], gop=YEARS[yr]["gop"],
            total=summarize(tot.votes_gop, tot.votes_dem, tot.total_votes),
            ink={kk: round(float(v), 4) for kk, v in ink.items()},
            states={f: dict(**summarize(q.votes_gop, q.votes_dem, q.total_votes), pop=int(q[f"pop{yr}"]),
                            carto=vb(q.carto)) for f, q in s.iterrows()},
            counties={q.GEOID: [int(q.votes_gop), int(q.votes_dem), int(q.total_votes)] if pd.notna(q.margin) else None
                      for _, q in c.iterrows()},
        )
        for _, q in c.iterrows():
            name = q.NAME if q.GEOID == "02000" else q.NAMELSAD
            shapes.setdefault(q.GEOID, dict(name=name, st=q.abbr, land=round(q.land / 2.59e6), d=svg_d(q.geometry, 0.35)))
    data = dict(W=W, H=H, hexR=r, title=TITLE, day=DAY, source=SOURCE, sourceUrl=SOURCE_URL,
                stops=MARGIN_STOPS, states=state_rows, counties=shapes, years=year_data)
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print(f"wrote out/map.html ({size / 1e6:.1f} MB)")

    png(years[PNG_YEAR], st, hc, W, H, r, tx, k)
    linkedin(years[PNG_YEAR], hc, r)


TITLE = "The same 2020 votes can look 81% red or 43% red, depending on how you draw the map"


def draw_maps(axes, s, c, voted, hc, r, ew, hexfs):
    """The four maps into a 2x2 array of axes: states, counties, population cartogram, hexagons."""
    ax = axes[0, 0]
    gpd.GeoSeries(s.geometry).plot(ax=ax, color=[margin_color(m) for m in s.margin], edgecolor=S.EDGE, linewidth=ew * 2)
    ax = axes[0, 1]
    c[voted].plot(ax=ax, color=[margin_color(m) for m in c.margin[voted]], edgecolor=S.EDGE, linewidth=ew * 0.4)
    c[~voted].plot(ax=ax, color=S.CLOSED, edgecolor=S.EDGE, linewidth=ew * 0.4)
    gpd.GeoSeries(s.geometry).boundary.plot(ax=ax, color=S.EDGE, linewidth=ew * 2)
    ax = axes[1, 0]
    gpd.GeoSeries(list(s.carto)).plot(ax=ax, color=[margin_color(m) for m in s.margin], edgecolor=S.EDGE, linewidth=ew * 2)
    # common extent for the three geographic panels
    allb = np.array([gpd.GeoSeries(list(s.geometry)).total_bounds, gpd.GeoSeries(list(s.carto)).total_bounds])
    x0, y0 = allb[:, :2].min(0)
    x1, y1 = allb[:, 2:].max(0)
    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        a.set_xlim(x0, x1); a.set_ylim(y0 - (y1 - y0) * 0.02, y1 + (y1 - y0) * 0.02)

    ax = axes[1, 1]
    for f, row in s.iterrows():
        x, yy = hc[row.STUSPS]
        ax.add_patch(RegularPolygon((x, -yy), 6, radius=r * 0.97, orientation=0, fc=margin_color(row.margin),
                                    ec=S.EDGE, lw=1))
        dark = abs(row.margin) > 0.2
        ax.text(x, -yy, row.STUSPS, ha="center", va="center", fontsize=hexfs,
                fontweight="semibold", color="white" if dark else S.INK)
    hx = np.array(list(hc.values()))
    ax.set_xlim(hx[:, 0].min() - r * 1.2, hx[:, 0].max() + r * 1.2)
    ax.set_ylim(-hx[:, 1].max() - r * 1.2, -hx[:, 1].min() + r * 1.2)



def png(y, st, hc, W, H, r, tx, k):
    s, c = y["states"], y["counties"]
    yr = PNG_YEAR
    voted = c.margin.notna()
    fig, axes = plt.subplots(2, 2, figsize=S.MAP["png_size_in"])
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.04, wspace=0.06, hspace=0.2)
    ew = S.MAP["cell_edge_width"]
    red = s.margin > 0

    def pct(v):
        return f"{v:.0%}"

    panels = [
        ("States", f"{pct(s.area[red].sum() / s.area.sum())} of the map is red"),
        ("Counties", f"{pct(c.area[voted & (c.margin > 0)].sum() / c.area[voted].sum())} of the map is red"),
        ("States sized by population", f"{pct(carea(s)[red].sum() / carea(s).sum())} of the map is red"),
        ("One hexagon per state", f"{red.sum()} of 51 hexagons are red"),
    ]
    for ax, (head, sub) in zip(axes.flat, panels):
        ax.set_axis_off()
        ax.set_aspect("equal")
        # heading + finding above the axes (the maps fill the axes, so there is no room inside)
        ax.annotate(head, (0, 1), xycoords="axes fraction", xytext=(0, 15), textcoords="offset points",
                    fontsize=S.SIZE["panel_title"], fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 1), textcoords="offset points",
                    fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)

    draw_maps(axes, s, c, voted, hc, r, ew, S.SIZE["annotation"])

    # callouts: the biggest gap between land and people
    ak = s.loc["02"]
    S.callout(axes[0, 0], (ak.geometry.centroid.x, ak.geometry.centroid.y), (0.47, 0.03),
              "Alaska", f"{ak.land / s.land.sum():.0%} of the land, {ak.total_votes / s.total_votes.sum():.1%} of the votes",
              outline=ak.geometry)
    ca = s.loc["06"]
    S.callout(axes[1, 0], (ca.carto.centroid.x, ca.carto.centroid.y), (0.0, 0.2),
              "California", f"{ca[f'pop{yr}'] / s[f'pop{yr}'].sum():.0%} of Americans live here",
              outline=ca.carto)

    # legend: margin scale, top center under the title
    lax = fig.add_axes([0.38, 0.875, 0.24, 0.016])
    xs = np.linspace(-0.6, 0.6, 241)
    lax.imshow([[np.array(plt.matplotlib.colors.to_rgb(margin_color(v))) for v in xs]], aspect="auto",
               extent=(-0.6, 0.6, 0, 1))
    lax.set_yticks([])
    lax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    lax.set_xticklabels([f"{YEARS[yr]['dem']} +60", "+30", "0", "+30", f"{YEARS[yr]['gop']} +60"],
                        fontsize=S.SIZE["legend"])
    lax.tick_params(length=0, pad=3)
    for sp in lax.spines.values():
        sp.set_visible(False)
    tv = s[["votes_gop", "votes_dem", "total_votes"]].sum()
    fig.text(0.5, 0.912, f"Same votes, four maps. {yr} presidential margin, percentage points. Nationwide: "
             f"{YEARS[yr]['dem']} {tv.votes_dem / tv.total_votes:.1%}, {YEARS[yr]['gop']} {tv.votes_gop / tv.total_votes:.1%}.",
             ha="center", fontsize=S.SIZE["subtitle"], color=S.MUTED)

    S.title(fig, TITLE)
    S.credit(fig, f"Area shares from Albers equal-area maps (Alaska and Hawaii as drawn); cartogram after Dougenik et al. 1985. "
                  f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])
    print("wrote out/map.png")


LINKEDIN_TITLE = "The same 2020 votes look 81% red\nor 43% red, depending on the map"


def linkedin(y, hc, r):
    """4:5 portrait (1080x1350) for the LinkedIn feed: the four maps in a 2x2 grid, type sized for a phone."""
    s, c = y["states"], y["counties"]
    yr = PNG_YEAR
    voted = c.margin.notna()
    red = s.margin > 0
    fig = plt.figure(figsize=(5.4, 6.75))
    left, right, top, bottom, gapx = 0.035, 0.975, 0.785, 0.09, 0.03
    cw = (right - left - gapx) / 2
    head = 0.075   # heading + two finding lines above each map, in figure height
    ch = (top - bottom) / 2
    axes = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            ax = fig.add_axes([left + j * (cw + gapx), top - (i + 1) * ch, cw, ch - head])
            ax.set_axis_off(); ax.set_aspect("equal"); ax.set_anchor("N")
            axes[i, j] = ax
    gop = s.votes_gop.sum() / s.total_votes.sum()
    shares = [s.area[red].sum() / s.area.sum(),
              c.area[voted & (c.margin > 0)].sum() / c.area[voted].sum(),
              carea(s)[red].sum() / carea(s).sum(),
              red.sum() / 51]
    heads = ["States", "Counties", "States sized by population", "One hexagon per state"]
    for ax, h, sh in zip(axes.flat, heads, shares):
        gap = round((sh - gop) * 100)
        what = f"{red.sum()} of 51 hexagons are red" if h.startswith("One") else f"{sh:.0%} of the map is red"
        vs = "as red as the vote" if gap == 0 else f"{abs(gap)} pts {'redder' if gap > 0 else 'bluer'} than the vote"
        ax.annotate(h, (0, 1), xycoords="axes fraction", xytext=(0, 24), textcoords="offset points",
                    fontsize=9.5, fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)
        ax.annotate(f"{what}\n{vs}", (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=6.5, color=S.MUTED, linespacing=1.3)
    draw_maps(axes, s, c, voted, hc, r, S.MAP["cell_edge_width"] * 0.6, 4.2)

    h = fig.text(0.5, 0.965, LINKEDIN_TITLE, ha="center", va="top", fontsize=16, fontweight="bold",
                 fontfamily=S.HEAD_FONT, color=S.INK, linespacing=1.18)
    renderer = fig.canvas.get_renderer()
    while h.get_window_extent(renderer).width > 0.94 * fig.bbox.width:
        h.set_fontsize(h.get_fontsize() - 0.25)
    tv = s[["votes_gop", "votes_dem", "total_votes"]].sum()
    fig.text(0.5, 0.865, f"{YEARS[yr]['dem']} won {tv.votes_dem / tv.total_votes:.1%} of the vote. "
             f"{YEARS[yr]['gop']} won {tv.votes_gop / tv.total_votes:.1%}. Same votes, four maps.",
             ha="center", va="center", fontsize=8, color=S.MUTED)
    lax = fig.add_axes([0.3, 0.818, 0.4, 0.013])
    xs = np.linspace(-0.6, 0.6, 241)
    lax.imshow([[np.array(plt.matplotlib.colors.to_rgb(margin_color(v))) for v in xs]], aspect="auto",
               extent=(-0.6, 0.6, 0, 1))
    lax.set_yticks([])
    lax.set_xticks([-0.6, 0, 0.6])
    lax.set_xticklabels([f"{YEARS[yr]['dem']} +60", "0", f"{YEARS[yr]['gop']} +60"], fontsize=6.5, color=S.INK)
    lax.tick_params(length=0, pad=2)
    for sp in lax.spines.values():
        sp.set_visible(False)
    fig.text(0.5, 0.012, "Explore 2020 and 2024: daniellecarrick.github.io/30-day-map-challenge/day22\n"
             f"{SOURCE}\n#30DayMapChallenge {DAY}", ha="center", va="bottom", fontsize=5.5,
             color=S.MUTED, linespacing=1.4)
    fig.savefig(OUT / "map_linkedin.png", dpi=200)
    plt.close(fig)
    print("wrote out/map_linkedin.png")


if __name__ == "__main__":
    main()
