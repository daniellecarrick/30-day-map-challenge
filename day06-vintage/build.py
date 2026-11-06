"""Day 6 · Vintage: every NYC building colored by the era it was built, drawn like an old
fire insurance atlas.

Writes:
  - out/nyc_building_age.png   (static, for posting)
  - out/nyc_building_age.html  (interactive, self-contained; every building)

Data: NYC Building Footprints (5zhs-2jue), 2020 Neighborhood Tabulation Areas (9nt8-h7nd).
Run:  sh fetch.sh, then ../.venv/bin/python build.py
"""
import base64
import gzip
import json
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 6: Vintage"
CRS_M = "EPSG:32618"
SOURCE = "NYC Building Footprints"
SOURCE_URL = "https://data.cityofnewyork.us/City-Government/BUILDING/5zhs-2jue"
HOW = ("Each building counted once by its listed construction year, garages excluded; "
       "many older years are rounded estimates.")

# 25-year eras that sit on the headline's 1940. 1915 keeps the blur of estimates around 1920 in one era.
ERA_EDGES = [1915, 1940, 1965, 1990, 2015]
ERA_LABELS = ["Before 1915", "1915–1939", "1940–1964", "1965–1989", "1990–2014", "2015 and later"]
ERAS = S.VINTAGE["eras"]
OLDEST, NEWEST = "Crown Heights (South)", "Spring Creek-Starrett City"
SHORT = {OLDEST: "Crown Heights South", NEWEST: "Spring Creek"}
PLATE_M = 1600  # both PNG plates show the same ground distance
# Neighborhood tooltip histogram: 20-year spans, placed so one edge falls on the headline's 1940.
HIST_EDGES = list(range(1880, 2021, 20))
HIST_LABELS = ["Before 1880", *[f"{y}–{y + 19}" for y in HIST_EDGES[:-1]], "2020 and later"]


def load_buildings() -> gpd.GeoDataFrame:
    cache = RAW / "buildings.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    a = pd.read_csv(RAW / "buildings.csv", low_memory=False,
                    usecols=["doitt_id", "feature_code", "construction_year", "name"])
    a = a[(a.feature_code == 2100) & (a.construction_year > 0)]
    g = pd.read_csv(RAW / "buildings_geom.csv")
    m = a.merge(g, on="doitt_id", how="inner")
    geom = shapely.from_wkt(m.the_geom.values)
    gdf = gpd.GeoDataFrame(m.drop(columns="the_geom"), geometry=geom, crs=4326).to_crs(CRS_M)
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    cache.write_bytes(pickle.dumps(gdf))
    return gdf


def neighborhood_stats(b, nta):
    pts = gpd.GeoDataFrame(geometry=shapely.point_on_surface(b.geometry.values), crs=CRS_M)
    j = gpd.sjoin(pts, nta[["ntaname", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    b = b.assign(ntaname=j.ntaname.values)
    res = nta[nta.ntatype.astype(str) == "0"]
    s = b[b.ntaname.isin(res.ntaname)]
    g = s.groupby("ntaname").construction_year
    st = pd.DataFrame(dict(n=g.size(), median=g.median(),
                           pre1940=g.apply(lambda v: (v < 1940).mean() * 100),
                           post2000=g.apply(lambda v: (v >= 2000).mean() * 100)))
    st = st[st.n >= 100]
    return b, st, s


def footprint_packed(b, x0, y1, cell_m, ncol):
    """True footprints as compact columns: spatially sorted, half-metre grid, delta-coded vertices."""
    parts, owner = shapely.get_parts(b.geometry.values, return_index=True)
    assert len(parts) == len(b) and (owner == np.arange(len(b))).all(), "expected one polygon per building"
    # within half a metre of the source: drops redundant points, not corners anyone can see
    s = shapely.simplify(parts, .5, preserve_topology=True)
    s = shapely.transform(s, lambda c: np.c_[c[:, 0] - x0, y1 - c[:, 1]])  # y runs down the page
    box = shapely.bounds(s)
    gx = ((box[:, 0] + box[:, 2]) / 2 // cell_m).astype(int)
    gy = ((box[:, 1] + box[:, 3]) / 2 // cell_m).astype(int)
    cell = gy * ncol + gx
    order = np.lexsort((np.arange(len(cell)), cell))
    ids, counts = np.unique(cell[order], return_counts=True)

    rings, ring_owner = shapely.get_rings(s[order], return_index=True)
    n_rings = np.bincount(ring_owner, minlength=len(s))
    co, ci = shapely.get_coordinates(rings, return_index=True)
    starts = np.r_[0, np.flatnonzero(np.diff(ci)) + 1]
    ends = np.r_[starts[1:], len(ci)]
    keep = np.ones(len(co), bool)
    keep[ends - 1] = False                                   # rings repeat their first point
    q = np.round(co[keep] * 2).astype(np.int64)
    d = np.diff(q, axis=0, prepend=np.zeros((1, 2), np.int64))
    z = (d << 1) ^ (d >> 63)                                 # zigzag: small magnitudes, small bytes
    assert z.max() < 2 ** 24
    verts = [((z[:, k] >> sh) & 255).astype(np.uint8).tobytes() for k in (0, 1) for sh in (0, 8, 16)]
    return dict(order=order, ids=ids, counts=counts, n_rings=n_rings,
                n_verts=ends - starts - 1, verts=verts, n_vert_total=int(keep.sum()))


def shoreline(city):
    """The city's edge, without its land borders with Westchester and Nassau, plus those two masks."""
    to_m = Transformer.from_crs(4326, CRS_M, always_xy=True).transform
    west = shapely.ops.transform(to_m, shapely.Polygon(
        [(-73.914, 40.97), (-73.914, 40.908), (-73.86, 40.90), (-73.822, 40.882), (-73.795, 40.885), (-73.74, 40.93), (-73.74, 40.97)]))
    nass = shapely.ops.transform(to_m, shapely.Polygon(
        [(-73.755, 40.775), (-73.755, 40.72), (-73.752, 40.60), (-73.60, 40.60), (-73.60, 40.80)]))
    return city.boundary.difference(west).difference(nass), west, nass


def water_rings(city):
    shore, west, nass = shoreline(city)
    shore = shore.simplify(15)
    keep = []
    for d in (80, 190, 330, 520):
        ring = shore.buffer(d, quad_segs=4).boundary.difference(city).difference(west).difference(nass)
        keep.append((d, ring.simplify(12)))
    return keep


def lines(geom):
    if geom.is_empty:
        return []
    return [np.asarray(g.coords) for g in shapely.get_parts(geom)]


def add_paper(fig):
    h, w = int(fig.get_figheight() * S.MAP["png_dpi"]), int(fig.get_figwidth() * S.MAP["png_dpi"])
    rng = np.random.default_rng(1909)
    blotch = gaussian_filter(rng.standard_normal((h // 4, w // 4)), 24)
    blotch = np.kron(blotch / blotch.std(), np.ones((4, 4)))[:h, :w]
    blotch = np.pad(blotch, ((0, h - blotch.shape[0]), (0, w - blotch.shape[1])), mode="edge")
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx, w - xx, yy, h - yy]) / (min(h, w) * .12)
    shade = np.clip(1 - edge, 0, 1) ** 2 * .05 + blotch * .006
    paper = np.array(to_rgb(S.BG))
    fig.figimage(np.clip(paper * (1 - shade[..., None]), 0, 1), zorder=-10, origin="lower")


def zebra(ax):
    """Old-map frame: alternating ink and paper along each edge, one band per arc minute."""
    from_m = Transformer.from_crs(CRS_M, 4326, always_xy=True).transform
    x0, x1, y0, y1 = *ax.get_xlim(), *ax.get_ylim()
    for horiz, fixed, a, b in ((True, y0, x0, x1), (True, y1, x0, x1), (False, x0, y0, y1), (False, x1, y0, y1)):
        t = np.linspace(a, b, 1200)
        lon, lat = from_m(t, np.full_like(t, fixed)) if horiz else from_m(np.full_like(t, fixed), t)
        minutes = np.abs(lon if horiz else lat) * 60
        band = np.floor(minutes).astype(int)
        for m in np.unique(band):
            tt = t[band == m]
            xy = (tt, np.full_like(tt, fixed)) if horiz else (np.full_like(tt, fixed), tt)
            ax.plot(*xy, color=S.INK if m % 2 else S.BG, lw=2.6, solid_capstyle="butt", clip_on=False, zorder=8)
            if m % 10 == 0 and fixed in (y0, x0):
                deg, mi = int(m // 60), int(m % 60)
                lab = f"{deg}°{mi:02d}′{'W' if horiz else 'N'}"
                if horiz:
                    ax.annotate(lab, (tt[0], fixed), xytext=(0, -8), textcoords="offset points", ha="center", va="top",
                                fontsize=S.SIZE["credit"], color=S.MUTED, annotation_clip=False)
                else:
                    ax.annotate(lab, (fixed, tt[0]), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                                fontsize=S.SIZE["credit"], color=S.MUTED, annotation_clip=False, rotation=90)
    pos = ax.get_position()
    fx, fy = 1.5 / (pos.width * ax.figure.get_figwidth() * 72), 1.5 / (pos.height * ax.figure.get_figheight() * 72)
    for sign in (-1, 1):
        ax.add_patch(Rectangle((sign * fx, sign * fy), 1 - 2 * sign * fx, 1 - 2 * sign * fy, transform=ax.transAxes,
                               fill=False, ec=S.INK, lw=.6, clip_on=False, zorder=9))


def compass(ax, x, y, r=.042):
    """A small compass rose in axes fraction."""
    asp = ax.get_position().width * ax.figure.get_figwidth() / (ax.get_position().height * ax.figure.get_figheight())
    for k in range(8):
        a = np.pi / 4 * k
        rr = r if k % 2 == 0 else r * .55
        tip = (x + rr * np.sin(a) / asp, y + rr * np.cos(a))
        for side, col in ((1, S.INK), (-1, S.BG)):
            b = a + side * np.pi / 2
            base = (x + r * .11 * np.sin(b) / asp, y + r * .11 * np.cos(b))
            ax.fill([x, tip[0], base[0]], [y, tip[1], base[1]], transform=ax.transAxes, fc=col, ec=S.INK, lw=.35, zorder=8)
    ax.text(x, y + r + .006, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=S.SIZE["annotation"],
            color=S.INK, fontfamily=S.HEAD_FONT, fontweight="bold", zorder=8)


def draw_buildings(ax, b, era, fatten=0):
    polys, cols = [], []
    geoms = shapely.get_parts(b.geometry.values, return_index=True)
    exteriors = shapely.get_exterior_ring(geoms[0])
    coords, idx = shapely.get_coordinates(exteriors, return_index=True)
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    polys = np.split(coords, starts[1:])
    cols = np.array([to_rgb(c) for c in ERAS])[era[geoms[1]]]
    ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=cols if fatten else "none", linewidths=fatten, zorder=3))


def land(ax, city, lw=.5):
    gpd.GeoSeries([city]).plot(ax=ax, color=S.VINTAGE["color"]["land"], edgecolor=S.INK, linewidth=lw, zorder=2)


def make_png(b, era, nta, city, rings, st, title, share):
    S.apply(palette="vintage")
    fig = plt.figure(figsize=S.MAP["png_size_in"], facecolor=S.BG)
    add_paper(fig)
    fig.patch.set_alpha(0)

    ax = fig.add_axes([.024, .062, .64, .845]); ax.set_axis_off(); ax.patch.set_alpha(0)
    x0, y0, x1, y1 = city.bounds
    pad = (y1 - y0) * .025
    h = (y1 - y0) + 2 * pad
    pos = ax.get_position()
    w = h * (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())
    cx = (x0 + x1) / 2 + w * .015
    ax.set_xlim(cx - w / 2, cx + w / 2); ax.set_ylim(y0 - pad, y1 + pad); ax.set_aspect("equal")
    for k, (d, ring) in enumerate(rings):
        gpd.GeoSeries([ring]).plot(ax=ax, color=S.VINTAGE["color"]["water"], linewidth=.5 - k * .08, alpha=.8 - k * .17, zorder=1)
    land(ax, city)
    draw_buildings(ax, b, era, fatten=.22)
    zebra(ax)
    compass(ax, .92, .10)

    # legend in New Jersey, the map's emptiest corner
    t = ax.transAxes
    lx, ly = .025, .955
    ax.text(lx, ly, "Year built", transform=t, fontsize=S.SIZE["panel_title"], fontfamily=S.HEAD_FONT, fontweight="semibold", va="top", zorder=9)
    ax.text(lx, ly - .032, "Share of NYC's buildings", transform=t, fontsize=S.SIZE["annotation"], color=S.MUTED, va="top", zorder=9)
    for k, lab in enumerate(ERA_LABELS):
        yy = ly - .083 - k * .037
        ax.add_patch(Rectangle((lx, yy - .002), .024, .024, transform=t, fc=ERAS[k], ec=S.INK, lw=.3, zorder=9))
        ax.text(lx + .033, yy + .01, f"{lab}", transform=t, fontsize=S.SIZE["legend"] + .5, va="center", zorder=9)
        ax.text(lx + .19, yy + .01, f"{share[k]:.0f}%", transform=t, fontsize=S.SIZE["legend"] + .5, va="center", ha="right", color=S.MUTED, zorder=9)

    # extremes
    named = nta.set_index("ntaname").geometry
    old_g, new_g = named[OLDEST], named[NEWEST]
    op, np_ = old_g.representative_point(), new_g.representative_point()
    S.callout(ax, (op.x, op.y), (.13, .515), "Oldest: Crown Heights South",
              f"{st.pre1940[OLDEST]:.0f}% of buildings predate 1940", ha="left", outline=old_g, halo=True)
    S.callout(ax, (np_.x, np_.y), (.625, .125), "Newest: Spring Creek",
              f"{st.pre1940[NEWEST]:.1f}% predate 1940; half built since {st['median'][NEWEST]:.0f}", ha="left", outline=new_g, halo=True)
    wy = b[b.construction_year == b.construction_year.min()].iloc[0]
    wp = wy.geometry.representative_point()
    S.callout(ax, (wp.x, wp.y), (.555, .045), f"Oldest listed: {wy['name']}, {wy.construction_year:.0f}", None, ha="right", halo=True)

    S.title(fig, title, y=.978)
    fig.text(.5, .925, "AN ATLAS OF THE AGE OF NEW YORK CITY'S BUILDINGS", ha="center", va="center", fontsize=S.SIZE["subtitle"], color=S.MUTED, fontweight="semibold")

    # plates: same ground distance, true footprints
    for num, name, slot, heading in ((1, OLDEST, [.688, .508, .285, .372], "Plate I: Crown Heights South"), (2, NEWEST, [.688, .062, .285, .372], "Plate II: Spring Creek")):
        pa = fig.add_axes(slot); pa.set_axis_off()
        g = named[name]
        mine = b[b.ntaname == name].geometry.centroid
        c = shapely.Point(mine.x.median(), mine.y.median())
        hh = PLATE_M / 2
        pp = pa.get_position(); ww = hh * (pp.width * fig.get_figwidth()) / (pp.height * fig.get_figheight())
        pa.set_xlim(c.x - ww, c.x + ww); pa.set_ylim(c.y - hh, c.y + hh); pa.set_aspect("equal")
        pa.patch.set_alpha(0)
        gpd.GeoSeries([city]).plot(ax=pa, color=S.VINTAGE["color"]["land"], edgecolor="none", zorder=1)
        sel = b.cx[c.x - ww * 1.1:c.x + ww * 1.1, c.y - hh * 1.1:c.y + hh * 1.1]
        draw_buildings(pa, sel, era[sel.index.values])
        gpd.GeoSeries([g.boundary]).plot(ax=pa, color=S.INK, linewidth=.8, zorder=5)
        pa.add_patch(Rectangle((0, 0), 1, 1, transform=pa.transAxes, fill=False, ec=S.INK, lw=.7, clip_on=False, zorder=9))
        pa.add_patch(Rectangle((-.012, -.016), 1.024, 1.032, transform=pa.transAxes, fill=False, ec=S.INK, lw=.35, clip_on=False, zorder=9))
        fig.text(slot[0], slot[1] + slot[3] + .012, heading, fontsize=S.SIZE["panel_title"], fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)
        fig.text(slot[0] + slot[2], slot[1] + slot[3] + .012, "Both plates 1 mile tall", ha="right", fontsize=S.SIZE["annotation"], color=S.MUTED)

    S.credit(fig, f"{HOW} Data: {SOURCE}, NYC Open Data  ·  #30DayMapChallenge {DAY}", y=.014)
    fig.savefig(OUT / "nyc_building_age.png", dpi=S.MAP["png_dpi"], facecolor=S.BG)
    plt.close(fig)


def web_data(b, era, nta, city, rings, st, title, share):
    x0, y0, x1, y1 = city.bounds
    pad = 1200
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    EW, EH = x1 - x0, y1 - y0
    CELL = 128.0
    ncol = int(EW // CELL) + 1
    names = sorted(st.index)
    nidx = {n: i for i, n in enumerate(names)}
    boro = b.ntaname.map(nta.set_index("ntaname").boroname).fillna("Shoreline")

    # one packed chunk per borough, so the page can show each as soon as it arrives
    chunks = []
    for name in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "Shoreline"]:
        sub = b[boro.values == name]
        if not len(sub):
            continue
        fp = footprint_packed(sub, x0, y1, CELL, ncol)
        order = fp["order"]
        n_arr = sub.ntaname.map(nidx).fillna(255).astype(np.uint8).values
        cols = [
            fp["ids"].astype(np.uint32).tobytes(), fp["counts"].astype(np.uint16).tobytes(),
            fp["n_rings"].astype(np.uint16).tobytes(), n_arr[order].tobytes(),
            sub.construction_year.values[order].astype(np.uint16).tobytes(),
            fp["n_verts"].astype(np.uint16).tobytes(), *fp["verts"],
        ]
        chunks.append(dict(name=name, count=len(sub), ncell=len(fp["ids"]), nring=int(fp["n_rings"].sum()),
                           nvert=fp["n_vert_total"], blob=base64.b64encode(gzip.compress(b"".join(cols), 9)).decode()))

    # neighborhoods and shore, in page coordinates, metres rounded
    def rel(geom, tol):
        g = shapely.transform(geom.simplify(tol), lambda c: np.c_[c[:, 0] - x0, y1 - c[:, 1]])
        return g
    def poly_paths(geom, tol=18):
        parts = [p for p in shapely.get_parts(rel(geom, tol)) if p.area > 900]
        d = ""
        for p in parts:
            for ring in [p.exterior, *p.interiors]:
                c = np.round(np.asarray(ring.coords)[:-1]).astype(int)
                d += "M" + "L".join(f"{a} {b_}" for a, b_ in c) + "Z"
        return d
    span = np.digitize(b.construction_year.values, HIST_EDGES)
    hist = pd.crosstab(b.ntaname.values, span).reindex(columns=range(len(HIST_LABELS)), fill_value=0)
    feats = []
    for _, r in nta.iterrows():
        rec = {"d": poly_paths(r.geometry), "name": r.ntaname, "boro": r.boroname}
        if r.ntaname in st.index:
            s = st.loc[r.ntaname]
            rec.update(n=int(s.n), pre=round(float(s.pre1940), 1), post=round(float(s.post2000), 1), med=int(s["median"]), h=hist.loc[r.ntaname].astype(int).tolist())
        feats.append(rec)
    ring_d = []
    for d, ring in rings:
        out = ""
        for c in lines(rel(ring, 15)):
            c = np.round(c).astype(int)
            out += "M" + "L".join(f"{a} {b_}" for a, b_ in c)
        ring_d.append(out)

    return dict(
        title=title, day=DAY, how=HOW, source=SOURCE, source_url=SOURCE_URL,
        lede="Every building in New York City, colored by when it went up. Old neighborhoods aren't always where the city began: "
             "they're where the buildings survived.",
        extent=[EW, EH], cell=CELL, ncol=ncol, count=len(b),
        chunk_names=[c["name"] for c in chunks],
        hist_edges=HIST_EDGES, hist_labels=HIST_LABELS, eras=ERAS, era_labels=ERA_LABELS, era_edges=ERA_EDGES, share=[round(float(s), 1) for s in share],
        names=[[n, nta.set_index("ntaname").boroname[n], round(float(st.pre1940[n]), 1)] for n in names],
        feats=feats, rings=ring_d, land=poly_paths(city, 12), seq=S.SEQUENTIAL_AMBER,
        citywide=round(float((b.construction_year < 1940).mean() * 100), 1),
    ), chunks


def main():
    b = load_buildings()
    nta = gpd.read_file(RAW / "nta2020.geojson").to_crs(CRS_M)
    nta["geometry"] = nta.geometry.buffer(0)
    b, st, res = neighborhood_stats(b, nta)

    top, low = st.pre1940.idxmax(), st.pre1940.idxmin()
    assert top == OLDEST and low == NEWEST, f"extremes changed: {top}, {low}; update the headline"
    title = (f"In {SHORT[OLDEST]}, {st.pre1940[OLDEST]:.0f}% of buildings predate 1940. "
             f"In {SHORT[NEWEST]}, {st.pre1940[NEWEST]:.1f}%.")
    era = np.digitize(b.construction_year.values, ERA_EDGES).astype(int)
    share = np.bincount(era, minlength=len(ERAS)) / len(era) * 100
    print(title)
    print("buildings", len(b), "| in ranked neighborhoods", len(res), "| neighborhoods", len(st))
    print("era shares", dict(zip(ERA_LABELS, share.round(1))))
    print(st.sort_values("pre1940").iloc[[0, 1, -2, -1]].round(1))

    city = shapely.make_valid(shapely.union_all(nta.geometry.values))
    rings = water_rings(city)

    if "--web-only" not in sys.argv:
        make_png(b, era, nta, city, rings, st, title, share)
        print("wrote png")
    data, chunks = web_data(b, era, nta, city, rings, st, title, share)
    push = lambda c: "(window.__chunks=window.__chunks||[]).push(" + json.dumps(c, separators=(",", ":")) + ");"
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n" + "\n".join(push(c) for c in chunks))
    out = OUT / "nyc_building_age.html"
    S.build_page(HERE / "web" / "index.html", data, out)
    # Buildings go after the page's own script: the map shows first, then each borough as it streams in.
    page = out.read_text()
    earcut = (HERE / "web" / "earcut.min.js").read_text()
    page = page.replace('<script src="earcut.min.js"></script>', f"<script>{earcut}</script>")
    tags = "\n".join(f"<script>{push(c)}</script>" for c in chunks)
    at = page.rindex("</body>")
    page = page[:at] + tags + "\n" + page[at:]
    out.write_text(page)
    print(f"wrote html ({len(page) / 1e6:.1f} MB)", {c["name"]: f"{len(c['blob']) / 1e6:.1f} MB" for c in chunks})


if __name__ == "__main__":
    main()
