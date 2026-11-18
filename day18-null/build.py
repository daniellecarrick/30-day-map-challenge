"""Day 18 · Null — the US as a paper sheet with every zero-population Census block cut out.

Each 2020 Census block with 0 residents and some land is a hole in the paper. The sheet casts a
soft shadow onto the table below, so the holes read as depth rather than as a color.

Stage 1 (slow, cached in data/proc/): read each state's TIGER/Line blocks, keep the empty ones,
  simplify them, and total land / empty land / people by state and county.
Stage 2: rasterize the paper masks (national + one per state), draw the PNG and build the page.
  The page gets 1-bit masks and draws the paper, shadow and cut edge itself on a canvas.

Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained)

Data: US Census Bureau, 2020 Census TIGER/Line blocks (POP20, ALAND20)
      https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/  and cartographic boundaries (cb_2020 500k)
Run:  ../.venv/bin/python build.py            (add --only 08 to stage-1 a single state)
"""
import base64
import io
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
from matplotlib.colors import to_rgb
from PIL import Image
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
PROC = DATA / "proc"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
PROC.mkdir(exist_ok=True)

DAY = "Day 18: Null"
SOURCE = "Data: US Census Bureau, 2020 Census blocks (TIGER/Line)"
SOURCE_URL = "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2020.html"
SQMI = 2_589_988.11  # m² per square mile

V = S.VINTAGE["color"]


def mix(a, b, t):
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return a + (b - a) * t


# Paper-cut colors, all mixed from the vintage tokens.
PAPER = np.array(to_rgb(V["land"]))
TABLE = mix(V["bg"], V["muted"], 0.38)    # the surface under the sheet, seen through the holes
WALL = mix(V["land"], V["muted"], 0.45)   # the cut edge (paper thickness)
SHADOW = np.array(to_rgb(V["ink"]))
hexc = lambda c: "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in c)  # noqa: E731

STATES = gpd.read_file(DATA / "cb_2020_us_state_500k.zip")
STATES = STATES[STATES.STATEFP.astype(int) <= 56].set_index("STATEFP")  # 50 states + DC
COUNTIES = gpd.read_file(DATA / "cb_2020_us_county_500k.zip")
COUNTIES = COUNTIES[COUNTIES.STATEFP.isin(STATES.index)]


# ---------------------------------------------------------------- stage 1: per state, cached
def process(fips: str) -> dict:
    cache = PROC / f"{fips}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    f = DATA / "blocks" / f"tl_2020_{fips}_tabblock20.zip"
    b = gpd.read_file(f, columns=["GEOID20", "POP20", "HOUSING20", "ALAND20"])
    b["county"] = b.GEOID20.str[:5]
    b["zero"] = (b.POP20 == 0) & (b.ALAND20 > 0)
    b["empty_land"] = b.ALAND20.where(b.zero, 0)
    b["land_block"] = b.ALAND20 > 0
    by = b.groupby("county").agg(land=("ALAND20", "sum"), empty_land=("empty_land", "sum"),
                                 pop=("POP20", "sum"), blocks=("land_block", "sum"), empty=("zero", "sum"))
    holes = b.loc[b.zero, ["geometry"]].to_crs("EPSG:5070")
    holes["geometry"] = holes.geometry.simplify(30)
    holes = holes[~holes.geometry.is_empty].to_crs("EPSG:4326")
    res = {
        "fips": fips,
        "land": float(b.ALAND20.sum()), "empty_land": float(b.empty_land.sum()),
        "pop": int(b.POP20.sum()), "blocks": int(b.land_block.sum()), "empty": int(b.zero.sum()),
        "empty_housing": int((b.zero & (b.HOUSING20 > 0)).sum()),
        "counties": by.reset_index().to_dict("records"),
        "holes": holes.geometry.values,
    }
    cache.write_bytes(pickle.dumps(res))
    print(f"  {fips} {STATES.loc[fips, 'STUSPS']}: {res['empty']:,} empty blocks, "
          f"{res['empty_land'] / res['land']:.1%} of land", flush=True)
    return res


# ---------------------------------------------------------------- geometry helpers
def state_crs(fips):
    """Equal-area projection centered on one state."""
    lon0, lat0, lon1, lat1 = STATES.loc[[fips]].to_crs("EPSG:4326").total_bounds
    if fips == "02":
        return "EPSG:3338"
    h = lat1 - lat0
    return (f"+proj=aea +lat_1={lat0 + h / 6:.3f} +lat_2={lat1 - h / 6:.3f} +lat_0={(lat0 + lat1) / 2:.3f} "
            f"+lon_0={(lon0 + lon1) / 2:.3f} +datum=NAD83 +units=m +no_defs")


HI_CRS = "+proj=aea +lat_1=19 +lat_2=22 +lat_0=20.5 +lon_0=-157 +datum=NAD83 +units=m +no_defs"
# National layout: CONUS in EPSG:5070; Alaska and Hawaii moved into insets (scale, dx, dy).
INSETS = {"02": ("EPSG:3338", 0.36, -1_850_000, -150_000), "15": (HI_CRS, 1.0, -720_000, 520_000)}


def to_layout(geoms: gpd.GeoSeries, fips: str) -> gpd.GeoSeries:
    if fips not in INSETS:
        return geoms.to_crs("EPSG:5070")
    crs, k, dx, dy = INSETS[fips]
    g = geoms.to_crs(crs)
    return g.affine_transform([k, 0, 0, k, dx, dy])


def grid(bounds, long_side, pad=0.04, pad_bottom=None):
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    p = max(w, h) * pad
    pb = p if pad_bottom is None else max(w, h) * pad_bottom
    x0, x1, y0, y1 = x0 - p, x1 + p, y0 - pb, y1 + p
    res = max(x1 - x0, y1 - y0) / long_side
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    return from_origin(x0, y1, res, res), (H, W), (x0, x1, y0, y1)


def burn(geoms, tr, shape):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, np.uint8)
    return rasterio.features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=tr, dtype="uint8")


def svg_path(geom, tr, tol=0.6):
    """Geometry (in the raster's CRS) -> SVG path in pixel units of that raster."""
    a, _, c, _, e, f = tr.a, tr.b, tr.c, tr.d, tr.e, tr.f
    g = affinity.affine_transform(geom, [1 / a, 0, 0, 1 / e, -c / a, -f / e]).simplify(tol)
    polys = g.geoms if isinstance(g, MultiPolygon) else [g] if isinstance(g, Polygon) else []
    out = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 4:
                continue
            out.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in xy[:-1]) + "Z")
    return "".join(out)


def mask_png(paper: np.ndarray) -> str:
    """1-bit PNG (white = paper) as a data URL."""
    im = Image.fromarray((paper > 0).astype(np.uint8) * 255).convert("1")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def supersample(fn, tr, shape, k=2):
    """Rasterize at k× and average down, so the 1-bit mask keeps fine shapes honestly."""
    tr2 = tr * tr.scale(1 / k)
    big = fn(tr2, (shape[0] * k, shape[1] * k)).astype(np.float32)
    return big.reshape(shape[0], k, shape[1], k).mean(axis=(1, 3))


# ---------------------------------------------------------------- the paper-cut render (PNG)
def paper_cut(paper: np.ndarray, s: float, lines=None) -> np.ndarray:
    """paper: float mask 0..1. s: effect scale (1 = tuned for a 4800px-wide raster).
    lines: optional 0..1 mask of pencil lines drawn on the table (state outlines), seen only through holes."""
    wall = np.clip(shift(paper, (3 * s, 0), order=1) - paper, 0, 1)
    sh = gaussian_filter(shift(paper, (10 * s, 7 * s), order=1), 9 * s)
    sh_tight = gaussian_filter(shift(paper, (3 * s, 2 * s), order=1), 2.5 * s)
    shadow = np.clip(0.38 * sh + 0.45 * sh_tight, 0, 0.75)
    table = np.broadcast_to(TABLE, paper.shape + (3,))
    if lines is not None:
        table = table * (1 - 0.5 * lines[..., None]) + SHADOW * 0.5 * lines[..., None]
    img = table * (1 - shadow[..., None]) + SHADOW[None, None, :] * shadow[..., None] * 0.35
    img = img * (1 - wall[..., None]) + WALL[None, None, :] * wall[..., None]
    img = img * (1 - paper[..., None]) + PAPER[None, None, :] * paper[..., None]
    return np.clip(img, 0, 1)


def pct(x):
    return f"{x:.0%}" if x >= 0.01 else "<1%"


# ---------------------------------------------------------------- stage 2
def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    fipss = [only] if only else sorted(STATES.index)
    if "--partial" in sys.argv:  # dev: stage 2 from a few finished states, e.g. --partial 08,10,11
        fipss = sys.argv[sys.argv.index("--partial") + 1].split(",")
    missing = [f for f in fipss if not (DATA / "blocks" / f"tl_2020_{f}_tabblock20.zip").exists()]
    if missing:
        sys.exit(f"missing block files: {missing}")
    todo = [f for f in fipss if not (PROC / f"{f}.pkl").exists()]
    if todo:
        print(f"stage 1: {len(todo)} states")
        with Pool(3) as pool:
            pool.map(process, todo, chunksize=1)
    if only:
        return
    R = {f: process(f) for f in fipss}

    S.apply(palette="vintage")
    land = sum(r["land"] for r in R.values())
    empty_land = sum(r["empty_land"] for r in R.values())
    share = empty_land / land
    n_empty = sum(r["empty"] for r in R.values())
    n_blocks = sum(r["blocks"] for r in R.values())
    conus = [f for f in fipss if f not in INSETS]
    conus_share = sum(R[f]["empty_land"] for f in conus) / sum(R[f]["land"] for f in conus)
    print(f"US: {share:.1%} of land in empty blocks ({n_empty:,} of {n_blocks:,} land blocks, "
          f"{n_empty / n_blocks:.1%}); lower 48 + DC: {conus_share:.1%}")
    st = pd.DataFrame([{"fips": f, "abbr": STATES.loc[f, "STUSPS"], "name": STATES.loc[f, "NAME"],
                        "share": R[f]["empty_land"] / R[f]["land"], "empty_sqmi": R[f]["empty_land"] / SQMI}
                       for f in fipss]).sort_values("share")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- national layout
    shapes = {f: to_layout(STATES.loc[[f]].geometry, f).iloc[0] for f in fipss}
    lay_bounds = gpd.GeoSeries(list(shapes.values())).total_bounds

    def national_paper(tr, shape):
        paper = burn(shapes.values(), tr, shape)
        for f in fipss:
            holes = to_layout(gpd.GeoSeries(R[f]["holes"], crs="EPSG:4326"), f)
            paper &= 1 - burn(holes, tr, shape)
        return paper

    # PNG (supersampled 4800 wide, drawn at 2400)
    tr, shape, ext = grid(lay_bounds, 4800, pad=0.03)
    paper = supersample(national_paper, tr, shape)
    borders = [g.boundary for g in shapes.values()]
    lines = rasterio.features.rasterize(((g, 1) for g in borders), out_shape=shape, transform=tr,
                                        all_touched=True, dtype="uint8").astype(np.float32)
    img = paper_cut(paper, 1.0, lines)
    small = Image.fromarray((img * 255).astype(np.uint8)).resize((2400, round(2400 * shape[0] / shape[1])),
                                                                 Image.LANCZOS)
    title = f"{share:.0%} of America's land is home to no one"
    fig = plt.figure(figsize=(12, 12 * small.height / small.width + 1.5))
    fig.patch.set_facecolor(TABLE)
    ax = fig.add_axes([0, 0.04, 1, 0.84])
    ax.imshow(np.asarray(small), extent=ext, interpolation="lanczos")
    ax.set_axis_off()
    by_share = sorted(fipss, key=lambda f: R[f]["empty_land"] / R[f]["land"])
    for f, xy, ha, note in [(by_share[-1], (0.03, 0.355), "left", "the emptiest state"),
                            (by_share[0], (0.985, 0.33), "right", "the fullest state")]:
        g = shapes[f]
        sh_ = R[f]["empty_land"] / R[f]["land"]
        S.callout(ax, (g.representative_point().x, g.representative_point().y), xy,
                  f"{STATES.loc[f, 'NAME']}: {sh_:.0%}", note,
                  ha=ha, outline=g)
    S.title(fig, title, y=0.965)
    fig.text(0.5, 0.905, "Every hole is a 2020 Census block where nobody lives", ha="center",
             fontsize=S.SIZE["subtitle"] + 2, color=S.MUTED)
    S.credit(fig, f"Share of land area in blocks with 0 residents  ·  {SOURCE}  ·  #30DayMapChallenge {DAY}",
             y=0.015)
    fig.savefig(OUT / "map.png", dpi=200, facecolor=TABLE)
    plt.close(fig)
    if "--png-only" in sys.argv:  # dev: skip the web build
        return

    # web mask (3000 wide, 1-bit)
    trw, shapew, _ = grid(lay_bounds, 3000, pad=0.03)
    nat_mask = supersample(national_paper, trw, shapew) >= 0.5
    nat_states = []
    for f in fipss:
        r = R[f]
        nat_states.append({"fips": f, "abbr": STATES.loc[f, "STUSPS"], "name": STATES.loc[f, "NAME"],
                           "d": svg_path(shapes[f], trw, 0.8), "share": r["empty_land"] / r["land"],
                           "emptySqmi": r["empty_land"] / SQMI, "landSqmi": r["land"] / SQMI,
                           "empty": r["empty"], "blocks": r["blocks"], "pop": r["pop"]})

    # ---- one view per state
    views = {}
    cnames = COUNTIES.set_index("GEOID")
    for f in fipss:
        r = R[f]
        crs = state_crs(f)
        outline = STATES.loc[[f]].to_crs(crs).geometry.iloc[0]
        holes = gpd.GeoSeries(r["holes"], crs="EPSG:4326").to_crs(crs)
        trs, shp, _ = grid(outline.bounds, 2000, pad=0.03)
        m = supersample(lambda t, s: burn([outline], t, s) & (1 - burn(holes, t, s)), trs, shp) >= 0.5
        cg = cnames[cnames.STATEFP == f].to_crs(crs)
        counties = []
        for c in r["counties"]:
            if c["county"] not in cg.index or c["land"] <= 0:
                continue
            row = cg.loc[c["county"]]
            counties.append({"name": row.NAMELSAD, "d": svg_path(row.geometry, trs, 0.8),
                             "share": c["empty_land"] / c["land"], "emptySqmi": c["empty_land"] / SQMI,
                             "empty": int(c["empty"]), "blocks": int(c["blocks"]), "pop": int(c["pop"])})
        views[f] = {"w": shp[1], "h": shp[0], "mask": mask_png(m), "counties": counties,
                    "outline": svg_path(outline, trs, 0.8)}
        print(f"  view {STATES.loc[f, 'STUSPS']}: {shp[1]}x{shp[0]}, mask {len(views[f]['mask']) / 1e3:.0f} KB",
              flush=True)

    data = {
        "title": title, "day": DAY, "source": SOURCE, "sourceUrl": SOURCE_URL,
        "share": share, "emptyBlocks": n_empty, "blocks": n_blocks, "emptySqmi": empty_land / SQMI,
        "colors": {"paper": hexc(PAPER), "table": hexc(TABLE), "wall": hexc(WALL), "shadow": hexc(SHADOW)},
        "national": {"w": shapew[1], "h": shapew[0], "mask": mask_png(nat_mask), "states": nat_states},
        "views": views,
    }
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
