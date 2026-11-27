"""Day 27 · A new tool — a demographics lens over New York, powered by DuckDB running in the browser.

ACS block-group counts (sex by age, household income brackets) are spread over each block group's
2020 census blocks by population, packed into Parquet and inlined in the page. DuckDB-WASM sums the
blocks inside the circle under the cursor; medians are interpolated from the summed brackets.
Writes:
  - out/nyc_lens.png    (static, for posting)
  - out/nyc_lens.html   (interactive; self-contained apart from DuckDB-WASM from jsDelivr)

Data (fetch.sh):
  - ACS 2020–24 5-year, tables B01001 (sex by age) and B19001 (household income), block groups
  - 2020 Census blocks (POP20, HOUSING20, internal points, shapes): TIGER/Line tl_2020_36_tabblock20
Run:  ../.venv/bin/python build.py
"""
import base64
import io
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import shapely
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 27: A new tool"
SOURCE = "Data: US Census Bureau, American Community Survey 2020–24 (block groups) and 2020 Census blocks"
SOURCE_URL = "https://www.census.gov/programs-surveys/acs/data/summary-file.html"
CRS_M = "EPSG:32618"           # UTM 18N, metres
HALF_MILE = 804.672            # metres
MIN_POP = 10000                # a circle needs this many residents to count as an extreme
COUNTIES = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan", "081": "Queens", "085": "Staten Island"}

# B01001 fine age bins (same for men, cells 3-25, and women, cells 27-49), by starting age
FINE_AGE = [0, 5, 10, 15, 18, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60, 62, 65, 67, 70, 75, 80, 85]
AGE_EDGES = [0, 10, 18, 25, 35, 45, 55, 65, 75, 85, 100]            # coarse bins kept for the page
AGE_COLS = [f"age_{a:02d}_{b - 1:02d}" if b < 100 else f"age_{a:02d}_up" for a, b in zip(AGE_EDGES, AGE_EDGES[1:])]
# B19001 household income brackets (cells 2-17), $ thousands
_INC_K = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100, 125, 150, 200, None]
INC_COLS = [f"inc_{a:03d}_{b:03d}" if b else f"inc_{a:03d}_up" for a, b in zip(_INC_K, _INC_K[1:])]
INC_EDGES = [k * 1000 if k is not None else None for k in _INC_K]   # dollars

# The six income groups shared by the map, its legend and the bar charts: (label, first bracket, end bracket).
# Brown below $75k, blue from $75k up; the city median ($80,900) sits just above the split.
BUCKETS = [("Under $25k", 0, 4), ("$25–50k", 4, 9), ("$50–75k", 9, 11),
           ("$75–100k", 11, 12), ("$100–200k", 12, 15), ("$200k+", 15, 16)]
BUCKET_LOW = [INC_EDGES[a] for _, a, _ in BUCKETS]
_ramp = S.make_scale([(k / 4, c) for k, c in enumerate(S.DIVERGING[::-1])])   # brown -> neutral -> blue
BUCKET_COLORS = [_ramp(t) for t in (0, 0.2, 0.38, 0.62, 0.8, 1.0)]          # skip the neutral middle


def bucket_of(v):
    return int(np.searchsorted(BUCKET_LOW, v, side="right") - 1)


def bucket_shares(inc_counts):
    c = np.asarray(inc_counts, float)
    tot = c.sum() or 1.0
    return [c[a:b].sum() / tot for _, a, b in BUCKETS]

# Friendlier names for the extremes than the NTA names (filled in after looking at the output)
PLACE = {"inc_max": "Tribeca", "inc_min": "Mott Haven", "age_max": "Glen Oaks", "age_min": "Borough Park"}


def median_from_bins(counts, edges):
    """Census-style linear interpolation of a median from binned counts (edges: len(counts)+1;
    None for an open top). In the open top bin, returns its lower edge."""
    counts = np.asarray(counts, float)
    tot = counts.sum()
    if tot <= 0:
        return np.nan
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, tot / 2))
    lo, hi = edges[k], edges[k + 1]
    if hi is None:
        return float(lo)
    before = cum[k - 1] if k else 0.0
    return lo + (tot / 2 - before) / counts[k] * (hi - lo)


# ---------- loading ----------
def load():
    blk = pyogrio.read_dataframe(DATA / "tabblock20.zip",
                                 columns=["GEOID20", "COUNTYFP20", "POP20", "HOUSING20", "INTPTLAT20", "INTPTLON20", "ALAND20"])
    blk = blk[blk.COUNTYFP20.isin(COUNTIES)].copy()
    land_m2 = float(blk.ALAND20.sum())
    blk["bg"] = blk.GEOID20.str[:12]
    blk = blk.to_crs(CRS_M)

    def acs(table):
        d = pd.read_csv(DATA / f"acs_{table.lower()}.dat", sep="|", dtype=str)
        d["bg"] = d.GEO_ID.str[-12:]
        return d.set_index("bg")

    a = acs("B01001")
    e = lambda c: a[f"B01001_E{c:03d}"].astype(float)
    g = pd.DataFrame(index=a.index)
    g["pop"], g["men"], g["women"] = e(1), e(2), e(26)
    coarse = np.searchsorted(AGE_EDGES, FINE_AGE, side="right") - 1
    for j, col in enumerate(AGE_COLS):
        g[col] = sum(e(3 + i) + e(27 + i) for i, c in enumerate(coarse) if c == j)
    inc = acs("B19001")
    g["hh"] = inc["B19001_E001"].astype(float)
    for j, col in enumerate(INC_COLS):
        g[col] = inc[f"B19001_E{j + 2:03d}"].astype(float)
    g["age_med"] = [median_from_bins(r, AGE_EDGES) for r in g[AGE_COLS].values]
    g["inc_med"] = [median_from_bins(r, INC_EDGES) for r in g[INC_COLS].values]

    # spread each block group over its blocks: people by 2020 population, households by housing units
    blk = blk[blk.bg.isin(g.index)].copy()
    pop_share = (blk.POP20 / blk.groupby("bg").POP20.transform("sum")).fillna(0)
    hu_share = (blk.HOUSING20 / blk.groupby("bg").HOUSING20.transform("sum")).fillna(pop_share)
    gb = g.loc[blk.bg]
    people = ["pop", "men", "women"] + AGE_COLS
    homes = ["hh"] + INC_COLS
    pts = pd.DataFrame(gb[people].values * pop_share.values[:, None], columns=people)
    pts[homes] = gb[homes].values * hu_share.values[:, None]
    ip = gpd.GeoSeries(gpd.points_from_xy(blk.INTPTLON20.astype(float), blk.INTPTLAT20.astype(float)), crs=4326).to_crs(CRS_M)
    pts["x"], pts["y"] = ip.x.values, ip.y.values
    pts = pts[(pts["pop"] > 0.05) | (pts.hh > 0.05)].reset_index(drop=True)

    # block-group shapes (land only) for the choropleth layers
    bgs = blk[blk.ALAND20 > 0][["bg", "geometry"]].dissolve("bg").join(g[["pop", "hh", "age_med", "inc_med"]])
    return pts, bgs, land_m2


def parquet_b64(cols: dict) -> str:
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf, compression="zstd", compression_level=19, row_group_size=1 << 20)
    return base64.b64encode(buf.getvalue()).decode()


# ---------- lens extremes ----------
def lens_sums(xy, W, r=HALF_MILE, cell=50.0):
    """Sum each column of W inside a radius-r circle centred on every cell of a 50 m grid."""
    x0, y0 = xy.min(0) - r
    nx, ny = ((np.ptp(xy, 0) + 2 * r) // cell + 1).astype(int)
    rng = [[x0, x0 + nx * cell], [y0, y0 + ny * cell]]
    k = int(np.ceil(r / cell))
    gx, gy = np.mgrid[-k:k + 1, -k:k + 1] * cell
    disk = (gx ** 2 + gy ** 2 <= r ** 2).astype(float)
    out = []
    for w in W.T:
        H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[nx, ny], range=rng, weights=w)
        out.append(np.clip(fftconvolve(H, disk, mode="same"), 0, None))
    return np.stack(out, -1), x0 + (np.arange(nx) + 0.5) * cell, y0 + (np.arange(ny) + 0.5) * cell


def summarize(s):
    return {"shares": bucket_shares(s[INC_COLS].values),"pop": float(s["pop"]), "women": float(s["women"]), "hh": float(s["hh"]),
            "age_med": median_from_bins(s[AGE_COLS].values, AGE_EDGES),
            "inc_med": median_from_bins(s[INC_COLS].values, INC_EDGES),
            "under18": float(s[AGE_COLS[:2]].sum()), "over65": float(s[AGE_COLS[7:]].sum())}


def exact(pts, cx, cy, r=HALF_MILE):
    return summarize(pts[(pts.x - cx) ** 2 + (pts.y - cy) ** 2 <= r * r].sum())


def place_name(x, y, nta):
    """Neighbourhood (NTA) at the point, or the nearest one."""
    p = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=CRS_M).iloc[0]
    return nta.ntaname.iloc[int(nta.distance(p).values.argmin())]


def extremes(pts, land):
    """Centres of the half-mile circles (with at least MIN_POP residents) with the highest and
    lowest median household income and median age."""
    sums, gxs, gys = lens_sums(pts[["x", "y"]].values, pts[["pop"] + AGE_COLS + INC_COLS].values)
    idx = np.argwhere(sums[..., 0] >= MIN_POP)
    idx = idx[:: max(1, len(idx) // 80000)]
    idx = idx[shapely.contains_xy(land, gxs[idx[:, 0]], gys[idx[:, 1]])]      # centre on populated land
    na = len(AGE_COLS)
    found = {}
    for key, sl, edges in [("inc", slice(1 + na, None), INC_EDGES)]:
        med = np.array([median_from_bins(sums[i, j, sl], edges) for i, j in idx])
        if key == "inc":   # many circles top out at $200k+: among those, take the one with the most $200k+ share
            top = sums[idx[:, 0], idx[:, 1], -1] / sums[idx[:, 0], idx[:, 1], sl].sum(-1)
            med = med + top
        for tag, k in [("max", np.nanargmax(med)), ("min", np.nanargmin(med))]:
            i, j = idx[k]
            found[f"{key}_{tag}"] = (gxs[i], gys[j])
    return found


def money(v):
    return f"${v / 1000:,.0f}k" + ("+" if v >= 200000 else "")


# ---------- PNG ----------
def draw_png(bgs, ext, city, title, sub):
    S.apply()
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.0, 0.05, 0.64, 0.84])
    shown = bgs[(bgs.hh >= 20) & bgs.inc_med.notna()]
    bgs.plot(ax=ax, color=S.C["other"], edgecolor="none")
    shown.plot(ax=ax, color=[BUCKET_COLORS[bucket_of(v)] for v in shown.inc_med], edgecolor=S.EDGE, linewidth=0.1)
    gpd.GeoSeries([bgs.union_all()], crs=CRS_M).boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"])
    for key, head, txy in [("inc_max", "Highest median income", (0.03, 0.78)),
                           ("inc_min", "Lowest median income", (0.66, 0.95))]:
        e = ext[key]
        ax.add_patch(Circle((e["x"], e["y"]), HALF_MILE, fill=False, ec=S.INK, lw=1.8, zorder=6))
        S.callout(ax, (e["x"], e["y"]), txy, head,
                  f"{money(e['inc_med'])} a year within half a mile of {e['place']}", halo=True)
    ax.set_axis_off()
    ax.set_aspect("equal")
    x0, y0, x1, y1 = bgs.total_bounds
    ax.set_xlim(x0 - 800, x1 + 800)
    ax.set_ylim(y0 - 800, y1 + 800)

    # legend: the six groups, highest first (same order as the bar charts)
    lg = fig.add_axes([0.03, 0.40, 0.16, 0.26])   # empty water west of Manhattan, above Staten Island
    lg.set_axis_off()
    lg.text(0, 1.0, "Median household income,\nby block group", fontsize=S.SIZE["legend"] + 1, fontweight="semibold",
            color=S.INK, va="top")
    rows = list(zip([b[0] for b in BUCKETS], BUCKET_COLORS))[::-1] + [("Under 20 households", S.C["other"])]
    for k, (lab, c) in enumerate(rows):
        y = 0.70 - k * 0.115
        lg.add_patch(plt.Rectangle((0, y - 0.07), 0.09, 0.08, color=c, transform=lg.transAxes, clip_on=False))
        lg.text(0.14, y - 0.03, lab, fontsize=S.SIZE["legend"] + 0.5, color=S.INK, va="center")

    # the two circles as bar charts of the same six groups
    a, b = ext["inc_max"], ext["inc_min"]
    fig.text(0.69, 0.80, f"Two half-mile circles, {ext['dist_mi']:.0f} miles apart", fontsize=S.SIZE["panel_title"],
             fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK, va="top")
    fig.text(0.69, 0.765, "Share of households in each income group", fontsize=S.SIZE["legend"] + 1, color=S.MUTED, va="top")
    for k, e in enumerate([a, b]):
        bx = fig.add_axes([[0.755, 0.885][k], 0.22, 0.085, 0.48])   # left chart carries the group labels
        vals = e["shares"][::-1]
        labels = [b_[0] for b_ in BUCKETS][::-1]
        cols = BUCKET_COLORS[::-1]
        yy = np.arange(len(vals))[::-1]
        bx.barh(yy, vals, color=cols, height=0.72)
        for y_, v in zip(yy, vals):
            bx.text(v + 0.02, y_, f"{100 * v:.0f}%", va="center", fontsize=S.SIZE["legend"] + 0.5, color=S.INK)
        bx.set_xlim(0, 0.95)
        bx.set_yticks(yy, labels if k == 0 else [""] * len(labels))
        bx.tick_params(axis="y", length=0, labelsize=S.SIZE["legend"] + 0.5)
        bx.set_xticks([])
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.set_title(f"{e['place']}\nmedian {money(e['inc_med'])}", loc="left", fontsize=S.SIZE["legend"] + 1.5,
                     fontweight="semibold", color=S.INK, pad=6)

    S.title(fig, title.replace("$", r"\$"))   # two $ would switch matplotlib into math mode
    fig.text(0.5, 0.925, sub, ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, "Circles: block-group estimates spread over census blocks by housing units; medians interpolated from "
                  f"income brackets. Grey: under 20 households.  {SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_lens.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


def headline(ext):
    a, b = ext["inc_max"], ext["inc_min"]
    assert a["inc_med"] >= 200000, "headline assumes the top bracket"
    title = (f"Within half a mile of {a['place']}, the typical household makes $200,000 or more. "
             f"In {b['place']}, ${round(b['inc_med'], -3):,.0f}")
    sub = "The interactive version sums household income for any half-mile circle you point at, with DuckDB running in the browser"
    return title, sub


def main():
    pts, bgs, land_m2 = load()
    nta = gpd.read_file(ROOT / "day20-hexagons/data/nta.geojson").to_crs(CRS_M)
    city = summarize(pts.sum())
    print("city:", {k: (round(v, 1) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in city.items()})

    ext = {}
    land = bgs[bgs.hh >= 50].union_all()        # centre in a neighbourhood with households, not a shelter or park
    shapely.prepare(land)
    for k, (cx, cy) in extremes(pts, land).items():
        ext[k] = {"x": cx, "y": cy, **exact(pts, cx, cy), "place": PLACE.get(k) or place_name(cx, cy, nta)}
        print(k, {kk: (round(v) if isinstance(v, float) else v) for kk, v in ext[k].items() if kk != "shares"}, [round(x, 2) for x in ext[k]["shares"]])
    a, b = ext["inc_max"], ext["inc_min"]
    ext["dist_mi"] = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) / 1609.344)
    print("income extremes are", round(ext["dist_mi"], 1), "miles apart")
    title, sub = headline(ext)
    print(title)
    draw_png(bgs, ext, city, title, sub)

    # ---- interactive page: coordinates in metres from the SW corner, as uint16 ----
    x0, y0, _, _ = bgs.total_bounds
    x0, y0 = np.floor(x0) - 200, np.floor(y0) - 200
    cols = {"x": pa.array(np.round(pts.x - x0).astype(np.uint16)), "y": pa.array(np.round(pts.y - y0).astype(np.uint16))}
    cols.update({c: pa.array(np.round(pts[c].values, 2).astype(np.float32))
                 for c in ["pop", "hh"] + INC_COLS})
    tables = {"blocks": parquet_b64(cols)}
    print(f"  blocks: {len(tables['blocks']) * 3 / 4 / 1e6:.2f} MB parquet, {len(pts):,} rows")

    shapes = bgs[["geometry", "hh", "pop", "inc_med"]].copy()
    shapes["geometry"] = shapes.geometry.simplify(12).translate(-x0, -y0)
    shapes["inc_med"] = shapes.inc_med.round(-2)
    shapes[["hh", "pop"]] = shapes[["hh", "pop"]].round()
    outline = gpd.GeoSeries([bgs.union_all().simplify(15)], crs=CRS_M).translate(-x0, -y0)
    data = {
        "title": title, "day": DAY, "source": SOURCE, "sourceUrl": SOURCE_URL,
        "tables": tables, "landM2": land_m2, "city": city,
        "incCols": INC_COLS, "incEdges": INC_EDGES,
        "buckets": [{"label": l, "a": a_, "b": b_, "low": lo, "color": c}
                    for (l, a_, b_), lo, c in zip(BUCKETS, BUCKET_LOW, BUCKET_COLORS)],
        "bgs": json.loads(shapes.to_json(drop_id=True)),
        "outline": json.loads(gpd.GeoDataFrame(geometry=outline).to_json(drop_id=True)),
        "start": [round(a["x"] - x0), round(a["y"] - y0)],
        "halfMile": HALF_MILE, "nBlocks": len(pts),
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "nyc_lens.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    print("wrote out/nyc_lens.png and out/nyc_lens.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
