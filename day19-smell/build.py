"""Day 19 · Senses: Smell — the lilac wave.

When lilacs first open across the lower 48, from the USA National Phenology Network's Extended
Spring Index: a model of first bloom for a cloned lilac, driven by daily PRISM temperatures (4 km).
  - Typical year: 1991–2020 average of the yearly first-bloom maps
  - Dots: real lilacs watched by Nature's Notebook volunteers, 2009–2025 (median first open flowers)
Grey: the typical first bloom comes before mid-March (day 75). Winters there are mostly too warm for
lilacs to flower; the model still gives a date, so we leave it out.

Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained)

Data: USA-NPN geoserver (si-x:lilac_bloom_prism, 1981–2025) and NPN observations API,
      downloaded by fetch.sh into data/
Run:  ../.venv/bin/python build.py
"""
import base64
import json
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from rasterio import features
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean over ocean cells

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 19: Senses: Smell"
SOURCE = "Data: USA National Phenology Network, Extended Spring Index (PRISM) and Nature's Notebook"
SOURCE_URL = "https://www.usanpn.org/data/spring_indices"
CRS = "EPSG:5070"        # Albers equal-area, lower 48
RES = 5000               # metres per web/PNG grid cell
YEARS = np.arange(1981, 2026)
WARM = 75                # typical first bloom before this day of year (Mar 16) -> too warm for lilacs
D0, D1 = 60, 222         # season shown on the page: Mar 1 .. Aug 10
# most of the country opens in April–May, so the purple steps are packed there: Mar 16, Apr 1, Apr 20, May 10, Jun 1, Jul 1
SEASON_STOPS = list(zip([75, 91, 110, 130, 152, 182], S.SEQUENTIAL_PURPLE[1:]))


def load_stack():
    cache = DATA / "proc" / "stack.npz"
    if cache.exists():
        z = np.load(cache)
        return z["A"], rasterio.Affine(*z["tr"]), str(z["crs"])
    A = []
    for y in YEARS:
        with rasterio.open(DATA / "rasters" / f"lilac_bloom_{y}.tif") as r:
            a = r.read(1).astype("float32")
            a[a == r.nodata] = np.nan
            A.append(a)
            tr, crs = r.transform, r.crs.to_string()
    A = np.stack(A)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, A=A, tr=np.array(tr)[:6], crs=crs)
    return A, tr, crs


def to_albers(arrays, tr, crs, shape):
    """Warp lat/lon grids to the Albers web/PNG grid (bilinear, NaN stays NaN)."""
    h, w = shape
    dst_tr, dw, dh = calculate_default_transform(crs, CRS, w, h, *rasterio.transform.array_bounds(h, w, tr),
                                                 resolution=RES)
    out = []
    for a in arrays:
        d = np.full((dh, dw), np.nan, "float32")
        reproject(a, d, src_transform=tr, src_crs=crs, dst_transform=dst_tr, dst_crs=CRS,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        out.append(d)
    return out, dst_tr


def doy_label(d):
    import datetime as dt
    t = dt.date(2021, 1, 1) + dt.timedelta(days=int(round(d)) - 1)   # non-leap year
    return t.strftime("%b %-d")


def observations(stack, tr):
    """One dot per Nature's Notebook site: median first open flowers over the springs it was watched.
    Keep onsets with a 'no' report at most 7 days earlier, so the first 'yes' is close to the real date."""
    rows = json.loads((DATA / "npn_lilac_individual.json").read_text())
    rows = [r for r in rows if 0 <= r["numdays_since_prior_no"] <= 7 and r["first_yes_doy"] > 30
            and r["first_yes_year"] in set(YEARS.tolist())]
    sites = {}
    for r in rows:
        sites.setdefault(r["site_id"], []).append(r)
    out = []
    for rs in sites.values():
        lat, lon = rs[0]["latitude"], rs[0]["longitude"]
        row, col = int((lat - tr.f) / tr.e), int((lon - tr.c) / tr.a)
        if not (0 <= row < stack.shape[1] and 0 <= col < stack.shape[2]):
            continue
        model = [stack[r["first_yes_year"] - YEARS[0], row, col] for r in rs]
        model = [m for m in model if not np.isnan(m)]
        if not model:
            continue
        out.append({"lat": lat, "lon": lon, "state": rs[0]["state"], "n": len({r["first_yes_year"] for r in rs}),
                    "obs": float(np.median([r["first_yes_doy"] for r in rs])), "model": float(np.mean(model))})
    g = gpd.GeoDataFrame(out, geometry=[Point(o["lon"], o["lat"]) for o in out], crs="EPSG:4269").to_crs(CRS)
    return g


def b64(a: np.ndarray) -> str:
    return base64.b64encode(a.astype("uint8").tobytes()).decode()


def main():
    S.apply()
    A, tr, crs = load_stack()
    normal = np.nanmean(A[(YEARS >= 1991) & (YEARS <= 2020)], 0)
    # high peaks miss bloom in some cold years; average the years they do bloom (cells that bloom in
    # under a third of years stay empty)
    full = (~np.isnan(A[(YEARS >= 1991) & (YEARS <= 2020)])).sum(0) >= 10
    normal[~full] = np.nan

    # headline check: in the East, how fast does the typical first bloom move north?
    rows, cols = np.indices(normal.shape)
    lat, lon = tr.f + (rows + .5) * tr.e, tr.c + (cols + .5) * tr.a
    lilac = full & (normal >= WARM)
    east = lilac & (lon > -100)
    days_per_deg = np.polyfit(lat[east], normal[east], 1)[0]
    mi_per_day = 69.0 / days_per_deg
    print(f"East of 100°W: {days_per_deg:.2f} days per degree north = {mi_per_day:.1f} miles/day")

    (n,), gtr = to_albers([normal], tr, crs, normal.shape)
    H, W = n.shape
    x0, y1 = gtr.c, gtr.f
    extent = (x0, x0 + W * RES, y1 - H * RES, y1)

    states = gpd.read_file(f"zip://{DATA / 'cb_2020_us_state_500k.zip'}")
    states = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])].to_crs(CRS)
    land = features.rasterize(((g, i + 1) for i, g in enumerate(states.geometry)), out_shape=(H, W),
                              transform=gtr, dtype="uint8")
    n[land == 0] = np.nan
    warm = n < WARM
    obs = observations(A, tr)

    # ---------------- static PNG ----------------
    season = S.make_scale(SEASON_STOPS)
    WARM_GREY = S.make_scale([(0, S.CLOSED), (1, S.BG)])(0.5)   # same mix the page uses
    cmap = ListedColormap([season(d) for d in range(D0, D1 + 1)])
    img = np.ma.masked_invalid(np.where(warm, np.nan, n))
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.8])
    ax.imshow(np.ma.masked_where(~warm, np.ones_like(n)), extent=extent, cmap=ListedColormap([WARM_GREY]),
              interpolation="nearest", zorder=1)
    ax.imshow(img, extent=extent, cmap=cmap, vmin=D0, vmax=D1, interpolation="bilinear", zorder=2)
    states.boundary.plot(ax=ax, color=S.EDGE, linewidth=0.5, zorder=3)
    states.dissolve().boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"], zorder=4)
    ax.set_xlim(extent[0] + 2e5, extent[1] - 1e5)
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    ax.set_aspect("equal")

    def at(la, lo):
        p = gpd.GeoSeries([Point(lo, la)], crs="EPSG:4326").to_crs(CRS).iloc[0]
        c, r = int((p.x - x0) / RES), int((y1 - p.y) / RES)
        return (p.x, p.y), float(n[r, c])
    # latest bloom in lilac country (high Rockies)
    nn = np.where(warm, np.nan, n)
    r_, c_ = np.unravel_index(np.nanargmax(nn), nn.shape)
    late_xy = (x0 + (c_ + .5) * RES, y1 - (r_ + .5) * RES)
    late_ll = gpd.GeoSeries([Point(late_xy)], crs=CRS).to_crs("EPSG:4326").iloc[0]
    print(f"latest: {doy_label(nn[r_, c_])} at {late_ll.y:.2f}, {late_ll.x:.2f}")
    atl_xy, atl = at(33.75, -84.39)
    ban_xy, ban = at(44.80, -68.77)
    for xy in (atl_xy, ban_xy, late_xy):
        ax.plot(*xy, "o", ms=3.2, color=S.INK, mec=S.BG, mew=0.8, zorder=6)
    S.callout(ax, atl_xy, (0.79, 0.12), f"Atlanta: {doy_label(atl)}", "lilacs open in late March", ha="left")
    S.callout(ax, ban_xy, (0.955, 0.93), f"Bangor, Maine: {doy_label(ban)}",
              "about 9 weeks after Atlanta", ha="right")
    S.callout(ax, late_xy, (0.24, 0.93), f"Latest: {doy_label(nn[r_, c_])}", LATE_PLACE, ha="left")

    fig.text(0.5, 0.955, TITLE, ha="center", va="top", fontsize=S.SIZE["title"], fontweight="bold",
             fontfamily=S.HEAD_FONT, color=S.INK)
    fig.text(0.5, 0.905, SUBTITLE, ha="center", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)

    cax = fig.add_axes([0.05, 0.12, 0.26, 0.016])
    cax.imshow(np.linspace(74, 196, 300)[None, :], aspect="auto", cmap=cmap, vmin=D0, vmax=D1,
               extent=(74, 196, 0, 1))
    cax.set_yticks([])
    ticks = [74, 105, 135, 166, 196]     # Mar 15, Apr 15, May 15, Jun 15, Jul 15
    cax.set_xticks(ticks)
    cax.set_xticklabels([doy_label(t) for t in ticks], fontsize=S.SIZE["legend"], color=S.INK)
    cax.tick_params(length=2, pad=2, color=S.MUTED)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.set_title("Typical date lilacs first open, 1991–2020", fontsize=S.SIZE["legend"] + 0.5,
                  fontweight="semibold", color=S.INK, loc="left", pad=5)
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=WARM_GREY, markersize=8,
                               label="Typical bloom before mid-March: mostly too warm for lilacs")],
               loc="upper left", bbox_to_anchor=(0.045, 0.098), frameon=False, fontsize=S.SIZE["legend"],
               labelcolor=S.INK, handletextpad=0.4, borderaxespad=0)
    S.credit(fig, f"Modeled first bloom of a cloned lilac from daily temperatures, 4 km, averaged 1991–2020  ·  "
                  f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)

    # ---------------- interactive page ----------------
    def enc(a, off=0):
        v = np.where(np.isnan(a), 0, np.clip(np.round(a) + off, 1, 255))
        return b64(v)
    st = states.reset_index(drop=True)
    st_geo = st[["NAME", "geometry"]].copy()
    st_geo["geometry"] = st_geo.geometry.simplify(2500)
    pts = [{"x": round((g.x - x0) / RES, 1), "y": round((y1 - g.y) / RES, 1), "state": o.state, "n": int(o.n),
            "obs": round(o.obs), "model": round(o.model)}
           for o, g in zip(obs.itertuples(), obs.geometry)]
    data = {
        "title": TITLE, "lede": LEDE, "day": DAY, "source": SOURCE, "sourceUrl": SOURCE_URL,
        "grid": {"w": W, "h": H, "x0": x0, "y1": y1, "res": RES,
                 "normal": enc(n), "state": b64(land)},
        "states": json.loads(st_geo.to_json(drop_id=True)), "stateNames": st.NAME.tolist(),
        "nation": json.loads(gpd.GeoSeries([st.geometry.union_all().simplify(2500)], crs=CRS).to_json())["features"][0]["geometry"],
        "sites": pts, "warm": WARM, "d0": D0, "d1": D1,
        "seasonStops": SEASON_STOPS,
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data) + ";")
    print(f"wrote out/map.png and out/map.html ({size / 1e6:.1f} MB); {len(pts)} observation sites")


TITLE = "The smell of lilacs moves north about 13 miles a day"
SUBTITLE = ("Typical date lilacs first open, 1991–2020. In the East, each degree north waits about five more days; "
            "mountains wait longer still.")
LEDE = ("When lilacs first open across the lower 48, modeled from daily temperatures. Press play to watch the bloom "
        "roll north and uphill. Dots are real lilacs watched by volunteers.")
LATE_PLACE = "on Montana's Beartooth Plateau"

if __name__ == "__main__":
    main()
