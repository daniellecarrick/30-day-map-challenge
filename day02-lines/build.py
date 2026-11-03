"""Day 2 · Lines — Which way does Citi Bike flow?

Every Citi Bike trip in September 2026 becomes a direction (straight line, start station ->
end station, reduced to a unit vector so short and long rides count equally). The city is
cut into ~400 m hexagons; each trip adds its direction to the hex where it starts AND the
hex where it ends, so a hex's summed vector is the direction traffic moves *through* it.
  - arrow angle  = main direction of travel
  - arrow length, width and color = size of that vector per day: net trips in that
    direction (trips heading opposite ways cancel out)
Three 3-hour windows: weekday 7-10am, weekday 4-7pm, weekend 1-4pm. Labor Day is excluded.

Writes:
  - out/citibike_flow.png    (static, for posting: three panels, streamlines)
  - out/citibike_flow.html   (interactive, self-contained: arrows, toggle between windows)
  - data/od_202609.csv.gz    (trips per station pair per window; cache so raw CSVs aren't needed)
  - data/stations_202609.csv

Data: Citi Bike System Data, https://citibikenyc.com/system-data (trip CSVs from
      s3.amazonaws.com/tripdata, unzipped into data/raw/). Borough Boundaries (NYC Open Data).
Run:  ../.venv/bin/python build.py
"""
import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 2: Lines"
SOURCE_URL = "https://citibikenyc.com/system-data"
CRS_M = "EPSG:32618"  # UTM 18N, metres (same as days 1 and 3)
MONTH = "2026-09"
HOLIDAYS = {"2026-09-07"}  # Labor Day: not a normal weekday
HEX_R = 230  # hex centre-to-corner, metres -> ~400 m across
MIN_TRIPS_PER_DAY = 8  # hexes with fewer trip ends per day are too noisy to draw

WINDOWS = {  # key -> (label, weekday?, hours [start, end))
    "am": ("Weekday mornings, 7–10am", True, (7, 10)),
    "pm": ("Weekday evenings, 4–7pm", True, (16, 19)),
    "we": ("Weekend afternoons, 1–4pm", False, (13, 16)),
}


# ---------------------------------------------------------------- trips -> station pairs
def build_od():
    """Read the raw trip CSVs once and reduce them to trips per (window, start, end) station."""
    cols = ["started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name",
            "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng"]
    ids = {"start_station_id": str, "end_station_id": str}
    files = sorted(glob.glob(str(RAW / f"{MONTH.replace('-', '')}-citibike-tripdata_*.csv")))
    if not files:
        sys.exit(f"No trip CSVs in {RAW}. Download {MONTH.replace('-', '')}-citibike-tripdata.zip "
                 "from https://s3.amazonaws.com/tripdata/ and unzip it there.")
    df = pd.concat(pd.read_csv(f, usecols=cols, dtype=ids) for f in files)
    n0 = len(df)
    for c in ("started_at", "ended_at"):
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    dur = (df.ended_at - df.started_at).dt.total_seconds()
    df = df[(df.started_at.dt.strftime("%Y-%m") == MONTH)
            & dur.between(60, 2 * 3600)
            & df.start_station_id.notna() & df.end_station_id.notna()
            & (df.start_station_id != df.end_station_id)]
    print(f"{n0:,} trips -> {len(df):,} after dropping round trips, missing ends, <1 min, >2 h")

    # one fixed location per station (e-bike trips carry GPS jitter around the dock)
    pts = pd.concat([
        df[["start_station_id", "start_station_name", "start_lat", "start_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1),
        df[["end_station_id", "end_station_name", "end_lat", "end_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1)])
    stations = pts.groupby("id").agg(name=("name", lambda s: s.mode().iat[0]),
                                     lat=("lat", "median"), lng=("lng", "median")).reset_index()
    stations.to_csv(DATA / f"stations_{MONTH.replace('-', '')}.csv", index=False)

    date = df.started_at.dt.strftime("%Y-%m-%d")
    weekday = (df.started_at.dt.dayofweek < 5) & ~date.isin(HOLIDAYS)
    weekend = df.started_at.dt.dayofweek >= 5
    hour = df.started_at.dt.hour
    parts = []
    for key, (_, wd, (h0, h1)) in WINDOWS.items():
        m = (weekday if wd else weekend) & hour.between(h0, h1 - 1)
        g = df[m].groupby(["start_station_id", "end_station_id"]).size().rename("trips").reset_index()
        g.insert(0, "window", key)
        g["days"] = date[m].nunique()
        parts.append(g)
    od = pd.concat(parts).rename(columns={"start_station_id": "a", "end_station_id": "b"})
    od.to_csv(DATA / f"od_{MONTH.replace('-', '')}.csv.gz", index=False)
    return od, stations


def load():
    tag = MONTH.replace("-", "")
    od_f, st_f = DATA / f"od_{tag}.csv.gz", DATA / f"stations_{tag}.csv"
    if od_f.exists() and st_f.exists():
        od = pd.read_csv(od_f, dtype={"a": str, "b": str})
        stations = pd.read_csv(st_f, dtype={"id": str})
    else:
        od, stations = build_od()
    st = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.lng, stations.lat),
                          crs="EPSG:4326").to_crs(CRS_M)
    stations["x"], stations["y"] = st.geometry.x.values, st.geometry.y.values
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    return od, stations.set_index("id"), boros


# ---------------------------------------------------------------- hex binning
SQ3 = np.sqrt(3)


def hex_of(x, y, R=HEX_R):
    """Pointy-top hex (axial q, r) containing each point."""
    q = (SQ3 / 3 * x - y / 3) / R
    r = (2 / 3 * y) / R
    # cube rounding
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


def hex_center(q, r, R=HEX_R):
    return R * SQ3 * (q + r / 2), R * 1.5 * r


def hex_polygon(x, y, R=HEX_R):
    from shapely.geometry import Polygon
    ang = np.radians(30 + 60 * np.arange(6))
    return Polygon(zip(x + R * np.cos(ang), y + R * np.sin(ang)))


COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]


def compass(vx, vy):
    """Map bearing as a word. The grid is UTM, so 'north' is true north (Manhattan's
    avenues run ~29° east of it)."""
    a = np.degrees(np.arctan2(vy, vx)) % 360
    return COMPASS[int(((a + 22.5) % 360) // 45)]


def flow(od, st):
    """Per window and hex: summed unit vectors (per day), trip ends, departures, arrivals."""
    od = od.join(st[["x", "y"]], on="a").join(st[["x", "y"]], on="b", rsuffix="_b").dropna().reset_index(drop=True)
    dx, dy = od.x_b - od.x, od.y_b - od.y
    d = np.hypot(dx, dy)
    od = od[d > 0].assign(ux=dx / d, uy=dy / d)
    od["qa"], od["ra"] = hex_of(od.x.values, od.y.values)
    od["qb"], od["rb"] = hex_of(od.x_b.values, od.y_b.values)

    out = []
    for key, g in od.groupby("window"):
        days = g.days.iat[0]
        w = g.trips
        start = pd.DataFrame({"q": g.qa, "r": g.ra, "vx": g.ux * w, "vy": g.uy * w, "n": w, "dep": w, "arr": 0})
        end = pd.DataFrame({"q": g.qb, "r": g.rb, "vx": g.ux * w, "vy": g.uy * w, "n": w, "dep": 0, "arr": w})
        h = pd.concat([start, end]).groupby(["q", "r"]).sum()
        h = h / days
        h["window"] = key
        h["days"] = days
        out.append(h.reset_index())
    h = pd.concat(out)
    h["mag"] = np.hypot(h.vx, h.vy)
    h["x"], h["y"] = hex_center(h.q, h.r)

    # name each hex after its busiest station
    st = st.copy()
    st["q"], st["r"] = hex_of(st.x.values, st.y.values)
    busy = pd.concat([od.groupby("a").trips.sum(), od.groupby("b").trips.sum()]).groupby(level=0).sum()
    st["busy"] = busy.reindex(st.index).fillna(0)
    names = st.sort_values("busy").groupby(["q", "r"]).name.last()
    h = h.join(names, on=["q", "r"])
    return h


# ---------------------------------------------------------------- findings
def findings(h, st, boros):
    """The numbers the headline and lede quote, computed so they stay literally true."""
    pts = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h.x, h.y), crs=CRS_M)
    h = h.assign(boro=gpd.sjoin_nearest(pts, boros, how="left").boroname.values)
    f = {}
    # morning vs evening: share of hexes (busy in both) whose arrows point >120° apart
    a = h[h.window == "am"].set_index(["q", "r"])
    p = h[h.window == "pm"].set_index(["q", "r"])
    j = a.join(p, rsuffix="_p", how="inner")
    cos = (j.vx * j.vx_p + j.vy * j.vy_p) / (j.mag * j.mag_p)
    f["flip_share"] = float((cos < -0.5).mean())
    f["flip_hexes"] = len(j)
    # Brooklyn mornings: share of arrows aimed at Midtown (within 45° of Grand Central)
    gc = st.loc[st.name == "Park Ave & E 42 St"].iloc[0]
    g = h[(h.window == "am") & (h.boro == "Brooklyn")]
    bx, by = gc.x - g.x, gc.y - g.y
    f["bk_midtown"] = float(((g.vx * bx + g.vy * by) / (g.mag * np.hypot(bx, by)) > np.cos(np.pi / 4)).mean())
    # extremes
    am = h[h.window == "am"]
    f["am_strongest"] = am.loc[am.mag.idxmax()]
    f["am_sink"] = am.loc[(am.arr - am.dep).idxmax()]
    pm = h[h.window == "pm"]
    f["pm_strongest"] = pm.loc[pm.mag.idxmax()]
    return f, h


# ---------------------------------------------------------------- smooth field for the PNG
def field(g, xs, ys, sigma=280, reach=380):
    """Gaussian-weighted average of hex vectors on a regular grid (Nadaraya–Watson).
    Cells farther than `reach` from any drawn hex are masked so lines stop at the edge."""
    X, Y = np.meshgrid(xs, ys)
    px, py = X.ravel(), Y.ravel()
    hx, hy = g.x.values, g.y.values
    U = np.zeros(px.size); V = np.zeros(px.size)
    W = np.zeros(px.size); near = np.full(px.size, np.inf)
    # (np.errstate: Apple's Accelerate BLAS raises spurious matmul warnings on underflowed weights)
    with np.errstate(all="ignore"):
        for s in range(0, px.size, 4000):
            d2 = (px[s:s + 4000, None] - hx) ** 2 + (py[s:s + 4000, None] - hy) ** 2
            w = np.exp(-d2 / (2 * sigma ** 2))
            W[s:s + 4000] = w.sum(1)
            U[s:s + 4000] = w @ g.vx.values
            V[s:s + 4000] = w @ g.vy.values
            near[s:s + 4000] = np.sqrt(d2.min(1))
    mask = (near > reach) | (W < 1e-6)
    W[W == 0] = 1
    shp = X.shape
    U, V = (np.ma.array((arr / W).reshape(shp), mask=mask.reshape(shp)) for arr in (U, V))
    return U, V


def draw_png(h, f, boros, title, lede, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    LW_REF = 100  # net trips/day that draws the widest, darkest line
    cmap = LinearSegmentedColormap.from_list("blue", S.SEQUENTIAL_BLUE[1:])
    norm = Normalize(0, LW_REF)
    ext = h[h.n >= MIN_TRIPS_PER_DAY]
    pad = 500
    x0, x1 = ext.x.min() - pad, ext.x.max() + pad
    y0, y1 = ext.y.min() - pad, ext.y.max() + pad
    xs, ys = np.arange(x0, x1, 90), np.arange(y0, y1, 90)
    land = boros.clip((x0 - 2000, y0 - 2000, x1 + 2000, y1 + 2000))

    fig = plt.figure(figsize=S.MAP["png_size_in"])
    S.title(fig, title)
    fig.text(0.5, 0.925, lede, ha="center", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED,
             linespacing=1.4)
    axes = {}
    for i, (key, (label, _, _)) in enumerate(WINDOWS.items()):
        ax = fig.add_axes([0.02 + i * 0.325, 0.065, 0.31, 0.79])
        axes[key] = ax
        g = h[(h.window == key) & (h.n >= MIN_TRIPS_PER_DAY)]
        land.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor="none", zorder=0)
        land.boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"] * 0.6, alpha=0.4, zorder=1)
        U, V = field(g, xs, ys)
        speed = np.hypot(U, V)
        lw = 0.35 + 2.05 * np.clip(speed / LW_REF, 0, 1)
        ax.streamplot(xs, ys, U, V, density=2.6, color=speed, cmap=cmap, norm=norm, linewidth=lw,
                      arrowsize=0.55, arrowstyle="-|>", minlength=0.08, maxlength=2.5,
                      broken_streamlines=True, zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.text(0.03, 0.995, label, transform=ax.transAxes, va="top", fontsize=S.SIZE["panel_title"],
                fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)
        ax.text(0.03, 0.962, f"{f['trips'][key]:,.0f} trips a day",
                transform=ax.transAxes, va="top", fontsize=S.SIZE["subtitle"], color=S.MUTED)

    # callouts: the strongest current in each rush, and the biggest morning sink
    def call(ax, row, xy, head, detail, ha="left"):
        S.callout(ax, (row.x, row.y), xy, head, detail, ha=ha, outline=hex_polygon(row.x, row.y), halo=True)

    r = f["am_sink"]
    call(axes["am"], r, (0.36, 0.56), "Grand Central: biggest bike sink",
         f"{r.arr - r.dep:.0f} more bikes arrive than leave\neach weekday morning")
    r = f["am_strongest"]
    call(axes["am"], r, (0.36, 0.33), "Strongest morning current",
         f"{r['name']}: {r.mag:.0f} trips a day,\nnet, heading {compass(r.vx, r.vy)}")
    r = f["pm_strongest"]
    call(axes["pm"], r, (0.03, 0.62), "Strongest evening current",
         f"{r['name']}: {r.mag:.0f} trips\na day, net, heading {compass(r.vx, r.vy)}")

    # legend: width and shade both = net trips a day (bottom right of the last panel: east Brooklyn, no lines)
    L = 0.85
    fig.text(L, 0.165, "Net trips a day\nin that direction", fontsize=S.SIZE["legend"], va="bottom",
             fontweight="semibold")
    wax = fig.add_axes([L, 0.09, 0.09, 0.065])
    wax.set_xlim(0, 10); wax.set_ylim(0, 3); wax.set_axis_off()
    for k, v in enumerate([10, 50, 100]):
        wax.plot([0.2, 3.2], [2.5 - k, 2.5 - k], color=cmap(norm(v)), lw=0.35 + 2.05 * min(v / LW_REF, 1),
                 solid_capstyle="butt")
        wax.text(3.8, 2.5 - k, f"{v}{'+' if v == LW_REF else ''}", va="center", fontsize=S.SIZE["legend"] - 0.5)

    S.credit(fig, "Each trip is a straight line from start to end station; its direction is added to the ~400 m "
             "hexagons at both ends and smoothed. September 2026, Labor Day excluded.  "
             "Data: Citi Bike System Data (citibikenyc.com/system-data)"
             f"  ·  #30DayMapChallenge {DAY}")
    fig.savefig(out, dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", out)


def main():
    S.apply()
    od, st, boros = load()
    h = flow(od, st)
    h = h[h.n >= MIN_TRIPS_PER_DAY]
    f, h = findings(h, st, boros)
    # all cleaned trips per average day in each window (not just those in drawn hexes)
    f["trips"] = {k: float(g.trips.sum() / g.days.iat[0]) for k, g in od.groupby("window")}
    for k, v in f.items():
        print(k, v if not hasattr(v, "name") else dict(v[["name", "mag", "arr", "dep", "vx", "vy"]]))

    flip = round(f["flip_share"] * 5)
    title = "Citi Bike has a daily tide: evening riders flow back the way the morning came"
    lede = (f"In {flip} of every 5 busy spots, the evening current runs roughly opposite the morning one. "
            f"In the morning, {100 * f['bk_midtown']:.0f}% of Brooklyn's arrows aim at Midtown.\n"
            "Weekend afternoons have no prevailing current: riders head every which way.")
    draw_png(h, f, boros, title, lede, OUT / "citibike_flow.png")

    # ---- interactive page: one record per hex, a [vx, vy, n, dep, arr] per window it's busy in ----
    pad = 800
    x0, y0, x1, y1 = h.x.min() - pad, h.y.min() - pad, h.x.max() + pad, h.y.max() + pad
    b = boros.clip((x0 - 8000, y0 - 8000, x1 + 8000, y1 + 8000))  # loose: land must run past the edges when zoomed
    b["geometry"] = b.geometry.simplify(20)
    hexes = []
    for (q, r), g in h.groupby(["q", "r"]):
        row = g.iloc[0]
        hexes.append({"x": round(row.x, 1), "y": round(row.y, 1), "name": row["name"],
                      "w": {w.window: [round(w.vx, 2), round(w.vy, 2), round(w.n, 1), round(w.dep, 1), round(w.arr, 1)]
                            for _, w in g.iterrows()}})
    web = {
        "title": title, "lede": lede.replace("\n", " "), "day": DAY, "source_url": SOURCE_URL,
        "windows": [{"key": k, "label": v[0], "days": int(h[h.window == k].days.iat[0]),
                     "trips": round(f["trips"][k])} for k, v in WINDOWS.items()],
        "hex_r": HEX_R, "ref": 80, "min_trips": MIN_TRIPS_PER_DAY,
        "blues": S.SEQUENTIAL_BLUE[1:],
        "hexes": hexes,
        "boroughs": json.loads(b.to_json(drop_id=True)),
        "extent": [x0, y0, x1, y1],
    }
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "citibike_flow.html")
    print("wrote", OUT / "citibike_flow.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
