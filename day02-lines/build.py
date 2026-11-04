"""Day 2 · Lines — Which way does Citi Bike flow?

Every Citi Bike trip in September 2026 becomes a direction (straight line, start station ->
end station, reduced to a unit vector so short and long rides count equally). The city is
cut into ~400 m hexagons; each trip adds its direction to the hex where it starts AND the
hex where it ends, so a hex's summed vector is the direction traffic moves *through* it.
  - arrow angle  = main direction of travel
  - arrow length, width and color = size of that vector per average day: net trips in that
    direction (trips heading opposite ways cancel out)
Labor Day is excluded.

Writes:
  - out/citibike_flow.html   (unpublished alternative: the arrows hour by hour through an
                              average weekday or weekend day, with play/pause and a timeline)
  - out/citibike_flow.png    (static, for posting: three 3-hour windows as streamlines:
                              weekday 7-10am, weekday 4-7pm, weekend 1-4pm)
  - out/linkedin_citibike_flow.png  (1080x1350 feed image: morning vs evening)
  - out/citibike_windmap.html (the published page: particle wind map, 15-minute steps, canvas)
  - data/hex_hourly_202609.csv.gz (hex sums per day type and start hour; cache for the page)
  - data/hex_15min_202609.csv.gz  (same per 15-minute slot; cache for the wind map)
Earlier interactive versions (three-window toggle; rolling 3-hour animation) are kept as
self-contained snapshots in experiments/.
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
MIN_TRIPS_PER_DAY = 8  # hexes with fewer trip ends per day are too noisy to draw (3-hour windows)
MIN_TRIPS_PER_HOUR = 4  # same idea for the one-hour frames on the page
REF_HOUR = 30  # net trips in an hour that draws the longest, darkest arrow on the page

WINDOWS = {  # key -> (label, weekday?, hours [start, end))
    "am": ("Weekday mornings, 7–10am", True, (7, 10)),
    "pm": ("Weekday evenings, 4–7pm", True, (16, 19)),
    "we": ("Weekend afternoons, 1–4pm", False, (13, 16)),
}


# ---------------------------------------------------------------- trips -> station pairs
def read_trips():
    """Read the raw trip CSVs and drop trips that carry no direction or look broken."""
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
    return df


def build_od():
    """Reduce the raw trips to trips per (window, start, end) station, plus station locations."""
    df = read_trips()

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


# ---------------------------------------------------------------- hour by hour (all-day page)
DAYTYPES = {"wd": "Weekdays", "we": "Weekends"}


def load_hourly(st):
    """Hex sums per (day type, start hour) for the whole month, cached in
    data/hex_hourly_YYYYMM.csv.gz (small, unlike station pairs x 24 hours).
    Day types: 'wd' = weekdays (Labor Day excluded), 'we' = weekends; `days` = how many."""
    f = DATA / f"hex_hourly_{MONTH.replace('-', '')}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    df = read_trips()
    date = df.started_at.dt.strftime("%Y-%m-%d")
    dow = df.started_at.dt.dayofweek
    df = df.assign(daytype=np.where(dow >= 5, "we", np.where(date.isin(HOLIDAYS), "", "wd")),
                   hour=df.started_at.dt.hour, date=date)
    df = df[df.daytype != ""]
    days = df.groupby("daytype").date.nunique()
    od = (df.groupby(["daytype", "hour", "start_station_id", "end_station_id"]).size().rename("trips")
            .reset_index().rename(columns={"start_station_id": "a", "end_station_id": "b"}))
    h = hex_sums(od, st, ["daytype", "hour"])
    h["days"] = h.daytype.map(days)
    h.round(3).to_csv(f, index=False)
    return h


SLOT_MIN = 15  # wind-map page: time step in minutes
# wind-map page frame (lon/lat): ~110th St to Prospect Heights, the Hudson to western Queens
WINDMAP_BBOX = (-74.028, 40.672, -73.855, 40.802)


def load_slots(st):
    """Hex sums per (day type, 15-minute slot of the day), cached in data/hex_15min_YYYYMM.csv.gz.
    slot = hour * 4 + minute // 15. Same day types and `days` as load_hourly."""
    f = DATA / f"hex_15min_{MONTH.replace('-', '')}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    df = read_trips()
    date = df.started_at.dt.strftime("%Y-%m-%d")
    dow = df.started_at.dt.dayofweek
    df = df.assign(daytype=np.where(dow >= 5, "we", np.where(date.isin(HOLIDAYS), "", "wd")),
                   slot=df.started_at.dt.hour * (60 // SLOT_MIN) + df.started_at.dt.minute // SLOT_MIN,
                   date=date)
    df = df[df.daytype != ""]
    days = df.groupby("daytype").date.nunique()
    od = (df.groupby(["daytype", "slot", "start_station_id", "end_station_id"]).size().rename("trips")
            .reset_index().rename(columns={"start_station_id": "a", "end_station_id": "b"}))
    h = hex_sums(od, st, ["daytype", "slot"])
    h["days"] = h.daytype.map(days)
    h.round(3).to_csv(f, index=False)
    return h


def hour_range(h0, n=1):
    """'7–8am', '11am–12pm', '12–1am' for trips starting h0:00 to (h0+n):00."""
    def part(h):
        h %= 24
        return (12 if h % 12 == 0 else h % 12), ("am" if h < 12 else "pm")
    (a, pa), (b, pb) = part(h0), part(h0 + n)
    return f"{a}–{b}{pb}" if pa == pb else f"{a}{pa}–{b}{pb}"


def hourly_frames(hh, n=1, min_trips=MIN_TRIPS_PER_HOUR):
    """One frame per start hour covering `n` hours (frame 8 with n=1 = trips starting 8–9am).
    Values per average day; quiet hexes dropped."""
    out = []
    for dt, g in hh.groupby("daytype"):
        days = g.days.iat[0]
        for i in range(24):
            hrs = [(i + k) % 24 for k in range(n)]
            f = g[g.hour.isin(hrs)].groupby(["q", "r"])[["vx", "vy", "n", "dep", "arr"]].sum() / days
            f = f[f.n >= min_trips].reset_index()
            out.append(f.assign(daytype=dt, frame=i))
    fr = pd.concat(out)
    fr["x"], fr["y"] = hex_center(fr.q, fr.r)
    fr["mag"] = np.hypot(fr.vx, fr.vy)
    return fr


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


def hex_sums(od, st, keys):
    """Each trip (station pair x trips) adds its unit direction to the hex at both ends.
    Returns summed vx, vy, trip ends n, departures, arrivals per `keys` + hex (q, r)."""
    od = od.join(st[["x", "y"]], on="a").join(st[["x", "y"]], on="b", rsuffix="_b").dropna().reset_index(drop=True)
    dx, dy = od.x_b - od.x, od.y_b - od.y
    d = np.hypot(dx, dy)
    od = od[d > 0].assign(ux=dx / d, uy=dy / d)
    qa, ra = hex_of(od.x.values, od.y.values)
    qb, rb = hex_of(od.x_b.values, od.y_b.values)
    w = od.trips
    start = od[keys].assign(q=qa, r=ra, vx=od.ux * w, vy=od.uy * w, n=w, dep=w, arr=0)
    end = od[keys].assign(q=qb, r=rb, vx=od.ux * w, vy=od.uy * w, n=w, dep=0, arr=w)
    return pd.concat([start, end]).groupby(keys + ["q", "r"]).sum().reset_index()


def hex_names(od, st):
    """Name each hex after its busiest station (busy = trips in `od`)."""
    st = st.copy()
    st["q"], st["r"] = hex_of(st.x.values, st.y.values)
    busy = pd.concat([od.groupby("a").trips.sum(), od.groupby("b").trips.sum()]).groupby(level=0).sum()
    st["busy"] = busy.reindex(st.index).fillna(0)
    return st.sort_values("busy").groupby(["q", "r"]).name.last()


def flow(od, st):
    """Per window and hex: summed unit vectors (per day), trip ends, departures, arrivals."""
    h = hex_sums(od, st, ["window"])
    days = od.groupby("window").days.first()
    h["days"] = h.window.map(days)
    for c in ("vx", "vy", "n", "dep", "arr"):
        h[c] = h[c] / h.days
    h["mag"] = np.hypot(h.vx, h.vy)
    h["x"], h["y"] = hex_center(h.q, h.r)
    return h.join(hex_names(od, st), on=["q", "r"])


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


def social_card(h, f, boros, out):
    """1080x1350 (4:5) image for LinkedIn/Instagram feeds: morning vs evening side by side,
    cropped to the busy core, type sized for a phone."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    W_IN, H_IN, DPI = 5.4, 6.75, 200
    LW_REF = 100
    cmap = LinearSegmentedColormap.from_list("blue", S.SEQUENTIAL_BLUE[1:])
    norm = Normalize(0, LW_REF)
    # crop: centred on the morning-flow-weighted centre, nudged north to keep Midtown + Harlem in frame
    am = h[h.window == "am"]
    cx = np.average(am.x, weights=am.mag)
    cy = np.average(am.y, weights=am.mag) + 2200
    HW, HH = 4300, 7900
    x0, x1, y0, y1 = cx - HW, cx + HW, cy - HH, cy + HH
    xs, ys = np.arange(x0 - 1000, x1 + 1000, 70), np.arange(y0 - 1000, y1 + 1000, 70)
    land = boros.clip((x0 - 3000, y0 - 3000, x1 + 3000, y1 + 3000))

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.text(0.06, 0.955, "Citi Bike has a daily tide", ha="left", va="top", fontsize=19, fontweight="bold",
             color=S.INK, fontfamily=S.HEAD_FONT)
    fig.text(0.06, 0.898, "Every weekday morning, riders pour toward Midtown. By evening,\n"
             f"the current runs back the way it came: in {round(f['flip_share'] * 5)} of every 5 busy spots,\n"
             "the evening flow points roughly opposite the morning one.",
             ha="left", va="top", fontsize=8.3, color=S.MUTED, linespacing=1.35)

    panels = [("am", "Weekday mornings, 7–10am"), ("pm", "Weekday evenings, 4–7pm")]
    for i, (key, label) in enumerate(panels):
        ax = fig.add_axes([0.03 + i * 0.48, 0.155, 0.46, 0.635])
        g = h[h.window == key]
        land.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor="none", zorder=0)
        land.boundary.plot(ax=ax, color=S.INK, linewidth=0.3, alpha=0.4, zorder=1)
        U, V = field(g, xs, ys)
        speed = np.hypot(U, V)
        ax.streamplot(xs, ys, U, V, density=1.7, color=speed, cmap=cmap, norm=norm,
                      linewidth=0.3 + 1.6 * np.clip(speed / LW_REF, 0, 1), arrowsize=0.5,
                      arrowstyle="-|>", minlength=0.08, maxlength=3, broken_streamlines=True, zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_axis_off()
        if key == "am":  # the extreme: biggest morning sink
            r = f["am_sink"]
            S.callout(ax, (r.x, r.y), (0.49, 0.935), "Grand Central",
                      f"{r.arr - r.dep:.0f} more bikes in\nthan out each morning",
                      outline=hex_polygon(r.x, r.y), halo=True)
        fig.text(0.03 + i * 0.48 + 0.02, 0.80, label, fontsize=10, fontweight="semibold",
                 fontfamily=S.HEAD_FONT, color=S.INK, va="bottom")

    # width/shade key, one row along the bottom
    fig.text(0.06, 0.118, "Net trips a day in that direction", fontsize=7.5, color=S.INK, fontweight="semibold",
             va="bottom")
    kax = fig.add_axes([0.06, 0.09, 0.6, 0.022])
    kax.set_xlim(0, 30); kax.set_ylim(-1, 1); kax.set_axis_off()
    for k, v in enumerate([10, 50, 100]):
        x = k * 8
        kax.plot([x, x + 3.2], [0, 0], color=cmap(norm(v)), lw=0.3 + 1.6 * min(v / LW_REF, 1), solid_capstyle="butt")
        kax.text(x + 3.8, 0, f"{v}{'+' if v == LW_REF else ''}", va="center", fontsize=7.5, color=S.INK)

    fig.text(0.06, 0.018, "Every Citi Bike trip, Sept 2026 (Labor Day excluded), as a straight line from start to end\n"
             "station; directions summed per ~400 m area and smoothed. Data: Citi Bike System Data\n"
             "#30DayMapChallenge Day 2: Lines  ·  daniellecarrick.github.io/30-day-map-challenge",
             ha="left", va="bottom", fontsize=6.3, color=S.MUTED, linespacing=1.4)
    fig.savefig(out, dpi=DPI)
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
    social_card(h, f, boros, OUT / "linkedin_citibike_flow.png")

    summary = web_page(od, st, boros)
    windmap_page(od, st, boros, summary)


def web_page(od, st, boros):
    """The interactive page: arrows hour by hour through an average weekday / weekend day."""
    fr = hourly_frames(load_hourly(st))
    # headline numbers: pooled net flow toward Grand Central, per frame (share of trip ends)
    gc = st.loc[st.name == "Park Ave & E 42 St"].iloc[0]
    bx, by = gc.x - fr.x, gc.y - fr.y
    fr["toward"] = (fr.vx * bx + fr.vy * by) / np.hypot(bx, by)
    pull = {dt: [float(g[g.frame == i].toward.sum() / g[g.frame == i].n.sum()) for i in range(24)]
            for dt, g in fr.groupby("daytype")}
    wd = pull["wd"]
    peak = int(np.argmax(wd))
    turn = next(i for i in range(peak, peak + 24) if wd[i % 24] < 0) % 24  # first window pointing away
    in_frames = [i for i in range(24) if wd[i] > 0]
    ratio = max(pull["we"]) / max(wd)
    print("pull toward Midtown, weekdays:", [round(100 * v, 1) for v in wd])
    print(f"peak {hour_range(peak)}, turns {hour_range(turn)}, weekend/weekday peak {ratio:.2f}, "
          f"toward-Midtown hours {in_frames}")
    print("net trips/hour quantiles:", fr.mag.quantile([.5, .9, .98, 1]).round(1).tolist())
    turn_clock = f"{turn % 12 or 12}{'am' if turn < 12 else 'pm'}"
    title = f"Every weekday, Citi Bike flows toward Midtown until about {turn_clock}, then turns around"
    lede = (f"Each frame is one hour of trips on an average day. On weekdays the net flow points toward Midtown from early "
            f"morning until about {turn_clock}, strongest {hour_range(peak)}, then away from it through the evening "
            f"and night. Weekends follow the same shape, "
            f"{'about half' if 0.4 <= ratio <= 0.6 else f'{100 * ratio:.0f}%'} as strong.")
    assert in_frames == list(range(in_frames[0], in_frames[-1] + 1)) and in_frames[0] <= 5, in_frames  # 'early morning' holds

    names = hex_names(od, st)
    # compact: hex grid position (q, r) instead of coordinates; whole trips per day per frame
    hexes = []
    for (q, r), g in fr.groupby(["q", "r"]):
        rec = {"q": int(q), "r": int(r), "name": names.get((q, r), "")}
        for dt in DAYTYPES:
            arr = [0] * 24  # 0 = fewer than MIN_TRIPS_PER_HOUR trips that hour
            for _, w in g[g.daytype == dt].iterrows():
                arr[int(w.frame)] = [int(round(v)) for v in (w.vx, w.vy, w.n, w.arr - w.dep)]
            rec[dt] = arr
        hexes.append(rec)
    hh = load_hourly(st)
    trips_by_hour = {dt: [round(float(g[g.hour == i].dep.sum() / g.days.iat[0])) for i in range(24)]
                     for dt, g in hh.groupby("daytype")}
    pad = 800
    x0, y0, x1, y1 = fr.x.min() - pad, fr.y.min() - pad, fr.x.max() + pad, fr.y.max() + pad
    b = boros.clip((x0 - 5000, y0 - 5000, x1 + 5000, y1 + 5000))
    b["geometry"] = b.geometry.simplify(35)
    web = {
        "title": title, "lede": lede, "day": DAY, "source_url": SOURCE_URL,
        "daytypes": [{"key": k, "label": v} for k, v in DAYTYPES.items()],
        "labels": [hour_range(i) for i in range(24)], "trips_by_hour": trips_by_hour,
        "hex_r": HEX_R, "ref": REF_HOUR, "keys": [5, 15, REF_HOUR], "min_trips": MIN_TRIPS_PER_HOUR,
        "blues": S.SEQUENTIAL_BLUE[1:],
        "hexes": hexes, "boroughs": json.loads(b.to_json(drop_id=True)), "extent": [x0, y0, x1, y1],
    }
    # data.js lets web/index.html open straight from disk while editing (gitignored)
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "citibike_flow.html")
    print("wrote", OUT / "citibike_flow.html", f"({size / 1e6:.2f} MB)")
    return {"title": title, "turn_clock": turn_clock, "peak": peak, "ratio": ratio}


def windmap_page(od, st, boros, summary):
    """Experiment: particle wind map. 15-minute slots of net flow per hex, as a rate per hour on an
    average day; the page smooths them into a field and lets particles drift along it."""
    import base64
    sl = load_slots(st)
    per_hour = 60 // SLOT_MIN
    nslot = 24 * per_hour
    sl = sl.assign(vx=sl.vx / sl.days * per_hour, vy=sl.vy / sl.days * per_hour, n=sl.n / sl.days * per_hour)
    sl = sl[sl.n >= MIN_TRIPS_PER_HOUR]  # rate per hour, same threshold as the hourly page
    # frame: a fixed box around the busy core; hexes outside it are dropped
    lo0, la0, lo1, la1 = WINDMAP_BBOX
    corners = gpd.GeoSeries(gpd.points_from_xy([lo0, lo0, lo1, lo1], [la0, la1, la0, la1]), crs="EPSG:4326").to_crs(CRS_M)
    x0, x1 = corners.x.min(), corners.x.max()
    y0, y1 = corners.y.min(), corners.y.max()
    cx, cy = hex_center(sl.q, sl.r)
    sl = sl[(cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)]
    keep = sl.groupby(["q", "r"]).size().index
    names = hex_names(od, st)
    hexes = [{"q": int(q), "r": int(r), "name": names.get((q, r), "")} for q, r in keep]
    idx = {k: i for i, k in enumerate(keep)}
    flow = {}
    for dt in DAYTYPES:
        a = np.zeros((nslot, len(keep), 2), dtype=np.int8)  # net trips/hour, clamped to int8
        g = sl[sl.daytype == dt]
        hi = np.array([idx[k] for k in zip(g.q, g.r)])
        a[g.slot.values, hi, 0] = np.clip(np.round(g.vx.values), -127, 127)
        a[g.slot.values, hi, 1] = np.clip(np.round(g.vy.values), -127, 127)
        flow[dt] = base64.b64encode(a.tobytes()).decode()
    print("wind map: hexes", len(keep), "| |v| quantiles", np.hypot(sl.vx, sl.vy).quantile([.5, .9, .99]).round(1).tolist())
    # trips started per hour (rate) for each 15-minute slot, for the timeline
    trips = {dt: [round(float(g[g.slot == i].dep.sum() / g.days.iat[0] * per_hour)) for i in range(nslot)]
             for dt, g in load_slots(st).groupby("daytype")}
    b = boros.clip((x0, y0, x1, y1))  # land stops at the frame
    b["geometry"] = b.geometry.simplify(25)
    print(f"wind map frame: {(x1 - x0) / 1000:.1f} x {(y1 - y0) / 1000:.1f} km")
    t = summary["turn_clock"]
    web = {
        "title": summary["title"],
        "lede": (f"Each streak drifts with the net flow of bike trips. Faster and darker streaks represent more trips. "
                 f"On weekdays the current runs toward Midtown from early morning until about {t}, "
                 f"strongest {hour_range(summary['peak'])}, then back out through the evening and night."),
        "day": DAY, "source_url": SOURCE_URL,
        "daytypes": [{"key": k, "label": v} for k, v in DAYTYPES.items()],
        "slot_min": SLOT_MIN, "trips": trips, "hex_r": HEX_R, "ref": REF_HOUR, "min_trips": MIN_TRIPS_PER_HOUR,
        "blues": S.SEQUENTIAL_BLUE[1:], "hexes": hexes, "flow": flow,
        "boroughs": json.loads(b.to_json(drop_id=True)), "extent": [x0, y0, x1, y1],
    }
    (HERE / "web" / "windmap_data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "windmap.html", web, OUT / "citibike_windmap.html", data_src="windmap_data.js")
    print("wrote", OUT / "citibike_windmap.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
